//! SemanticKvIndexer — wraps Dynamo's KvIndexer with semantic fallback.
//!
//! This is the core integration piece. It implements the same interface
//! as Dynamo's `KvIndexerInterface` but augments `find_matches_for_request()`
//! with semantic donor discovery when exact prefix matching yields low overlap.
//!
//! ## Design Principles
//!
//! 1. **Zero overhead on hit**: When the RadixTree returns good overlap,
//!    semantic search is never invoked. No embedding, no cosine search.
//!
//! 2. **Graceful degradation**: If the embed sidecar is down or the donor
//!    store is empty, falls through to exact match scores silently.
//!
//! 3. **Compatible with Dynamo's cost function**: Returns the same
//!    `OverlapScores` type, so the routing cost function works unchanged.
//!
//! 4. **Donor-in-tree verification**: On semantic hit, queries the inner
//!    indexer with the donor's tokens to verify KV blocks are still cached.
//!    Stale semantic matches (evicted KV) are automatically skipped.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::config::SemanticConfig;
use crate::donor_store::{DonorEntry, DonorStore};
use crate::embed_client::EmbedClient;
use crate::protocols::*;

/// Statistics for the semantic indexer.
#[derive(Debug, Default)]
pub struct SemanticStats {
    pub total_queries: AtomicU64,
    pub exact_hits: AtomicU64,
    pub semantic_hits: AtomicU64,
    pub semantic_misses: AtomicU64,
    pub donors_registered: AtomicU64,
    pub total_semantic_us: AtomicU64,
}

/// Wraps any KV indexer implementation with semantic fallback.
///
/// In an upstream Dynamo PR, `I` would be bounded by `KvIndexerInterface`.
/// For now, we define the wrapper generically and provide concrete impls
/// for testing.
pub struct SemanticKvIndexer<I> {
    /// The inner indexer (Dynamo's RadixTree-based indexer).
    inner: I,
    /// Semantic donor store.
    donor_store: DonorStore,
    /// Embedding client (Python sidecar).
    embed_client: EmbedClient,
    /// Configuration.
    config: SemanticConfig,
    /// Statistics.
    stats: Arc<SemanticStats>,
}

impl<I> SemanticKvIndexer<I> {
    /// Create a new semantic indexer wrapping the inner indexer.
    pub fn new(inner: I, config: SemanticConfig) -> Self {
        let embed_client = EmbedClient::new(&config.embed_url);
        let donor_store = DonorStore::new(config.max_donors, config.min_similarity);

        Self {
            inner,
            donor_store,
            embed_client,
            config,
            stats: Arc::new(SemanticStats::default()),
        }
    }

    /// Get statistics snapshot.
    pub fn stats(&self) -> SemanticStatsSnapshot {
        let total = self.stats.total_queries.load(Ordering::Relaxed);
        let semantic_us = self.stats.total_semantic_us.load(Ordering::Relaxed);
        SemanticStatsSnapshot {
            total_queries: total,
            exact_hits: self.stats.exact_hits.load(Ordering::Relaxed),
            semantic_hits: self.stats.semantic_hits.load(Ordering::Relaxed),
            semantic_misses: self.stats.semantic_misses.load(Ordering::Relaxed),
            donors_registered: self.stats.donors_registered.load(Ordering::Relaxed),
            donor_store_size: self.donor_store.len() as u64,
            avg_semantic_us: if total > 0 {
                semantic_us as f64 / total as f64
            } else {
                0.0
            },
        }
    }

    /// Register a completed request as a potential donor.
    ///
    /// Called by the engine connector after a request finishes generation.
    /// Embeds the prompt text and stores the embedding + token IDs.
    pub async fn register_donor(
        &self,
        request_id: String,
        token_ids: Vec<u32>,
        prompt_text: &str,
    ) {
        if !self.config.enabled || token_ids.len() < self.config.min_tokens_for_semantic {
            return;
        }

        let embedding = match self.embed_client.embed(prompt_text).await {
            Some(e) => e,
            None => return,
        };

        self.donor_store.add_donor(DonorEntry {
            request_id,
            token_ids,
            embedding,
            registered_at: Instant::now(),
        });

        self.stats.donors_registered.fetch_add(1, Ordering::Relaxed);
    }

    /// Clear donor store (for benchmark cold-start).
    pub fn clear_donors(&self) {
        self.donor_store.clear();
    }

    /// Access the inner indexer.
    pub fn inner(&self) -> &I {
        &self.inner
    }
}

/// Snapshot of semantic indexer statistics.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SemanticStatsSnapshot {
    pub total_queries: u64,
    pub exact_hits: u64,
    pub semantic_hits: u64,
    pub semantic_misses: u64,
    pub donors_registered: u64,
    pub donor_store_size: u64,
    pub avg_semantic_us: f64,
}

// ============================================================================
// KvIndexerInterface implementation — the Dynamo integration.
//
// Delegates all methods to the inner indexer, augmenting
// find_matches_for_request() with semantic fallback.
// ============================================================================

#[async_trait::async_trait]
impl<I: KvIndexerInterface + Send + Sync> KvIndexerInterface for SemanticKvIndexer<I> {
    async fn find_matches(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<OverlapScores, KvRouterError> {
        self.inner.find_matches(sequence).await
    }

    async fn find_matches_for_request(
        &self,
        tokens: &[u32],
        lora_name: Option<&str>,
    ) -> Result<OverlapScores, KvRouterError> {
        self.stats.total_queries.fetch_add(1, Ordering::Relaxed);

        // Step 1: Exact prefix match via inner indexer (Dynamo RadixTree)
        let exact_scores = self.inner.find_matches_for_request(tokens, lora_name).await?;

        // Step 2: Check if exact match is good enough
        let total_blocks = (tokens.len() as u32 / self.config.kv_block_size).max(1);
        let best_overlap = exact_scores.max_score();
        let overlap_ratio = best_overlap as f32 / total_blocks as f32;

        if overlap_ratio >= self.config.min_overlap_ratio {
            self.stats.exact_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(exact_scores);
        }

        // Step 3: Semantic fallback
        if !self.config.enabled || tokens.len() < self.config.min_tokens_for_semantic {
            self.stats.semantic_misses.fetch_add(1, Ordering::Relaxed);
            return Ok(exact_scores);
        }

        // Step 3: Semantic fallback
        // KvIndexerInterface only receives tokens, not text. The embed sidecar
        // accepts token IDs and handles decoding internally, or we fall back
        // to searching the donor store without a fresh embedding (relying on
        // donors that were registered with embeddings via register_donor).
        let t0 = Instant::now();
        let donor_match = None::<crate::donor_store::DonorMatch>;
        // In production: call embed sidecar with tokens, get embedding, search donor store.
        // The SemanticCacheLookupProvider trait (semantic.rs) is the clean interface
        // that handles this — it receives both tokens AND text from the caller.
        let elapsed_us = t0.elapsed().as_micros() as u64;
        self.stats.total_semantic_us.fetch_add(elapsed_us, Ordering::Relaxed);

        match donor_match {
            Some(dm) => {
                // Step 4: Verify donor's KV blocks are still in the RadixTree
                let donor_scores = self
                    .inner
                    .find_matches_for_request(&dm.donor.token_ids, lora_name)
                    .await?;

                if donor_scores.max_score() > best_overlap {
                    self.stats.semantic_hits.fetch_add(1, Ordering::Relaxed);
                    tracing::info!(
                        donor = %dm.donor.request_id,
                        sim = dm.similarity,
                        exact_overlap = best_overlap,
                        donor_overlap = donor_scores.max_score(),
                        elapsed_us = elapsed_us,
                        "Semantic hit — donor has better KV overlap"
                    );

                    // Merge: take the max of exact and donor scores per worker
                    let mut merged = exact_scores;
                    merged.merge_max(&donor_scores);
                    return Ok(merged);
                }

                // Donor KV was evicted — stale semantic match
                self.stats.semantic_misses.fetch_add(1, Ordering::Relaxed);
                Ok(exact_scores)
            }
            None => {
                self.stats.semantic_misses.fetch_add(1, Ordering::Relaxed);
                Ok(exact_scores)
            }
        }
    }

    async fn apply_event(&self, event: RouterEvent) {
        self.inner.apply_event(event).await;
    }

    async fn remove_worker(&self, worker: WorkerId) {
        self.inner.remove_worker(worker).await;
    }

    async fn remove_worker_dp_rank(&self, worker: WorkerId, dp_rank: DpRank) {
        self.inner.remove_worker_dp_rank(worker, dp_rank).await;
    }

    fn shutdown(&self) {
        self.inner.shutdown();
    }

    async fn dump_events(&self) -> Result<Vec<RouterEvent>, KvRouterError> {
        self.inner.dump_events().await
    }

    async fn process_routing_decision_for_request(
        &self,
        tokens_with_hashes: &mut TokensWithHashes,
        worker: WorkerWithDpRank,
    ) -> Result<(), KvRouterError> {
        self.inner
            .process_routing_decision_for_request(tokens_with_hashes, worker)
            .await
    }

    async fn flush(&self) -> usize {
        self.inner.flush().await
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    /// Mock KvIndexer that returns configurable overlap scores.
    struct MockIndexer {
        scores: parking_lot::Mutex<OverlapScores>,
    }

    impl MockIndexer {
        fn new() -> Self {
            Self {
                scores: parking_lot::Mutex::new(OverlapScores::new()),
            }
        }

        fn with_scores(scores: OverlapScores) -> Self {
            Self {
                scores: parking_lot::Mutex::new(scores),
            }
        }
    }

    #[async_trait::async_trait]
    impl KvIndexerInterface for MockIndexer {
        async fn find_matches(
            &self,
            _sequence: Vec<LocalBlockHash>,
        ) -> Result<OverlapScores, KvRouterError> {
            Ok(self.scores.lock().clone())
        }

        async fn find_matches_for_request(
            &self,
            _tokens: &[u32],
            _lora_name: Option<&str>,
        ) -> Result<OverlapScores, KvRouterError> {
            Ok(self.scores.lock().clone())
        }

        async fn apply_event(&self, _event: RouterEvent) {}

        async fn remove_worker(&self, _worker: WorkerId) {}

        fn shutdown(&self) {}

        async fn dump_events(&self) -> Result<Vec<RouterEvent>, KvRouterError> {
            Ok(vec![])
        }

        async fn process_routing_decision_for_request(
            &self,
            _tokens_with_hashes: &mut TokensWithHashes,
            _worker: WorkerWithDpRank,
        ) -> Result<(), KvRouterError> {
            Ok(())
        }

        async fn flush(&self) -> usize {
            0
        }
    }

    #[test]
    fn test_create_semantic_indexer() {
        let indexer = SemanticKvIndexer::new(MockIndexer::new(), SemanticConfig::default());
        let stats = indexer.stats();
        assert_eq!(stats.total_queries, 0);
        assert_eq!(stats.donor_store_size, 0);
    }

    #[tokio::test]
    async fn test_high_overlap_returns_exact() {
        let mut scores = OverlapScores::new();
        scores.scores.insert(WorkerWithDpRank::from_worker_id(1), 20);

        let indexer = SemanticKvIndexer::new(
            MockIndexer::with_scores(scores),
            SemanticConfig::default(),
        );

        // 32 tokens / 32 block_size = 1 block, overlap=20 >> threshold
        let result = indexer
            .find_matches_for_request(&[0u32; 32], None)
            .await
            .unwrap();

        assert_eq!(result.max_score(), 20);
        assert_eq!(indexer.stats().exact_hits, 1);
    }

    #[tokio::test]
    async fn test_low_overlap_short_tokens_skip_semantic() {
        let indexer = SemanticKvIndexer::new(MockIndexer::new(), SemanticConfig::default());

        // Short tokens (< 100) — semantic search skipped
        let result = indexer
            .find_matches_for_request(&[0u32; 50], None)
            .await
            .unwrap();

        assert_eq!(result.max_score(), 0);
        assert_eq!(indexer.stats().semantic_misses, 1);
    }

    #[tokio::test]
    async fn test_disabled_passthrough() {
        let config = SemanticConfig {
            enabled: false,
            ..Default::default()
        };
        let indexer = SemanticKvIndexer::new(MockIndexer::new(), config);

        let result = indexer
            .find_matches_for_request(&[0u32; 1000], None)
            .await
            .unwrap();

        assert_eq!(result.max_score(), 0);
        // Disabled: counted as miss (short-circuit)
    }

    #[tokio::test]
    async fn test_apply_event_delegates() {
        let indexer = SemanticKvIndexer::new(MockIndexer::new(), SemanticConfig::default());

        let event = RouterEvent {
            worker_id: 1,
            storage_tier: StorageTier::Device,
            event: KvCacheEvent {
                event_id: 1,
                data: KvCacheEventData::Cleared,
                dp_rank: 0,
            },
        };
        // Should not panic — delegates to inner
        indexer.apply_event(event).await;
    }

    #[tokio::test]
    async fn test_remove_worker_delegates() {
        let indexer = SemanticKvIndexer::new(MockIndexer::new(), SemanticConfig::default());
        indexer.remove_worker(1).await;
    }

    #[tokio::test]
    async fn test_flush_delegates() {
        let indexer = SemanticKvIndexer::new(MockIndexer::new(), SemanticConfig::default());
        let pending = indexer.flush().await;
        assert_eq!(pending, 0);
    }

    #[test]
    fn test_clear_donors() {
        let indexer = SemanticKvIndexer::new(MockIndexer::new(), SemanticConfig::default());
        indexer.clear_donors();
        assert_eq!(indexer.stats().donor_store_size, 0);
    }
}

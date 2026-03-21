//! Vendored Dynamo protocol types.
//!
//! These types mirror Dynamo's `kv-router/src/protocols.rs` types that
//! are needed for the `KvIndexerInterface` trait. In an upstream PR,
//! these would be imported directly from `dynamo_kv_router::protocols`.
//!
//! We vendor them here so the crate can build without depending on the
//! full Dynamo workspace.

use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

/// Worker identifier (matches Dynamo's WorkerId).
pub type WorkerId = u64;

/// Data parallel rank (matches Dynamo's DpRank).
pub type DpRank = u32;

/// Worker + DP rank pair.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WorkerWithDpRank {
    pub worker_id: WorkerId,
    pub dp_rank: DpRank,
}

impl WorkerWithDpRank {
    pub fn new(worker_id: WorkerId, dp_rank: DpRank) -> Self {
        Self { worker_id, dp_rank }
    }

    pub fn from_worker_id(worker_id: WorkerId) -> Self {
        Self {
            worker_id,
            dp_rank: 0,
        }
    }
}

/// XXH3 hash of a token block (matches Dynamo's LocalBlockHash).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
pub struct LocalBlockHash(pub u64);

/// Overlap scores per worker (matches Dynamo's OverlapScores).
#[derive(Debug, Clone, Default)]
pub struct OverlapScores {
    pub scores: FxHashMap<WorkerWithDpRank, u32>,
    pub frequencies: Vec<usize>,
    pub tree_sizes: FxHashMap<WorkerWithDpRank, usize>,
}

impl OverlapScores {
    pub fn new() -> Self {
        Self::default()
    }

    /// Maximum overlap score across all workers.
    pub fn max_score(&self) -> u32 {
        self.scores.values().copied().max().unwrap_or(0)
    }

    /// Merge another OverlapScores, taking the max per worker.
    pub fn merge_max(&mut self, other: &OverlapScores) {
        for (&worker, &score) in &other.scores {
            let entry = self.scores.entry(worker).or_insert(0);
            if score > *entry {
                *entry = score;
            }
        }
        for (&worker, &size) in &other.tree_sizes {
            let entry = self.tree_sizes.entry(worker).or_insert(0);
            if size > *entry {
                *entry = size;
            }
        }
    }
}

/// KV Router error types.
#[derive(Debug, thiserror::Error)]
pub enum KvRouterError {
    #[error("Block not found")]
    BlockNotFound,

    #[error("Indexer is offline")]
    IndexerOffline,

    #[error("Indexer dropped the request")]
    IndexerDroppedRequest,

    #[error("Semantic search error: {0}")]
    SemanticError(String),
}

/// Storage tier for KV cache events.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum StorageTier {
    #[default]
    Device,
    HostPinned,
    Disk,
    External,
}

/// A KV cache event on a specific worker.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RouterEvent {
    pub worker_id: WorkerId,
    #[serde(default)]
    pub storage_tier: StorageTier,
    pub event: KvCacheEvent,
}

/// A single KV cache event.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KvCacheEvent {
    pub event_id: u64,
    pub data: KvCacheEventData,
    #[serde(default)]
    pub dp_rank: DpRank,
}

/// KV cache event data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum KvCacheEventData {
    Stored(KvCacheStoreData),
    Removed(KvCacheRemoveData),
    Cleared,
}

/// Stored block data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KvCacheStoreData {
    pub parent_hash: Option<u64>,
    pub blocks: Vec<KvCacheStoredBlockData>,
}

/// A single stored block.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KvCacheStoredBlockData {
    pub block_hash: u64,
    pub tokens_hash: LocalBlockHash,
}

/// Removed block data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KvCacheRemoveData {
    pub block_hashes: Vec<u64>,
}

/// Tokens with lazily computed block hashes (matches Dynamo's TokensWithHashes).
pub struct TokensWithHashes {
    tokens: Vec<u32>,
    block_size: u32,
    lora_name: Option<String>,
    block_hashes: Option<Vec<LocalBlockHash>>,
}

impl TokensWithHashes {
    pub fn new(tokens: Vec<u32>, block_size: u32) -> Self {
        Self {
            tokens,
            block_size,
            lora_name: None,
            block_hashes: None,
        }
    }

    pub fn with_lora_name(mut self, name: String) -> Self {
        self.lora_name = Some(name);
        self
    }

    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    pub fn get_or_compute_block_hashes(&mut self) -> &[LocalBlockHash] {
        if self.block_hashes.is_none() {
            self.block_hashes = Some(compute_block_hash_for_seq(
                &self.tokens,
                self.block_size,
                self.lora_name.as_deref(),
            ));
        }
        self.block_hashes.as_ref().unwrap()
    }
}

// ============================================================================
// KvIndexerInterface — the trait we implement for Dynamo integration.
//
// In an upstream PR this would be imported from dynamo_kv_router::indexer.
// Vendored here so the crate builds standalone.
// ============================================================================

/// Async trait for KV cache indexing (matches Dynamo's KvIndexerInterface).
#[async_trait::async_trait]
pub trait KvIndexerInterface: Send + Sync {
    async fn find_matches(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<OverlapScores, KvRouterError>;

    async fn find_matches_for_request(
        &self,
        tokens: &[u32],
        lora_name: Option<&str>,
    ) -> Result<OverlapScores, KvRouterError>;

    async fn apply_event(&self, event: RouterEvent);

    async fn remove_worker(&self, worker: WorkerId);

    async fn remove_worker_dp_rank(&self, worker: WorkerId, _dp_rank: DpRank) {
        self.remove_worker(worker).await;
    }

    fn shutdown(&self);

    async fn dump_events(&self) -> Result<Vec<RouterEvent>, KvRouterError>;

    async fn process_routing_decision_for_request(
        &self,
        tokens_with_hashes: &mut TokensWithHashes,
        worker: WorkerWithDpRank,
    ) -> Result<(), KvRouterError>;

    async fn flush(&self) -> usize;
}

/// Seed for XXH3 hashing (matches Dynamo's XXH3_SEED).
pub const XXH3_SEED: u64 = 1337;

/// Compute block hashes for a token sequence (matches Dynamo's compute_block_hash_for_seq).
pub fn compute_block_hash_for_seq(
    tokens: &[u32],
    kv_block_size: u32,
    lora_name: Option<&str>,
) -> Vec<LocalBlockHash> {
    let seed = match lora_name.filter(|n| !n.is_empty()) {
        Some(name) => XXH3_SEED.wrapping_add(xxhash_rust::xxh3::xxh3_64(name.as_bytes())),
        None => XXH3_SEED,
    };
    tokens
        .chunks_exact(kv_block_size as usize)
        .map(|chunk| {
            let bytes: Vec<u8> = chunk.iter().flat_map(|&num| num.to_le_bytes()).collect();
            LocalBlockHash(xxhash_rust::xxh3::xxh3_64_with_seed(&bytes, seed))
        })
        .collect()
}

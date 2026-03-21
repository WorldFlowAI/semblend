//! In-process semantic donor store with SIMD-friendly cosine similarity.
//!
//! Stores donor embeddings in a flat f32 matrix [N, 384] and performs
//! brute-force cosine similarity search. At N=10K with 384-dim embeddings,
//! search takes ~0.5ms on modern x86 with AVX2 auto-vectorization.
//!
//! For N > 100K, this should be replaced with HNSW or GPU-accelerated search.

use parking_lot::RwLock;
use std::collections::VecDeque;
use std::time::Instant;

const EMBEDDING_DIM: usize = 384;

/// A registered donor with embedding and token IDs.
#[derive(Debug, Clone)]
pub struct DonorEntry {
    /// Unique identifier for this donor request.
    pub request_id: String,
    /// Token IDs of the donor prompt (used for RadixTree lookup).
    pub token_ids: Vec<u32>,
    /// L2-normalized embedding vector [384].
    pub embedding: Vec<f32>,
    /// Registration timestamp.
    pub registered_at: Instant,
}

/// Result of a donor search.
#[derive(Debug, Clone)]
pub struct DonorMatch {
    /// The matched donor entry.
    pub donor: DonorEntry,
    /// Cosine similarity score (0-1).
    pub similarity: f32,
}

/// Thread-safe donor store with LRU eviction and SIMD cosine search.
pub struct DonorStore {
    inner: RwLock<DonorStoreInner>,
}

struct DonorStoreInner {
    /// Donor entries in insertion order (front = oldest).
    entries: VecDeque<DonorEntry>,
    /// Flat embedding matrix [N * EMBEDDING_DIM].
    embeddings: Vec<f32>,
    /// Maximum entries before LRU eviction.
    max_entries: usize,
    /// Minimum cosine similarity for a match.
    min_similarity: f32,
    // Stats tracked via atomic counters in the SemanticKvIndexer layer.
}

impl DonorStore {
    /// Create a new donor store.
    pub fn new(max_entries: usize, min_similarity: f32) -> Self {
        Self {
            inner: RwLock::new(DonorStoreInner {
                entries: VecDeque::with_capacity(max_entries.min(1024)),
                embeddings: Vec::with_capacity(max_entries.min(1024) * EMBEDDING_DIM),
                max_entries,
                min_similarity,
            }),
        }
    }

    /// Register a donor. O(1) append with LRU eviction at capacity.
    pub fn add_donor(&self, entry: DonorEntry) {
        let mut inner = self.inner.write();

        // LRU eviction
        while inner.entries.len() >= inner.max_entries {
            inner.entries.pop_front();
            let drain_end = EMBEDDING_DIM.min(inner.embeddings.len());
            inner.embeddings.drain(..drain_end);
        }

        inner.embeddings.extend_from_slice(&entry.embedding);
        inner.entries.push_back(entry);
    }

    /// Search for the best donor by cosine similarity.
    /// Returns None if no donor above min_similarity threshold.
    pub fn find_donor(&self, query_embedding: &[f32]) -> Option<DonorMatch> {
        let inner = self.inner.read();
        let n = inner.entries.len();
        if n == 0 || query_embedding.len() != EMBEDDING_DIM {
            return None;
        }

        let t0 = Instant::now();
        let sims = cosine_similarities(query_embedding, &inner.embeddings, n);

        // Find best match above threshold
        let mut best_idx = None;
        let mut best_sim = inner.min_similarity;

        for (i, &sim) in sims.iter().enumerate() {
            if sim > best_sim {
                best_sim = sim;
                best_idx = Some(i);
            }
        }

        let _elapsed_us = t0.elapsed().as_micros() as f64;
        // Note: can't update stats through read lock, would need separate atomic counters
        // for production. Keeping simple for now.

        best_idx.map(|idx| DonorMatch {
            donor: inner.entries[idx].clone(),
            similarity: best_sim,
        })
    }

    /// Number of donors in the store.
    pub fn len(&self) -> usize {
        self.inner.read().entries.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.read().entries.is_empty()
    }

    /// Clear all donors.
    pub fn clear(&self) {
        let mut inner = self.inner.write();
        inner.entries.clear();
        inner.embeddings.clear();
    }
}

/// Compute cosine similarities between a query and N embeddings.
/// query: [EMBEDDING_DIM], embeddings: [N * EMBEDDING_DIM] (flat row-major).
///
/// Uses SIMD-friendly loop structure for LLVM auto-vectorization.
fn cosine_similarities(query: &[f32], embeddings: &[f32], n: usize) -> Vec<f32> {
    let dim = EMBEDDING_DIM;

    // Pre-compute query norm
    let query_norm_sq: f32 = query.iter().map(|x| x * x).sum();
    let query_norm = query_norm_sq.sqrt();
    if query_norm < 1e-10 {
        return vec![0.0; n];
    }
    let inv_query_norm = 1.0 / query_norm;

    let mut results = Vec::with_capacity(n);

    for i in 0..n {
        let offset = i * dim;
        let emb = &embeddings[offset..offset + dim];

        let mut dot = 0.0f32;
        let mut emb_norm_sq = 0.0f32;

        // Process in chunks of 8 for SIMD auto-vectorization
        let chunks = dim / 8;
        let remainder = dim % 8;

        for c in 0..chunks {
            let base = c * 8;
            for j in 0..8 {
                let q = query[base + j];
                let e = emb[base + j];
                dot += q * e;
                emb_norm_sq += e * e;
            }
        }
        for j in 0..remainder {
            let idx = chunks * 8 + j;
            dot += query[idx] * emb[idx];
            emb_norm_sq += emb[idx] * emb[idx];
        }

        let emb_norm = emb_norm_sq.sqrt();
        let sim = if emb_norm > 1e-10 {
            dot * inv_query_norm / emb_norm
        } else {
            0.0
        };
        results.push(sim);
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_embedding(seed: usize) -> Vec<f32> {
        let mut emb: Vec<f32> = (0..EMBEDDING_DIM)
            .map(|i| ((i * 17 + seed * 31 + 7) % 1000) as f32 / 1000.0)
            .collect();
        // L2 normalize
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        for v in &mut emb {
            *v /= norm;
        }
        emb
    }

    #[test]
    fn test_empty_store() {
        let store = DonorStore::new(100, 0.5);
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
        let query = make_embedding(999);
        assert!(store.find_donor(&query).is_none());
    }

    #[test]
    fn test_add_and_find() {
        let store = DonorStore::new(100, 0.5);
        let emb = make_embedding(42);

        store.add_donor(DonorEntry {
            request_id: "req-1".into(),
            token_ids: vec![1, 2, 3],
            embedding: emb.clone(),
            registered_at: Instant::now(),
        });

        assert_eq!(store.len(), 1);

        // Self-similarity should be ~1.0
        let result = store.find_donor(&emb);
        assert!(result.is_some());
        let m = result.unwrap();
        assert!((m.similarity - 1.0).abs() < 0.01);
        assert_eq!(m.donor.request_id, "req-1");
    }

    #[test]
    fn test_lru_eviction() {
        let store = DonorStore::new(3, 0.0);

        for i in 0..5 {
            store.add_donor(DonorEntry {
                request_id: format!("req-{i}"),
                token_ids: vec![i as u32],
                embedding: make_embedding(i),
                registered_at: Instant::now(),
            });
        }

        assert_eq!(store.len(), 3);
    }

    #[test]
    fn test_similarity_threshold() {
        let store = DonorStore::new(100, 0.99);
        store.add_donor(DonorEntry {
            request_id: "req-1".into(),
            token_ids: vec![1],
            embedding: make_embedding(1),
            registered_at: Instant::now(),
        });

        // Very different embedding should not match at 0.99 threshold
        let query = make_embedding(999);
        let result = store.find_donor(&query);
        // May or may not match depending on the pseudo-random embeddings
        if let Some(m) = result {
            assert!(m.similarity >= 0.99);
        }
    }

    #[test]
    fn test_cosine_self_similarity() {
        let emb = make_embedding(42);
        let sims = cosine_similarities(&emb, &emb, 1);
        assert!((sims[0] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_clear() {
        let store = DonorStore::new(100, 0.5);
        store.add_donor(DonorEntry {
            request_id: "req-1".into(),
            token_ids: vec![1],
            embedding: make_embedding(1),
            registered_at: Instant::now(),
        });
        assert_eq!(store.len(), 1);
        store.clear();
        assert_eq!(store.len(), 0);
    }
}

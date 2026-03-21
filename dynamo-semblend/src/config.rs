//! Configuration for the semantic KV indexer.

use serde::{Deserialize, Serialize};

/// Configuration for the SemBlend semantic extension to Dynamo KVBM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticConfig {
    /// Minimum cosine similarity for a semantic match (0.0-1.0).
    /// Default: 0.60 (tuned from SemBlend benchmarks across ShareGPT, MultiNews, Bitext).
    pub min_similarity: f32,

    /// Below this overlap ratio (matched_blocks / total_blocks), semantic search triggers.
    /// Default: 0.50 (if less than half the blocks match, try semantic).
    pub min_overlap_ratio: f32,

    /// Maximum number of donors in the semantic store.
    /// Default: 10,000 (LRU eviction beyond this).
    pub max_donors: usize,

    /// Embedding dimension (384 for MiniLM-L6-v2).
    pub embedding_dim: usize,

    /// URL of the Python MiniLM embedding sidecar.
    /// Default: "http://127.0.0.1:9999"
    pub embed_url: String,

    /// KV block size in tokens (must match Dynamo's kv_block_size).
    /// Default: 32 (Dynamo default).
    pub kv_block_size: u32,

    /// Minimum prompt length (in tokens) to attempt semantic search.
    /// Short prompts are not worth the embedding overhead.
    /// Default: 100 tokens.
    pub min_tokens_for_semantic: usize,

    /// Whether semantic search is enabled.
    /// Default: true.
    pub enabled: bool,
}

impl Default for SemanticConfig {
    fn default() -> Self {
        Self {
            min_similarity: 0.60,
            min_overlap_ratio: 0.50,
            max_donors: 10_000,
            embedding_dim: 384,
            embed_url: "http://127.0.0.1:9999".into(),
            kv_block_size: 32,
            min_tokens_for_semantic: 100,
            enabled: true,
        }
    }
}

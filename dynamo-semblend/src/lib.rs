//! # dynamo-semblend
//!
//! SemBlend semantic KV cache lookup for NVIDIA Dynamo KVBM.
//!
//! Extends Dynamo's RadixTree exact-prefix matching with embedding-based
//! semantic search. When the RadixTree returns low overlap for a request,
//! SemBlend searches for semantically similar cached prompts and returns
//! the donor's overlap scores.
//!
//! ## Architecture
//!
//! ```text
//! Request tokens
//!     -> Dynamo RadixTree (exact block-hash prefix match)
//!     -> if overlap < threshold: SemBlend semantic fallback
//!         -> embed prompt (MiniLM via sidecar, ~3ms)
//!         -> cosine search donor store (~0.5ms at 10K entries)
//!         -> if hit: query RadixTree with donor's tokens
//!         -> return max(exact_scores, donor_scores)
//!     -> Dynamo cost function routes to best worker
//! ```
//!
//! ## Integration Points
//!
//! 1. **SemanticKvIndexer**: Wraps any `KvIndexerInterface` impl with semantic fallback
//! 2. **DonorStore**: SIMD-accelerated cosine similarity search over 384-dim embeddings
//! 3. **EmbedClient**: HTTP client for Python MiniLM sidecar
//!
//! ## Usage (with Dynamo)
//!
//! ```rust,ignore
//! use dynamo_semblend::{SemanticKvIndexer, SemanticConfig};
//!
//! let inner_indexer = /* Dynamo's KvIndexer */;
//! let semantic = SemanticKvIndexer::new(inner_indexer, SemanticConfig {
//!     min_similarity: 0.60,
//!     min_overlap_ratio: 0.50,
//!     embed_url: "http://127.0.0.1:9999".into(),
//!     ..Default::default()
//! });
//! // Use `semantic` wherever `KvIndexerInterface` is expected
//! ```

pub mod config;
pub mod donor_store;
pub mod embed_client;
pub mod protocols;
pub mod semantic_indexer;

pub use config::SemanticConfig;
pub use donor_store::DonorStore;
pub use embed_client::EmbedClient;
pub use semantic_indexer::SemanticKvIndexer;

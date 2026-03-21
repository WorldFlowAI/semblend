"""SemBlend Dynamo KVBM integration — semantic KV-aware routing for Dynamo.

Extends Dynamo's token-level prefix matching (RadixTree) with semantic
donor discovery. When exact block-hash matching fails, SemBlend's embedding
search finds semantically similar cached requests.

Two integration paths:

1. KvIndexer wrapper (recommended):
   Wraps Dynamo's KvIndexer to add semantic fallback on prefix miss.
   Intercepts find_matches_for_request() — when the RadixTree returns
   low overlap, runs SemBlend pipeline and returns donor's overlap scores.

2. KvEventPublisher wrapper:
   Wraps Dynamo's event publisher to emit semantic events alongside
   block events. The SemRouter subscribes to both event streams.

Architecture:
    Request tokens
        -> Dynamo RadixTree (exact block-hash prefix match)
        -> if low overlap: SemBlend semantic search
            -> embed prompt text (MiniLM, ~3ms)
            -> cosine similarity against donor store
            -> if hit: return donor's block overlap scores
        -> Dynamo cost function routes to best worker

Compatible with:
    - Dynamo KV Router (Rust, ai-dynamo/dynamo)
    - Dynamo KVBM Logical (block lifecycle manager)
    - All Dynamo-supported engines: vLLM, TRT-LLM, SGLang
"""

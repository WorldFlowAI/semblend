# SemBlend

**Semantic KV cache reuse for LLM inference engines.**

SemBlend extends exact-prefix KV caching (LMCache, vLLM prefix cache, SGLang RadixAttention) with *semantic* donor discovery. When a new prompt is semantically similar to a prior one but lexically different — different instruction wording, sentence ordering, or template fields — SemBlend finds and reuses the cached KV tensors, eliminating redundant prefill computation.

```
Without SemBlend:  vLLM + LMCache  →  0% hit  →  full prefill every request
With SemBlend:     vLLM + LMCache  →  30–88% hit  →  reuse KV from similar past requests
```

## What It Does

| System | Hit Condition | Semantically Similar Prompts |
|--------|--------------|------------------------------|
| vLLM prefix cache | Exact token-level prefix match | Full prefill (0% hit) |
| LMCache alone | Exact 256-token chunk match | Full prefill (0% hit) |
| **LMCache + SemBlend** | Semantic similarity ≥ 0.60 | **Reuse donor KV (30–88% hit)** |

SemBlend runs ~8ms of in-process MiniLM embedding + cosine search on every request. On a miss it adds negligible overhead. On a hit it replaces a 2–17 second prefill with sub-second KV retrieval.

## Performance

Empirical results on A10G GPU, Qwen2.5-7B-AWQ + vLLM 0.14.1 + LMCache.

### TTFT speedup on hits vs. cold prefill (diverse real content)

| Context Length | Cold TTFT | SemBlend Miss (+overhead) | SemBlend Hit | Hit Speedup |
|---------------|-----------|--------------------------|--------------|-------------|
| 4K tokens | 1,859 ms | 1,864 ms (+0.3%) | 801 ms | **2.3x** |
| 8K tokens | 3,193 ms | 3,315 ms (+3.8%) | 817 ms | **3.9x** |
| 16K tokens | 5,852 ms | 6,064 ms (+3.6%) | 871 ms | **6.7x** |
| 32K tokens | 15,418 ms | — | 1,288 ms | **12.0x** |

*Hit TTFT is consistently ~800ms regardless of context length — it's bounded by KV retrieval, not prefill.*

Miss overhead grows with context length (5–212ms at ~20 donors in store). Break-even: P_hit > 5% at 8K is net-positive.

### Real-world hit rates (WildChat-1M user conversations)

| Workload | Requests | Hit Rate | Hit-only TTFT speedup |
|---------|----------|----------|----------------------|
| Short prompts (≥4K chars) | 250 | 29.2% | **1.63x** |
| Long prompts (≥8K chars) | 150 | 30.0% | **1.88x** |

Hit rate increases with cosine similarity: 17% at 0.50–0.60 → 60% at 0.90–1.00.

### Cross-dataset hit rates and speedups (8K tokens, Qwen2.5-7B)

| Dataset | Hit Rate | Overall Speedup | Hit-only Speedup |
|---------|----------|----------------|-----------------|
| CNN/DailyMail (summarization) | 50% | 2.29x | 2.39x |
| MultiNews (multi-doc summary) | 75% | 2.23x | 2.23x |
| SAMSum (dialogue summary) | 88% | 2.37x | 2.37x |

*Speedups measured over cold vLLM+LMCache baseline (0% hit on semantically different requests).*

### Multi-turn dialogue (4K context, 3 turns)

| Turn | Hit Rate | TTFT Speedup |
|------|----------|-------------|
| Turn 1 (cold) | — | 1.0x (baseline: 4019 ms) |
| Turn 2 | 99.5% | **5.1x** (791 ms) |
| Turn 3 | 99.5% | **5.1x** (787 ms) |

Multi-turn conversations naturally reuse the same prefix → near-perfect hit rates without any workload tuning.

### SGLang comparison (8K, cross-instruction RAG workload)

| Engine | Hit Rate | p50 TTFT | Speedup |
|--------|----------|----------|---------|
| SGLang (RadixAttention, no SemBlend) | 0% | 3,850 ms | 0.98x |
| vLLM + LMCache | 0% | 3,193 ms | 1.0x |
| vLLM + LMCache + SemBlend | 50–88% | 817 ms | **3.9x** |
| SGLang + SemBlend | 50–88% | 624 ms | **3.7x** |

### Quality

SemBlend injects donor KV with RoPE position correction. Output quality impact on high-hit workloads:

| Dataset | PPL ratio (SemBlend / cold) |
|---------|---------------------------|
| CNN/DailyMail | 1.006 |
| WikiHow | 1.012 |
| XSum | 1.025 |
| MultiNews (hit-only runs) | 1.007 |

PPL ratio ≤ 1.025 on most datasets; elevated ratios (≥1.1) on low-hit runs are due to miss-run variance, not KV injection.

## Installation

```bash
# Core (CPU-only: numpy + rapidfuzz)
pip install semblend

# With vLLM integration
pip install semblend[vllm]

# With SGLang integration
pip install semblend[sglang]

# With sentence-transformers embedder
pip install semblend[embedder]
```

## Quick Start: vLLM + LMCache

vLLM integrates via LMCache's `KVConnectorBase_V1` — a first-class public API. No patching required.

```bash
pip install semblend[vllm] vllm lmcache

vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
  --kv-transfer-config '{
    "kv_connector": "SemBlendConnectorV1",
    "kv_connector_module_path": "semblend.integration.vllm.connector_v1",
    "kv_role": "kv_both"
  }'
```

Configure via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SEMBLEND_ENABLED` | `1` | Enable semantic donor search |
| `SEMBLEND_MIN_SIMILARITY` | `0.60` | Cosine similarity threshold |
| `SEMBLEND_EMBEDDER` | `minilm` | Embedder type (`minilm`, `jaccard`, `onnx_gpu`) |
| `SEMBLEND_FUZZY_CHUNKS` | `0` | Enable fuzzy chunk matching |

## Quick Start: SGLang

SGLang integrates via a RadixCache patch applied at startup. This is necessary because SGLang's `RadixCache.match_prefix` does not currently have a hook for semantic fallback lookup — a [PR to SGLang](https://github.com/sgl-project/sglang) is in progress to add a first-class `SemanticPrefixProvider` interface (analogous to [LMCache PR #2803](https://github.com/LMCache/LMCache/pull/2803)).

```bash
pip install semblend[sglang] sglang

# Option 1: CLI launcher — applies the RadixCache patch automatically
semblend-sglang --model-path Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 --port 8000

# Option 2: Programmatic — call before SGLang initializes
from semblend.integration.sglang.radix_patcher import patch_radix_cache
patch_radix_cache()
import sglang as sgl
# ... start server ...
```

## How It Works

```
Request → MiniLM Embed (5ms) → Cosine Search (1ms) → Align (1ms) → Inject KV
              ↓                       ↓                    ↓
         384-dim vector        Find similar donor    Match chunk boundaries
                                in donor store       via MD5 hash alignment
```

1. **Embed**: Compute a 384-dim MiniLM-L6-v2 embedding (sliding-window for long docs)
2. **Search**: Brute-force cosine similarity against the donor store (<1ms at 1K donors)
3. **Align**: MD5 chunk hashing at 256-token boundaries finds reusable KV chunks
4. **Inject**: Replace target token IDs with donor token IDs — LMCache/RadixCache finds cached KV

## When SemBlend Helps

SemBlend is most effective for workloads where prompts share a large common context:

- **Document Q&A / RAG**: Same retrieved documents, different questions
- **Summarization**: Same article, different instruction phrasing
- **Multi-turn dialogue**: Conversation history grows across turns
- **Code completion**: Shared repository context across requests

SemBlend has minimal overhead on dissimilar workloads (e.g. code generation from scratch: measured 0% hit, 0.96x — 4% overhead from embedding + search).

## License

Business Source License 1.1 (BSL-1.1). Free for non-production use including testing, development, evaluation, and academic research. Production use requires a commercial license from WorldFlow AI. Converts to Apache License 2.0 on 2030-03-16.

Contact: research@worldflowai.com


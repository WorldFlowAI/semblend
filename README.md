# SemBlend

<p align="center">
  <a href="https://pypi.org/project/semblend/"><img alt="PyPI" src="https://img.shields.io/pypi/v/semblend?color=blue&label=pypi"></a>
  <a href="https://pypi.org/project/semblend/"><img alt="Python" src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue"></a>
  <a href="https://github.com/worldflowai/semblend/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/worldflowai/semblend/ci.yml?branch=main&label=CI"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache%202.0-green"></a>
  <!-- Paper badge: uncomment when arXiv submission is accepted -->
  <!-- <a href="https://arxiv.org/abs/XXXX.XXXXX"><img alt="Paper" src="https://img.shields.io/badge/paper-arXiv-red"></a> -->
</p>

**Semantic KV cache reuse for LLM inference engines.**

SemBlend extends exact-prefix KV caching (vLLM, LMCache, SGLang) with *semantic* donor discovery. When a prompt is semantically similar to a cached one but lexically different — different instruction phrasing, sentence order, or template fields — SemBlend finds and reuses the cached KV tensors, replacing a multi-second prefill with sub-second KV retrieval.

```
vLLM + LMCache alone:        semantically similar prompt  →  0% hit   →  full prefill
vLLM + LMCache + SemBlend:                                →  83–100% hit  →  reuse donor KV
```

## Performance

Measured on A10G GPU, Qwen2.5-7B-AWQ, vLLM 0.14.1 + LMCache.

### TTFT speedup vs cold prefill

| Context | n | Cold p50 | Hit p50 | Speedup | 95% CI | Hit Rate |
|---------|---|----------|---------|---------|--------|----------|
| 8K | 30 | 3,096 ms | 534 ms | **5.5x** | [5.2, 5.8] | 90% |
| 12K | 28 | 4,894 ms | 623 ms | **7.9x** | [7.8, 7.9] | 89% |
| 16K | 20 | 6,646 ms | 763 ms | **8.7x** | [8.6, 8.8] | 90% |
| 24K | 14 | 10,616 ms | 827 ms | **13.0x** | [12.8, 13.2] | 100% |

Hit TTFT stays <1s regardless of context length — bounded by KV retrieval from CPU offload, not prefill. Speedup scales with context because cold TTFT grows linearly (~0.46ms/token) while warm TTFT is sublinear. ~10% miss rate at 8K–16K is from instruction variants crossing the cosine similarity threshold; 24K achieves 100% because the instruction is <0.1% of total tokens.

### Hit rates on real workloads

| Workload | Hit Rate | Hit-only Speedup |
|---------|----------|-----------------|
| WildChat-1M conversations (≥4K) | **82.7%** | 1.69x |
| Summarization (CNN/DM, SAMSum) | 50–88% | 2.3–2.4x |
| Multi-turn dialogue (turn 2+) | 99.5% | 5.1x |
| Cross-instruction RAG (8K) | **90%** | 5.5x |
| Cross-instruction RAG (16K) | **90%** | **8.7x** |
| Code generation (dissimilar) | 0% | 0.96x |

Full-document segmented GPU embedding (v0.2.0) achieves 100% coverage of the prompt regardless of length, enabling 82.7% hit rate on real WildChat conversations (up from 29% with sparse sampling).

### Quality

RoPE position correction keeps output quality near baseline:

| Dataset | PPL ratio (SemBlend / cold) |
|---------|---------------------------|
| CNN/DailyMail | 1.006 |
| WikiHow | 1.012 |
| XSum | 1.025 |

See the benchmarks/ directory for full reproduction details.

## Installation

```bash
pip install semblend            # CPU-only core (numpy + rapidfuzz)
pip install semblend[vllm]      # + vLLM/LMCache integration
pip install semblend[sglang]    # + SGLang integration
pip install semblend[embedder]  # + sentence-transformers (MiniLM GPU)
```

## Quick Start: vLLM + LMCache

Integrates via LMCache's `KVConnectorBase_V1` — no patching required.

```bash
pip install semblend[vllm] vllm lmcache

vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
  --kv-transfer-config '{
    "kv_connector": "SemBlendConnectorV1",
    "kv_connector_module_path": "semblend.integration.vllm.connector_v1",
    "kv_role": "kv_both"
  }'
```

> **CacheBlend support:** For selective layer recomputation (CacheBlend), vLLM must expose
> the loaded model to KV connectors via `initialize_worker_connector()`. This is available
> in vLLM builds that include [PR #37339](https://github.com/vllm-project/vllm/pull/37339).
> Without it, SemBlend's semantic matching and KV injection still work — only CacheBlend's
> per-layer recomputation is unavailable.

## Quick Start: SGLang

```bash
pip install semblend[sglang] sglang

# CLI launcher — applies the RadixCache patch automatically
semblend-sglang --model-path Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8000
```

Or programmatically — call before SGLang initializes:

```python
from semblend.integration.sglang.radix_patcher import patch_radix_cache
patch_radix_cache()
# ... start SGLang server ...
```

A first-class [`SemanticPrefixProvider`](https://github.com/sgl-project/sglang/pull/20806) interface (no patching) is in progress upstream.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SEMBLEND_ENABLED` | `1` | Enable semantic donor search |
| `SEMBLEND_MIN_SIMILARITY` | `0.60` | Cosine similarity threshold |
| `SEMBLEND_EMBEDDER` | `minilm` | `minilm` (auto GPU) · `onnx_gpu` |
| `SEMBLEND_FUZZY_CHUNKS` | `0` | Fuzzy chunk matching for shifted prefixes |

## How It Works

```
Request → Embed (2–15ms) → Search (1ms) → Align (1ms) → Inject KV
              ↓                 ↓              ↓
         MiniLM-L6-v2    cosine search   MD5 chunk hash
         GPU (ONNX RT)   donor store     256-token boundary
         segmented pool
```

1. **Embed** — full-document segmented embedding on GPU via ONNX-runtime. Long prompts are split into overlapping 256-token windows, embedded in parallel, and mean-pooled into a single vector. 100% content coverage at any prompt length (~2ms short, ~10ms at 8K, ~15ms at 32K).
2. **Search** — brute-force cosine similarity against the donor store (<1ms at 1K donors; CAGRA GPU ANN for larger pools)
3. **Align** — MD5 chunk hashing finds reusable 256-token KV chunks; optional fuzzy matching handles shifted boundaries
4. **Inject** — donor token IDs substituted into the request; LMCache/RadixCache retrieves cached KV; RoPE correction applied in-place on K tensors

## When SemBlend Helps

Most effective when prompts share a large common context:

- **Document Q&A / RAG** — same retrieved passages, different questions
- **Summarization** — same article, different instruction phrasing
- **Multi-turn dialogue** — conversation history prefix reused across turns
- **Code completion** — shared repository context across requests

Dissimilar workloads (code generation from scratch, fully novel queries) see ~4% overhead with 0% hit — negligible in practice.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[Apache License 2.0](LICENSE).

Built at [WorldFlow AI](https://worldflowai.com). For enterprise support contact [research@worldflowai.com](mailto:research@worldflowai.com).

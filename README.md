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

Measured on A10G GPU (0.85 utilization), Qwen2.5-7B-AWQ, vLLM 0.14.1 + LMCache. All results from live benchmarks on real HuggingFace datasets with fresh pod isolation (n=15 per cell).

### TTFT speedup vs cold prefill

| Context | Cold TTFT | SemBlend TTFT | Speedup |
|---------|----------|---------------|---------|
| 4K | 2,102 ms | 433 ms | **4.9x** |
| 8K | 3,816 ms | 539 ms | **7.1x** |
| 12K | 5,655 ms | 648 ms | **8.7x** |
| 16K | 7,635 ms | 760 ms | **10.0x** |
| 24K | 11,977 ms | 972 ms | **12.3x** |

SemBlend TTFT stays under 1 second regardless of context length. Speedup scales linearly because cold prefill grows with context while SemBlend loads cached KV in constant time.

### Multi-dataset validation

Identical speedups across content types -- SemBlend is content-agnostic:

| Dataset | Content Type | 8K Speedup | 16K Speedup | 24K Speedup |
|---------|-------------|------------|-------------|-------------|
| XSum | News summaries | 7.1x | 10.0x | 12.3x |
| CNN/DailyMail | Long-form journalism | 7.1x | 9.4x | 12.2x |
| MultiNews | Multi-document news | -- | 9.3x | -- |

### Quality

Quality validated across 5 datasets, 4-5 context lengths each, with PPL ratio + LLM-as-judge faithfulness scoring (360 total runs):

| Dataset | PPL Range | Status | Judge (Cold) | Judge (SemBlend) | Faithful |
|---------|-----------|--------|--------------|------------------|----------|
| XSum | 1.018-1.054 | PASS | 0.84 | 0.84 | 100% |
| CNN/DailyMail | 1.011-1.049 | PASS | 0.87 | 0.86 | 97% |
| WikiHow | 0.987-1.037 | PASS | 0.82 | 0.84 | 97% |
| MultiNews | 0.958-1.064 | PASS | 0.79 | 0.78 | 100% |
| SAMSum | 1.140-1.198 | ELEVATED | 0.78 | 0.86 | 87% |

PPL < 1.065 for 4/5 datasets at all lengths. SAMSum shows elevated PPL due to short dialogue turns, but LLM-as-judge rates SemBlend output higher than cold (0.86 vs 0.78). 24 dataset-length cells, 360 total runs.

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

# SemBlend

<p align="center">
  <a href="https://pypi.org/project/semblend/"><img alt="PyPI" src="https://img.shields.io/pypi/v/semblend?color=blue&label=pypi"></a>
  <a href="https://pypi.org/project/semblend/"><img alt="Python" src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue"></a>
  <a href="https://github.com/worldflowai/semblend/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/worldflowai/semblend/ci.yml?branch=main&label=CI"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache%202.0-green"></a>
  <a href="https://arxiv.org/abs/TODO"><img alt="Paper" src="https://img.shields.io/badge/paper-arXiv-red"></a>
</p>

**Semantic KV cache reuse for LLM inference engines.**

SemBlend extends exact-prefix KV caching (vLLM, LMCache, SGLang) with *semantic* donor discovery. When a prompt is semantically similar to a cached one but lexically different — different instruction phrasing, sentence order, or template fields — SemBlend finds and reuses the cached KV tensors, replacing a multi-second prefill with sub-second KV retrieval.

```
vLLM + LMCache alone:        semantically similar prompt  →  0% hit   →  full prefill
vLLM + LMCache + SemBlend:                                →  30–88% hit  →  reuse donor KV
```

## Performance

Measured on A10G GPU, Qwen2.5-7B-AWQ, vLLM 0.14.1 + LMCache.

### TTFT speedup vs cold prefill

| Context | Cold TTFT | Hit TTFT | Speedup | Break-even P_hit |
|---------|-----------|----------|---------|-----------------|
| 4K | 1,859 ms | 801 ms | **2.3x** | <1% |
| 8K | 3,193 ms | 817 ms | **3.9x** | 4.9% |
| 16K | 5,852 ms | 871 ms | **6.7x** | 4.1% |
| 32K | 15,418 ms | 1,288 ms | **12.0x** | — |

Hit TTFT is ~800ms regardless of context length — bounded by KV retrieval, not prefill. Miss overhead is 5–212ms (negligible). SemBlend is net-positive at virtually any nonzero hit rate for contexts ≥ 4K.

### Hit rates on real workloads

| Workload | Hit Rate | Hit-only Speedup |
|---------|----------|-----------------|
| WildChat-1M short prompts (≥4K) | 29.2% | 1.63x |
| WildChat-1M long prompts (≥8K) | 30.0% | 1.88x |
| Summarization (CNN/DM, SAMSum) | 50–88% | 2.3–2.4x |
| Multi-turn dialogue (turn 2+) | 99.5% | 5.1x |
| Cross-instruction RAG (8K) | 100% | 3.3–3.7x |
| Code generation (dissimilar) | 0% | 0.96x |

Hit rate scales with semantic similarity: 17% at cos≥0.50 → 60% at cos≥0.90.

### Quality

RoPE position correction keeps output quality near baseline:

| Dataset | PPL ratio (SemBlend / cold) |
|---------|---------------------------|
| CNN/DailyMail | 1.006 |
| WikiHow | 1.012 |
| XSum | 1.025 |

See the [paper](https://arxiv.org/abs/TODO) for full benchmark details.

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
| `SEMBLEND_EMBEDDER` | `minilm` | `minilm` · `jaccard` · `onnx_gpu` |
| `SEMBLEND_FUZZY_CHUNKS` | `0` | Fuzzy chunk matching for shifted prefixes |

## How It Works

```
Request → Embed (5ms) → Search (1ms) → Align (1ms) → Inject KV
             ↓               ↓              ↓
        MiniLM-L6-v2   cosine search   MD5 chunk hash
        384-dim         donor store     256-token boundary
```

1. **Embed** — 384-dim MiniLM-L6-v2 embedding; sliding-window sampling for long prompts
2. **Search** — brute-force cosine similarity against the donor store (<1ms at 1K donors)
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

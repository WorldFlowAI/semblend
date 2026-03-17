# SemBlend

**Semantic KV cache reuse for LLM inference engines.**

SemBlend extends exact-prefix caching (LMCache, vLLM prefix caching, SGLang RadixAttention) with semantic donor discovery. It finds and reuses KV tensors from semantically similar prior requests, eliminating redundant prefill computation — even when prompts differ in instruction wording, sentence ordering, or template fields.

- **2–12x TTFT speedup** on semantically similar requests (8K–32K tokens)
- **Zero model modification** — drop-in integration with vLLM and SGLang
- **~8ms overhead** — in-process MiniLM embedding + cosine search + alignment
- **Quality preserved** — PPL ratio ≤ 1.065 on high-hit workloads

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

## Quick Start: vLLM

```bash
pip install semblend[vllm] vllm lmcache

vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
  --kv-transfer-config '{
    "kv_connector": "SemBlendConnectorV1",
    "kv_connector_module_path": "semblend.integration.vllm.connector_v1",
    "kv_role": "kv_both"
  }'
```

SemBlend activates automatically. Configure via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SEMBLEND_ENABLED` | `1` | Enable semantic donor search |
| `SEMBLEND_MIN_SIMILARITY` | `0.60` | Cosine similarity threshold |
| `SEMBLEND_EMBEDDER` | `minilm` | Embedder type (`minilm`, `jaccard`, `onnx_gpu`) |
| `SEMBLEND_FUZZY_CHUNKS` | `0` | Enable fuzzy chunk matching |

## Quick Start: SGLang

```bash
pip install semblend[sglang] sglang

# Option 1: CLI launcher (patches RadixCache automatically)
semblend-sglang --model-path Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 --port 8000

# Option 2: Programmatic patching
from semblend.integration.sglang.radix_patcher import patch_radix_cache
patch_radix_cache()  # Call before SGLang starts
```

## How It Works

```
Request → MiniLM Embed (5ms) → Cosine Search (1ms) → Align (1ms) → Inject KV
              ↓                      ↓                     ↓
         384-dim vector      Find similar donor      Match chunk boundaries
                             in donor store          via MD5 hash alignment
```

1. **Embed**: Compute a 384-dim MiniLM-L6-v2 embedding of the prompt (sliding-window sampling for long documents)
2. **Search**: Brute-force cosine similarity against the donor store (O(N×d), <1ms at 1K donors)
3. **Align**: MD5 chunk hashing at 256-token boundaries finds reusable KV chunks
4. **Inject**: Replace target token IDs with donor token IDs → LMCache/RadixCache finds cached KV

## Performance

| Context | Cold TTFT | SemBlend Hit | Speedup |
|---------|-----------|-------------|---------|
| 4K | 1,859ms | 801ms | 2.3x |
| 8K | 3,193ms | 817ms | 3.9x |
| 16K | 5,852ms | 871ms | 6.7x |
| 32K | 15,418ms | 1,288ms | 12.0x |

*Qwen2.5-7B-AWQ on A10G. Hit TTFT is consistent ~800ms regardless of context length.*

Cross-engine comparison at 8K tokens (cross-instruction workload):

| Engine | Hit Rate | Hit TTFT | Speedup |
|--------|----------|----------|---------|
| SGLang (vanilla RadixCache) | 0% | — | 0.98x |
| vLLM + SemBlend + LMCache | 100% | 1,026ms | 3.26x |
| SGLang + SemBlend | 100% | 624ms | 3.71x |

## License

Business Source License 1.1 (BSL-1.1). Free for non-production use including testing, development, evaluation, and academic research. Production use requires a commercial license from WorldFlow AI. Converts to Apache License 2.0 on 2030-03-16.

Contact: licensing@worldflow.ai

## Links

- [Paper](https://arxiv.org/abs/XXXX.XXXXX)
- [Documentation](https://docs.worldflow.ai/semblend)
- [GitHub](https://github.com/worldflowai/semblend)

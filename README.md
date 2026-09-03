# SemBlend

<p align="center">
  <a href="https://pypi.org/project/semblend/"><img alt="PyPI" src="https://img.shields.io/pypi/v/semblend?color=blue&label=pypi"></a>
  <a href="https://pypi.org/project/semblend/"><img alt="Python" src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue"></a>
  <a href="https://github.com/worldflowai/semblend/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/worldflowai/semblend/ci.yml?branch=main&label=CI"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache%202.0-green"></a>
  <!-- Paper badge: uncomment when arXiv submission is accepted -->
  <!-- <a href="https://arxiv.org/abs/XXXX.XXXXX"><img alt="Paper" src="https://img.shields.io/badge/paper-arXiv-red"></a> -->
</p>

**Semantic donor discovery for engine-native KV caches.**

SemBlend helps inference engines reuse expensive long-context prefill work when exact-prefix
cache matching misses. It finds semantically similar prior requests, checks whether their
cached KV may be useful, and hands bounded donor evidence to engine-native connector/provider
paths.

SemBlend is the engine-facing semantic KV adapter layer: semantic search is evidence, and the inference backend remains authoritative for whether donor KV is materialized, recomputed, declined, or ignored.

```text
Exact prefix cache only:
  same meaning, different framing  ->  exact cache miss  ->  cold prefill

Engine cache + SemBlend:
  same meaning, different framing  ->  semantic donor evidence  ->  backend validates reuse
```

## What SemBlend Is

- A Python package for semantic donor discovery, alignment, and quality-gated KV reuse planning.
- A set of engine adapters for vLLM, SGLang, TensorRT-LLM, and Dynamo-facing experiments.
- A process-local library by default: no hosted service, daemon, or network dependency is required.
- Complementary to exact-prefix caching: exact hits should remain the fastest and highest-priority path.

## What SemBlend Is Not

- Not a semantic response cache.
- Not a replacement for vLLM, SGLang, TensorRT-LLM, Dynamo, llm-d, LMCache, or HiCache.
- Not a guarantee that every semantic match becomes materialized KV reuse.
- Not a license to write approximate donor KV into exact prefix-cache state without backend support.

## Integration Status

| Surface | Status | What it is for | Notes |
|---------|--------|----------------|-------|
| Core pipeline | Stable beta | Embedding, donor search, alignment, quality gates, layer-deviation planning | Installed by default |
| SGLang | Active | Engine-local semantic provider and Radix/HiCache integration | Patch-based path today; upstream provider interface is in progress |
| vLLM compatibility | Supported | Reproduce existing vLLM benchmark path through dynamic connector loading | May use LMCache for KV transfer; direct vLLM connector work is the strategic direction |
| TensorRT-LLM | Experimental | Connector/provider mapping and launch surface | Validate backend semantics before claiming materialized reuse |
| Dynamo-facing provider | Experimental | Semantic evidence/provider surface for Dynamo-style routing experiments |  |

## Installation

```bash
pip install semblend            # CPU-only core: numpy + rapidfuzz
pip install semblend[embedder]  # + sentence-transformers + ONNX Runtime
pip install semblend[onnx-gpu]  # + ONNX Runtime GPU
pip install semblend[vllm]      # + vLLM compatibility integration
pip install semblend[sglang]    # + SGLang integration
pip install semblend[trtllm]    # + TensorRT-LLM integration surface
```

## Quick Start: SGLang

```bash
pip install semblend[sglang] sglang

# CLI launcher: applies the current RadixCache patch path automatically.
semblend-sglang \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000
```

Or patch programmatically before SGLang initializes:

```python
from semblend.integration.sglang.radix_patcher import patch_radix_cache

patch_radix_cache()
# Start SGLang server after patching.
```

A first-class `SemanticPrefixProvider` interface is in progress upstream:
https://github.com/sgl-project/sglang/pull/31057

## Quick Start: vLLM Compatibility Path

The current vLLM compatibility path integrates through vLLM dynamic connector loading and may
use LMCache for KV transfer. This remains the supported path for reproducing the benchmark
results below.

```bash
pip install semblend[vllm] vllm lmcache

vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
  --kv-transfer-config '{
    "kv_connector": "SemBlendConnectorV1",
    "kv_connector_module_path": "semblend.integration.vllm.connector_v1",
    "kv_role": "kv_both"
  }'
```

## Quick Start: TensorRT-LLM Experimental Surface

```bash
pip install semblend[trtllm] tensorrt_llm

python - <<'PY'
from tensorrt_llm.llmapi.llm_args import KvCacheConnectorConfig

kv_connector_config = KvCacheConnectorConfig(
    connector_module="semblend.integration.trtllm.connector",
    connector_scheduler_class="SemBlendKvConnectorScheduler",
    connector_worker_class="SemBlendKvConnectorWorker",
)
print(kv_connector_config)
PY
```

The TensorRT-LLM package exposes a dynamically importable KV connector and provider. It builds a
strict cache namespace from model/tokenizer revision, KV/cache dtype, quantization, adapter/LoRA,
RoPE config, tensor-parallel config, block size, and backend cache layout. Semantic hits are
request-local plans: SemBlend may materialize non-identical donor KV into the recipient's allocated
blocks with RoPE correction, but does not publish those blocks as exact cache state.

The connector also has an optional Synapse/semantic-KV contract emitter. Set
`SEMBLEND_NATS_URL` and `SEMBLEND_NATS_SUBJECT=semantic-kv-events` to publish
`provider_generation_reset` and `donor_registered` events for fleet routing.
`SEMBLEND_DONOR_TENANT` and `SEMBLEND_DONOR_TEMPLATE` populate the namespace
`extra` map used by Synapse policy gates.

The TensorRT-LLM integration includes an optional SemBlend-owned runtime engine
wrapper that can activate suffix-only execution after a recompute boundary. This
is a portable interface candidate: the engine path consumes `SemanticKvPlan`
metadata, donor spans, target spans, target blocks, layer recompute masks,
RoPE/layout metadata, and the suffix-attention boundary. A future TensorRT-LLM
integration should expose this through a stable engine API instead of requiring
a runtime wrapper.

GPU validation harness:

```bash
python -m benchmarks.suite.trtllm_semantic_validation \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --output results/trtllm-semblend.json \
  --min-speedup 1.05 \
  --min-rouge1 0.35 \
  --min-token-f1 0.10
```

Use `--ppl-model <model>` for an optional perplexity delta and `--judge-command "<cmd>"` for an
LLM-as-judge hook that reads `{baseline, semblend}` JSON on stdin and prints a numeric score.

## How It Works

```text
Request
  -> embed or fingerprint prompt
  -> search local donor store
  -> align donor and target tokens
  -> apply quality gates and layer-deviation policy
  -> return donor evidence or a backend-native load plan
  -> backend validates, materializes, recomputes, or declines
```

1. **Embed**: long prompts are segmented into overlapping windows and embedded with MiniLM or
   ONNX Runtime GPU paths when configured.
2. **Search**: SemBlend searches a process-local donor store for semantically similar prior
   requests.
3. **Align**: token and chunk alignment identify reusable spans. Optional fuzzy chunk matching
   can handle shifted boundaries.
4. **Plan**: quality gates, reuse ratio, and bathtub-style layer sensitivity decide whether
   donor evidence is safe enough to pass to the backend.
5. **Materialize or decline**: the engine-local connector/provider decides whether to load KV,
   recompute layers, fall back to normal prefill, or record discovery-only telemetry.

## Safety and Accounting Boundaries

SemBlend distinguishes these outcomes:

| Outcome | Meaning |
|---------|---------|
| Exact prefix route | The serving engine/router found an exact cache hit. This should normally win before semantic reuse. |
| Semantic discovery only | SemBlend found a likely donor but did not affect placement or materialization. |
| Semantic placement | A router or substrate used SemBlend evidence to choose a better endpoint, but reuse is not yet confirmed. |
| Backend materialized reuse | The backend confirmed donor KV was safely loaded, blended, or recomputed for the request. |
| Backend declined reuse | The backend rejected the donor because of stale state, namespace mismatch, capacity, unsupported mode, or validation failure. |
| Cold fallback | Normal prefill/decode path. This is expected behavior, not failure. |

Reuse ROI should be counted only after backend-confirmed materialization. A semantic match by
itself is evidence, not a cache hit.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SEMBLEND_ENABLED` | `1` | Enable semantic donor search |
| `SEMBLEND_MIN_SIMILARITY` | `0.60` | Cosine similarity threshold |
| `SEMBLEND_EMBEDDER` | `minilm` | Embedder type, such as `minilm` or ONNX GPU paths |
| `SEMBLEND_FUZZY_CHUNKS` | `0` | Enable fuzzy chunk matching for shifted prefixes |
| `SEMBLEND_DISCOVERY_ONLY` | `0` | Run lookup and telemetry without materializing donor KV |
| `SEMBLEND_STRICT_PREFIX_BOUNDARY` | `1` | Keep materialization inside prefix-shaped backend contracts |
| `SEMBLEND_EXACT_MATERIALIZATION_ONLY` | `1` | Materialize only token-exact spans where required by the backend |

## Performance

These numbers are from the current vLLM compatibility benchmark path: A10G GPU at 0.85
utilization, Qwen2.5-7B-AWQ, vLLM 0.14.1 plus the LMCache-compatible integration path. Runs used
fresh pod isolation and real HuggingFace datasets with n=15 per cell.

### TTFT speedup vs cold prefill

| Context | Cold TTFT | SemBlend TTFT | Speedup |
|---------|-----------|---------------|---------|
| 4K | 2,102 ms | 433 ms | **4.9x** |
| 8K | 3,816 ms | 539 ms | **7.1x** |
| 12K | 5,655 ms | 648 ms | **8.7x** |
| 16K | 7,635 ms | 760 ms | **10.0x** |
| 24K | 11,977 ms | 972 ms | **12.3x** |

The speedup grows with context length because cold prefill scales with prompt length while
validated KV retrieval is much less sensitive to context length. Results depend on backend,
model, cache state, donor density, and workload overlap.

### Multi-dataset validation

| Dataset | Content type | 8K speedup | 16K speedup | 24K speedup |
|---------|--------------|------------|-------------|-------------|
| XSum | News summaries | 7.1x | 10.0x | 12.3x |
| CNN/DailyMail | Long-form journalism | 7.1x | 9.4x | 12.2x |
| MultiNews | Multi-document news | - | 9.3x | - |

## Quality

Quality was validated across 5 datasets, 4-5 context lengths each, with PPL ratio and
LLM-as-judge faithfulness scoring.

| Dataset | PPL range | Status | Judge cold | Judge SemBlend | Faithful |
|---------|-----------|--------|------------|----------------|----------|
| XSum | 1.018-1.054 | PASS | 0.84 | 0.84 | 100% |
| CNN/DailyMail | 1.011-1.049 | PASS | 0.87 | 0.86 | 97% |
| WikiHow | 0.987-1.037 | PASS | 0.82 | 0.84 | 97% |
| MultiNews | 0.958-1.064 | PASS | 0.79 | 0.78 | 100% |
| SAMSum | 1.140-1.198 | ELEVATED | 0.78 | 0.86 | 87% |

PPL was below 1.065 for 4 of 5 datasets at all tested lengths. SAMSum showed elevated PPL due
to short dialogue turns, while LLM-as-judge rated SemBlend output higher than cold in that cell.
Treat these as workload-specific validation results, not a universal guarantee.

## When SemBlend Helps

SemBlend is most useful when many requests share large semantic context regions:

- **Document Q&A / RAG**: same retrieved passages, different questions.
- **Summarization**: same article, different instruction phrasing.
- **Agent workflows**: repeated tool traces, plans, or memory summaries.
- **Code assistance**: shared repository or file context across requests.
- **Enterprise knowledge workloads**: repeated policies, contracts, tickets, or manuals.

## When SemBlend Is Usually Not Worth It

- Short prompts where lookup overhead can dominate.
- Fully novel traffic with low donor density.

## Benchmarks and Reproduction

Install benchmark dependencies:

```bash
pip install semblend[benchmarks]
```

The public package contains benchmark and test scaffolding, but benchmark numbers should always
be reported with model, backend, cache configuration, hardware, dataset, context length, and
whether reuse was backend-confirmed or discovery-only.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[Apache License 2.0](LICENSE).

Built at [WorldFlow AI](https://worldflowai.com). For support contact
[research@worldflowai.com](mailto:research@worldflowai.com).

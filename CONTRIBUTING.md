# Contributing to SemBlend

Thank you for your interest in contributing. SemBlend is an active research project and welcomes contributions across the core engine, engine integrations, benchmarks, and documentation.

## Ways to Contribute

- **Bug reports** — open a [GitHub issue](https://github.com/worldflowai/semblend/issues) with a minimal reproducer
- **Engine integrations** — TRT-LLM, MLC-LLM, or other inference backends
- **Embedder backends** — alternative embedding models or ANN index backends
- **Benchmarks** — new datasets, workloads, or latency measurement methodologies
- **Documentation** — clarifications, examples, tutorials

## Development Setup

```bash
git clone https://github.com/worldflowai/semblend
cd semblend

python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev,embedder]"
```

## Running Tests

```bash
pytest                      # unit tests (CPU-only, no GPU required)
pytest -m integration       # requires vLLM + LMCache installed
```

All tests must pass before submitting a PR. GPU-dependent tests (Triton kernels, RoPE correction) are excluded from the default run and marked `@pytest.mark.gpu`.

## Code Style

```bash
ruff check .        # lint
ruff format .       # format
```

Line length is 100. Type annotations are expected on public functions and class methods. We follow standard Python conventions (PEP 8, PEP 484).

## Pull Request Process

1. Fork the repo and create a branch from `main`: `git checkout -b feat/my-change`
2. Write tests for any new behavior — aim for coverage on the changed code paths
3. Ensure `pytest` and `ruff check .` both pass locally
4. Open a PR with a clear description of what the change does and why
5. Link any related GitHub issues or upstream PRs (LMCache, vLLM, SGLang)

PRs that touch `semblend_core/` (the backend-agnostic engine) must not introduce dependencies on any specific inference engine. Engine-specific code belongs in `semblend/integration/`.

## Architecture Overview

```
semblend_core/          Backend-agnostic pipeline (embed → search → align → plan)
  pipeline.py           5-stage orchestrator
  embedder.py           MiniLM, Jaccard, ONNX-GPU embedders
  alignment.py          Exact and fuzzy chunk alignment
  rope_correction.py    RoPE delta correction + NoPE two-step
  bathtub.py            Per-layer deviation scoring (CacheBlend)

semblend/integration/
  vllm/                 vLLM + LMCache connector (KVConnectorBase_V1)
  sglang/               SGLang RadixCache patcher + SemanticPrefixProvider

synapse_kv_connector/   Legacy vLLM connector (thin re-export for backward compat)
tests/                  Unit tests
```

## Upstream Interface Work

SemBlend is driving standardized semantic caching interfaces upstream:

- **LMCache** — [`SemanticLookupProvider`](https://github.com/LMCache/LMCache/pull/2803) and [`PostLoadHook`](https://github.com/LMCache/LMCache/pull/2804)
- **SGLang** — [`SemanticPrefixProvider`](https://github.com/sgl-project/sglang/pull/20806)
- **vLLM** — [`register_model` for CacheBlend](https://github.com/vllm-project/vllm/pull/37339)

If you're working on an engine integration, coordinate with these upstream PRs to avoid duplicate work.

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).

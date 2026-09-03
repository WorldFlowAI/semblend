"""The adapter builds the core pipeline with min_reuse_ratio 0 in canonical
mode (it applies its own floor later). The consumable-coverage gate must
keep the real floor, or a scattered page-level match is returned as reuse
and the pipeline's top-k paraphrase probe never runs (measured on GPU:
60 of 60 paraphrases rejected by the adapter's single-candidate check
while the same items served 48 of 48 through the pipeline probe)."""

from __future__ import annotations

from semblend.integration.sglang.config import SemBlendProviderConfig
from semblend.integration.sglang.provider import SemBlendProviderAdapter


def test_build_pipeline_keeps_consumable_floor_in_canonical_mode(monkeypatch):
    import semblend_core.pipeline as core

    captured = {}

    class _Recorder:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(core, "SemBlendPipeline", _Recorder)
    monkeypatch.setenv("SEMBLEND_CANONICAL_MATCH", "1")
    config = SemBlendProviderConfig(min_reuse_ratio=0.5, model_arch="llama")
    SemBlendProviderAdapter._build_pipeline(config)  # noqa: SLF001
    assert captured["min_reuse_ratio"] == 0.0
    assert captured["min_consumable_coverage"] == 0.5


def test_pipeline_consumable_floor_defaults_to_reuse_floor():
    from semblend_core.pipeline import SemBlendPipeline

    p = SemBlendPipeline(min_reuse_ratio=0.4, embedder_type="jaccard")
    assert p._min_consumable_coverage == 0.4  # noqa: SLF001
    q = SemBlendPipeline(min_reuse_ratio=0.0, min_consumable_coverage=0.5, embedder_type="jaccard")
    assert q._min_consumable_coverage == 0.5  # noqa: SLF001

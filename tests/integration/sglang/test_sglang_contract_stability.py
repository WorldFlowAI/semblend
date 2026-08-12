"""Contract ratchet: the sglang wrapper (merged separately, versioned
independently) must keep working against ANY future semblend release.

The upstream sglang PR depends on `semblend>=0.3.12` with NO upper bound,
lazily imports these exact symbols, calls these exact signatures, and
copies these exact result fields. Renaming a field, making a new field
required, or breaking a signature breaks deployed engines at runtime on a
routine pip upgrade. Every assertion here is an external-source literal
(the sglang-side wrapper's usage); extend ONLY additively.
"""

import inspect

from semblend.integration.sglang.config import SemBlendProviderConfig
from semblend.integration.sglang.provider import SemBlendProviderAdapter
from semblend.integration.sglang.types import (
    FuzzyMatchResult,
    FuzzyMatchSegment,
    QualitySignals,
)


def test_adapter_public_methods_keep_wrapper_signatures():
    m = inspect.signature(SemBlendProviderAdapter.match)
    params = list(m.parameters)
    # wrapper calls: match(prompt_token_ids, already_matched_len,
    #                      prompt_text=..., extra_key=...)
    assert params[:3] == ["self", "prompt_token_ids", "already_matched_len"]
    assert "prompt_text" in params and "extra_key" in params
    for name in ("prompt_text", "extra_key"):
        assert m.parameters[name].default is None

    r = inspect.signature(SemBlendProviderAdapter.register_donor)
    rp = list(r.parameters)
    assert rp[:6] == [
        "self",
        "request_id",
        "token_ids",
        "kv_cache",
        "cache_start_pos",
        "cache_end_pos",
    ]
    for name in ("prompt_text", "extra_key", "radix_tree"):
        assert name in r.parameters
        assert r.parameters[name].default is not inspect.Parameter.empty


def test_result_fields_wrapper_reads_stay_present_and_optional_extras_default():
    fields = {f.name: f for f in FuzzyMatchResult.__dataclass_fields__.values()}
    # fields the wrapper reads directly (required, must never rename)
    for name in (
        "cached_token_count",
        "cached_token_ids",
        "prompt_token_count",
        "kv_cache_indices",
        "position_offset",
        "cached_start_pos",
        "segments",
        "layer_recompute_mask",
        "quality_signals",
        "_match_entry",
    ):
        assert name in fields, f"wrapper-required field renamed/removed: {name}"
    # every field AFTER the first six core ones must carry a default so a
    # wrapper constructing/copying old-style never breaks
    import dataclasses as dc

    for f in list(FuzzyMatchResult.__dataclass_fields__.values())[6:]:
        assert (
            f.default is not dc.MISSING or f.default_factory is not dc.MISSING
        ), f"new FuzzyMatchResult field must be optional: {f.name}"


def test_segment_fields_wrapper_reads_stay_present():
    fields = {f.name for f in FuzzyMatchSegment.__dataclass_fields__.values()}
    for name in (
        "target_positions",
        "donor_positions",
        "donor_node_id",
        "donor_offset",
        "length",
        "donor_kv_indices",
        "donor_req_id",
        "layer_recompute_mask",
    ):
        assert name in fields


def test_quality_signal_fields_wrapper_reads_stay_present():
    fields = {f.name for f in QualitySignals.__dataclass_fields__.values()}
    for name in (
        "cosine_similarity",
        "reuse_ratio",
        "confidence_tier",
        "passed_quality_gate",
        "rejection_reason",
    ):
        assert name in fields


def test_config_constructor_accepts_wrapper_kwargs():
    sig = inspect.signature(SemBlendProviderConfig)
    for name in ("model_arch", "block_size", "min_similarity", "min_reuse_ratio"):
        assert name in sig.parameters


def test_full_wrapper_surface_pinned():
    """The complete call surface of the engine-side wrapper (extracted
    from the integration source): every attribute it touches, with the
    exact keyword shapes. Any rename here breaks deployed engines that
    upgrade this library, so this test failing means the change must be
    additive instead."""
    import inspect

    # attributes the wrapper touches, including the executor fallback
    for attr in ("register_donor", "match", "on_donor_inserted", "clear"):
        assert callable(getattr(SemBlendProviderAdapter, attr, None)), attr
    # instances expose the executor handle the wrapper's reset path probes
    assert "_register_executor" in inspect.getsource(SemBlendProviderAdapter)

    sig = inspect.signature(SemBlendProviderAdapter.on_donor_inserted)
    for name in ("request_id", "donor_last_node_id"):
        assert name in sig.parameters

    # from_dict with the wrapper's exact key set must construct cleanly
    cfg = SemBlendProviderConfig.from_dict(
        {
            "min_similarity": 0.6,
            "min_reuse_ratio": 0.5,
            "min_match_length": 16,
            "max_entries": 100,
            "block_size": 16,
            "embedding_model_name": None,
            "model_arch": "qwen2.5-7b",
        }
    )
    assert cfg.min_match_length == 16

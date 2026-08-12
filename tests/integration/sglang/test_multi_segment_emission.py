"""Multi-segment emission (v0.4 line): gated emission vs the 0.3.x collapse."""

from __future__ import annotations

import pytest

from semblend.integration.sglang.config import SemBlendProviderConfig
from semblend.integration.sglang.provider import SemBlendProviderAdapter

from .test_provider import _StubPipeline, _StubPipelineResult, _StubPosMap


def _make(config_kwargs=None):
    config = SemBlendProviderConfig(
        min_similarity=0.60,
        min_reuse_ratio=0.50,
        min_match_length=8,
        max_entries=100,
        block_size=4,
        enable_bathtub=False,
        model_arch="llama",
        **(config_kwargs or {}),
    )
    pipeline = _StubPipeline()
    return SemBlendProviderAdapter(config=config, pipeline=pipeline), pipeline


def _register(adapter, donor_tokens):
    adapter.register_donor(
        request_id="donor-A",
        token_ids=list(donor_tokens),
        kv_cache=list(range(100, 100 + len(donor_tokens))),
        cache_start_pos=0,
        cache_end_pos=len(donor_tokens),
        prompt_text="registration",
    )


def _two_run_result(donor_tokens):
    """Two aligned runs: [0..31]→[4..35] and [40..71]→[36..67] (gap breaks them)."""
    # Run1 shifted by +4 (position-aligned runs are the exact cache's job
    # and get gated); run2 shifted by -4.
    donor_positions = list(range(0, 32)) + list(range(40, 72))
    target_positions = list(range(4, 36)) + list(range(36, 68))
    return _StubPipelineResult(
        found=True,
        donor_id="donor-A",
        similarity=0.8,
        reuse_ratio=0.7,
        donor_tokens=list(donor_tokens),
        position_map=_StubPosMap(donor_positions, target_positions),
        layer_deviations=[],
        confidence_tier="fuzzy",
    )


def _matching_target(donor_tokens):
    """Target whose tokens equal the donor's at every aligned position."""
    target = [-1] * 80
    for d, t in zip(range(0, 32), range(4, 36)):
        target[t] = donor_tokens[d]
    for d, t in zip(range(40, 72), range(36, 68)):
        target[t] = donor_tokens[d]
    return [tok if tok != -1 else 999_000 + i for i, tok in enumerate(target)]


DONOR = list(range(1000, 1080))


def test_emission_off_collapses_exactly_like_v03():
    adapter, pipeline = _make()  # multi_segment_emission defaults False
    _register(adapter, DONOR)
    pipeline.next_result = _two_run_result(DONOR)
    result = adapter.match(
        prompt_token_ids=_matching_target(DONOR), already_matched_len=0, prompt_text="q"
    )
    assert result is not None
    assert result.segments is None  # the 0.3.x collapse
    assert result.cached_token_count == 32  # head run only


def test_emission_on_emits_gated_segments():
    adapter, pipeline = _make({"multi_segment_emission": True})
    _register(adapter, DONOR)
    pipeline.next_result = _two_run_result(DONOR)
    result = adapter.match(
        prompt_token_ids=_matching_target(DONOR), already_matched_len=0, prompt_text="q"
    )
    assert result is not None
    # Default merge limit concatenates both runs into one scatter segment.
    assert result.segments is not None and len(result.segments) == 1
    assert len(list(result.segments[0].target_positions)) == 64
    # Contract Option A: cached_token_count keeps v1 head-run semantics so
    # the radix layer's kv-indices consistency check still holds; segments
    # carry the full gated plan for the segmented realizer.
    assert result.cached_token_count == 32


def test_env_var_parity(monkeypatch):
    adapter, pipeline = _make()  # config off
    monkeypatch.setenv("SEMBLEND_RETURN_SEGMENTS", "1")
    _register(adapter, DONOR)
    pipeline.next_result = _two_run_result(DONOR)
    result = adapter.match(
        prompt_token_ids=_matching_target(DONOR), already_matched_len=0, prompt_text="q"
    )
    assert result.segments is not None


def test_short_segments_dropped():
    adapter, pipeline = _make(
        {"multi_segment_emission": True, "segment_min_tokens": 40}
    )
    _register(adapter, DONOR)
    pipeline.next_result = _two_run_result(DONOR)
    result = adapter.match(
        prompt_token_ids=_matching_target(DONOR), already_matched_len=0, prompt_text="q"
    )
    # Both runs are 32 tokens < 40: all gated out -> MISS. (The old ungated
    # head fallback bypassed every safety gate and wedged the engine.)
    assert result is None
    assert adapter._stats.segments_dropped_short == 2


def test_entity_swap_shaped_segments_dropped_by_identity_gate():
    """Aligned runs whose tokens differ every ~10 positions (entity tags) sit
    below the identity gate and must not be emitted."""
    adapter, pipeline = _make({"multi_segment_emission": True})
    _register(adapter, DONOR)
    pipeline.next_result = _two_run_result(DONOR)
    target = _matching_target(DONOR)
    # Corrupt every 10th aligned position in both runs (entity-tag mismatch).
    for t in list(range(0, 32, 10)) + list(range(36, 68, 10)):
        target[t] = 555_000 + t
    result = adapter.match(
        prompt_token_ids=target, already_matched_len=0, prompt_text="q"
    )
    assert result is None  # identity ~0.9 < 0.98 gates both runs -> miss
    assert adapter._stats.segments_dropped_low_identity == 2


def test_identity_gate_configurable():
    adapter, pipeline = _make(
        {"multi_segment_emission": True, "segment_min_token_identity": 0.85}
    )
    _register(adapter, DONOR)
    pipeline.next_result = _two_run_result(DONOR)
    target = _matching_target(DONOR)
    for t in list(range(0, 32, 10)) + list(range(36, 68, 10)):
        target[t] = 555_000 + t
    result = adapter.match(
        prompt_token_ids=target, already_matched_len=0, prompt_text="q"
    )
    # identity ~0.90 >= 0.85: emitted at the looser gate (merged to one).
    assert result.segments is not None
    assert sum(len(list(s.target_positions)) for s in result.segments) == 64


def test_segments_merge_into_scatter_buckets():
    adapter, pipeline = _make(
        {"multi_segment_emission": True, "segment_merge_max_positions": 100}
    )
    _register(adapter, DONOR)
    pipeline.next_result = _two_run_result(DONOR)
    result = adapter.match(
        prompt_token_ids=_matching_target(DONOR), already_matched_len=0, prompt_text="q"
    )
    # Two 32-token runs fit one 100-position bucket -> ONE merged segment
    # containing all 64 positions in order.
    assert result.segments is not None and len(result.segments) == 1
    merged = result.segments[0]
    targets = list(merged.target_positions)
    assert len(targets) == 64
    assert targets == list(range(4, 36)) + list(range(36, 68))
    assert len(list(merged.donor_kv_indices)) == 64


def test_merge_respects_position_limit():
    adapter, pipeline = _make(
        {"multi_segment_emission": True, "segment_merge_max_positions": 40}
    )
    _register(adapter, DONOR)
    pipeline.next_result = _two_run_result(DONOR)
    result = adapter.match(
        prompt_token_ids=_matching_target(DONOR), already_matched_len=0, prompt_text="q"
    )
    # 32+32 exceeds the 40-position bucket -> stays two segments.
    assert result.segments is not None and len(result.segments) == 2


def test_merge_disabled_with_zero_limit():
    adapter, pipeline = _make(
        {"multi_segment_emission": True, "segment_merge_max_positions": 0}
    )
    _register(adapter, DONOR)
    pipeline.next_result = _two_run_result(DONOR)
    result = adapter.match(
        prompt_token_ids=_matching_target(DONOR), already_matched_len=0, prompt_text="q"
    )
    assert result.segments is not None and len(result.segments) == 2


def test_segment_min_tokens_env_override(monkeypatch):
    adapter, pipeline = _make({"multi_segment_emission": True})
    monkeypatch.setenv("SEMBLEND_SEGMENT_MIN_TOKENS", "40")
    _register(adapter, DONOR)
    pipeline.next_result = _two_run_result(DONOR)
    result = adapter.match(
        prompt_token_ids=_matching_target(DONOR), already_matched_len=0, prompt_text="q"
    )
    # 32-token runs < 40 env-override: all gated out -> miss.
    assert result is None
    assert adapter._stats.segments_dropped_short == 2


def test_trim_sink_rebases_noderef_addressing():
    from semblend.integration.sglang.types import FuzzyMatchSegment

    seg = FuzzyMatchSegment(
        target_positions=list(range(0, 32)),
        donor_positions=list(range(0, 32)),
        donor_node_id=7,
        donor_offset=0,
        length=32,
    )
    out = SemBlendProviderAdapter._trim_sink(seg, 16)
    assert out.target_positions == list(range(16, 32))
    assert out.donor_positions == list(range(16, 32))
    assert out.donor_offset == 16
    assert out.length == 16
    # Fully-protected run drops entirely.
    assert SemBlendProviderAdapter._trim_sink(seg, 32) is None
    # Run past the sink is returned unchanged.
    assert SemBlendProviderAdapter._trim_sink(seg, 0) is seg


def test_sink_protect_trims_first_run():
    adapter, pipeline = _make(
        {"multi_segment_emission": True, "sink_protect_tokens": 16}
    )
    _register(adapter, DONOR)
    pipeline.next_result = _two_run_result(DONOR)
    result = adapter.match(
        prompt_token_ids=_matching_target(DONOR), already_matched_len=0, prompt_text="q"
    )
    assert result.segments is not None
    positions = []
    for seg in result.segments:
        positions.extend(list(seg.target_positions))
    assert positions == list(range(16, 36)) + list(range(36, 68))


def test_sink_protect_drops_fully_covered_run():
    adapter, pipeline = _make(
        {"multi_segment_emission": True, "sink_protect_tokens": 36}
    )
    _register(adapter, DONOR)
    pipeline.next_result = _two_run_result(DONOR)
    result = adapter.match(
        prompt_token_ids=_matching_target(DONOR), already_matched_len=0, prompt_text="q"
    )
    assert adapter._stats.segments_dropped_sink == 1
    assert result.segments is not None
    positions = []
    for seg in result.segments:
        positions.extend(list(seg.target_positions))
    assert positions == list(range(36, 68))


def test_sink_protect_env_override(monkeypatch):
    adapter, pipeline = _make({"multi_segment_emission": True})
    monkeypatch.setenv("SEMBLEND_SINK_PROTECT_TOKENS", "36")
    _register(adapter, DONOR)
    pipeline.next_result = _two_run_result(DONOR)
    result = adapter.match(
        prompt_token_ids=_matching_target(DONOR), already_matched_len=0, prompt_text="q"
    )
    assert adapter._stats.segments_dropped_sink == 1
    assert all(p >= 36 for seg in result.segments for p in list(seg.target_positions))


def test_trim_sink_donor_key_rebases():
    from semblend.integration.sglang.types import FuzzyMatchSegment

    seg = FuzzyMatchSegment(
        target_positions=list(range(100, 132)),
        donor_positions=list(range(0, 32)),
        donor_node_id=7,
        donor_offset=0,
        length=32,
    )
    out = SemBlendProviderAdapter._trim_sink(seg, 16, key="donor")
    assert out.donor_positions == list(range(16, 32))
    assert out.target_positions == list(range(116, 132))
    assert out.donor_offset == 16 and out.length == 16
    assert SemBlendProviderAdapter._trim_sink(seg, 32, key="donor") is None
    # Donor run past its sink is untouched even when targets are early.
    late = FuzzyMatchSegment(
        target_positions=list(range(0, 32)),
        donor_positions=list(range(200, 232)),
        donor_node_id=7,
        donor_offset=200,
        length=32,
    )
    assert SemBlendProviderAdapter._trim_sink(late, 32, key="donor") is late


def test_donor_sink_protect_env_gate(monkeypatch):
    adapter, pipeline = _make({"multi_segment_emission": True})
    monkeypatch.setenv("SEMBLEND_DONOR_SINK_PROTECT_TOKENS", "40")
    _register(adapter, DONOR)
    pipeline.next_result = _two_run_result(DONOR)
    result = adapter.match(
        prompt_token_ids=_matching_target(DONOR), already_matched_len=0, prompt_text="q"
    )
    # Run1 (donor 0..31) fully inside donor sink -> dropped; run2 (donor 40..71) kept.
    assert adapter._stats.segments_dropped_donor_sink == 1
    positions = [p for seg in result.segments for p in list(seg.target_positions)]
    assert positions == list(range(36, 68))


def test_trim_run_edges_rebases_and_drops():
    from semblend.integration.sglang.types import FuzzyMatchSegment

    seg = FuzzyMatchSegment(
        target_positions=list(range(100, 132)),
        donor_positions=list(range(0, 32)),
        donor_node_id=7,
        donor_offset=0,
        length=32,
    )
    out = SemBlendProviderAdapter._trim_run_edges(seg, 8, 8)
    assert out.target_positions == list(range(108, 124))
    assert out.donor_positions == list(range(8, 24))
    assert out.donor_offset == 8 and out.length == 16
    # Head-only (donor-run-head redesign path).
    out2 = SemBlendProviderAdapter._trim_run_edges(seg, 16, 0)
    assert out2.target_positions == list(range(116, 132))
    # Fully consumed run drops.
    assert SemBlendProviderAdapter._trim_run_edges(seg, 16, 16) is None
    # No-op when both zero.
    assert SemBlendProviderAdapter._trim_run_edges(seg, 0, 0) is seg


def test_edge_trim_env_gate(monkeypatch):
    adapter, pipeline = _make({"multi_segment_emission": True, "segment_min_tokens": 8})
    monkeypatch.setenv("SEMBLEND_SEGMENT_EDGE_TRIM", "8")
    _register(adapter, DONOR)
    pipeline.next_result = _two_run_result(DONOR)
    result = adapter.match(
        prompt_token_ids=_matching_target(DONOR), already_matched_len=0, prompt_text="q"
    )
    positions = [p for seg in result.segments for p in list(seg.target_positions)]
    # Run1 targets 0..31 -> 8..23; run2 targets 36..67 -> 44..59.
    assert positions == list(range(12, 28)) + list(range(44, 60))


def test_head_trim_combines_with_edge_via_max(monkeypatch):
    adapter, pipeline = _make({"multi_segment_emission": True, "segment_min_tokens": 8})
    monkeypatch.setenv("SEMBLEND_SEGMENT_EDGE_TRIM", "4")
    monkeypatch.setenv("SEMBLEND_DONOR_RUN_HEAD_TRIM", "12")
    _register(adapter, DONOR)
    pipeline.next_result = _two_run_result(DONOR)
    result = adapter.match(
        prompt_token_ids=_matching_target(DONOR), already_matched_len=0, prompt_text="q"
    )
    positions = [p for seg in result.segments for p in list(seg.target_positions)]
    # head = max(4, 12) = 12, tail = 4: run1 0..31 -> 12..27; run2 36..67 -> 48..63.
    assert positions == list(range(16, 32)) + list(range(48, 64))


def test_tail_reserve_and_position_aligned_gates():
    from semblend.integration.sglang.types import FuzzyMatchSegment

    adapter, _ = _make({"multi_segment_emission": True})
    # Position-aligned run drops entirely.
    aligned = FuzzyMatchSegment(
        target_positions=list(range(10, 42)),
        donor_positions=list(range(10, 42)),
        donor_node_id=1, donor_offset=10, length=32,
    )
    shifted = FuzzyMatchSegment(
        target_positions=list(range(100, 132)),
        donor_positions=list(range(0, 32)),
        donor_node_id=1, donor_offset=0, length=32,
    )
    kept = adapter._gate_segments([aligned, shifted], [0] * 200, [0] * 200)
    assert adapter._stats.segments_dropped_position_aligned == 1
    assert len(kept) == 1 and list(kept[0].target_positions)[0] == 100

    # Tail reserve applies on long prompts only (> 4x reserve).
    long_target = [0] * 1000
    tail_run = FuzzyMatchSegment(
        target_positions=list(range(950, 1000)),
        donor_positions=list(range(50, 100)),
        donor_node_id=1, donor_offset=50, length=50,
    )
    kept2 = adapter._gate_segments([tail_run], [0] * 1000, long_target)
    # boundary = 1000 - 64 = 936 -> run 950.. fully reserved -> dropped
    assert adapter._stats.segments_dropped_tail_reserve == 1
    assert kept2 == []
    mid_run = FuzzyMatchSegment(
        target_positions=list(range(900, 950)),
        donor_positions=list(range(100, 150)),
        donor_node_id=1, donor_offset=100, length=50,
    )
    kept3 = adapter._gate_segments([mid_run], [0] * 1000, long_target)
    # trimmed at 936 -> keeps 900..935
    assert list(kept3[0].target_positions) == list(range(900, 936))


def test_single_boundary_anchored_run_routes_contiguous():
    from semblend.integration.sglang.types import FuzzyMatchSegment

    adapter, pipeline = _make({"multi_segment_emission": True})
    _register(adapter, DONOR)
    # One run anchored at the prefix boundary (tail-relative target start 0),
    # shifted donor (delta != 0 so the position-aligned gate keeps it).
    donor_positions = list(range(8, 40))
    target_positions = list(range(0, 32))
    pipeline.next_result = _StubPipelineResult(
        found=True, donor_id="donor-A", similarity=0.8, reuse_ratio=0.7,
        donor_tokens=list(DONOR),
        position_map=_StubPosMap(donor_positions, target_positions),
        layer_deviations=[], confidence_tier="fuzzy",
    )
    target = [-1] * 80
    for d, t in zip(range(8, 40), range(0, 32)):
        target[t] = DONOR[d]
    target = [tok if tok != -1 else 999_000 + i for i, tok in enumerate(target)]
    result = adapter.match(
        prompt_token_ids=target, already_matched_len=0, prompt_text="q"
    )
    assert result is not None
    assert result.segments is None  # contiguous contract
    assert result.cached_token_count == 32
    assert result.cached_start_pos == 8


def test_interior_single_run_stays_scatter():
    adapter, pipeline = _make({"multi_segment_emission": True})
    _register(adapter, DONOR)
    pipeline.next_result = _two_run_result(DONOR)
    result = adapter.match(
        prompt_token_ids=_matching_target(DONOR), already_matched_len=0, prompt_text="q"
    )
    # Two runs, first starts at 4 != already_matched 0 -> scatter path.
    assert result.segments is not None

"""Unit tests for SemBlendProviderAdapter.

Tests run without loading the real MiniLM embedder or SGLang: the adapter
accepts an injected `pipeline` object so we can stub its behavior. A few
smoke tests exercise the real pipeline construction path when the
environment can load it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pytest

from semblend.integration.sglang.config import SemBlendProviderConfig
from semblend.integration.sglang.provider import SemBlendProviderAdapter
from semblend.integration.sglang.types import (
    FuzzyMatchResult,
    QualitySignals,
)

# ---------------------------------------------------------------------
# Stub pipeline / donor store
# ---------------------------------------------------------------------


@dataclass
class _StubEmbedder:
    dim: int = 384

    def embed(self, text: str) -> np.ndarray:
        # Deterministic non-zero embedding keyed on text length; enough to
        # avoid "no embedder" rejection path. Tests don't exercise semantic
        # quality — they exercise the adapter's plumbing.
        vec = np.zeros(self.dim, dtype=np.float32)
        vec[len(text) % self.dim] = 1.0
        return vec


@dataclass
class _StubDonorStore:
    donors: list = field(default_factory=list)

    def add_donor(self, node: Any) -> None:
        self.donors.append(node)


@dataclass
class _StubPosMap:
    donor_positions: list
    target_positions: list


@dataclass
class _StubPipelineResult:
    found: bool
    donor_id: Optional[str] = None
    similarity: float = 0.0
    reuse_ratio: float = 0.0
    donor_tokens: list = field(default_factory=list)
    slot_actions: list = field(default_factory=list)
    layer_deviations: list = field(default_factory=list)
    position_map: _StubPosMap = field(default_factory=lambda: _StubPosMap([], []))
    confidence_tier: str = "exact"


class _StubPipeline:
    """Minimal stand-in for SemBlendPipeline.

    Configurable: the test sets `next_result` and `register_donor` pushes
    into `donors`. No real embedding or search.
    """

    def __init__(self) -> None:
        self._embedder = _StubEmbedder()
        self._donor_store = _StubDonorStore()
        self.next_result: _StubPipelineResult = _StubPipelineResult(found=False)
        self.find_donor_calls: list = []

    def find_donor(self, token_ids, prompt_text="", top_k=5):
        self.find_donor_calls.append(
            {"token_ids": list(token_ids), "prompt_text": prompt_text, "top_k": top_k}
        )
        return self.next_result


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def config() -> SemBlendProviderConfig:
    return SemBlendProviderConfig(
        min_similarity=0.60,
        min_reuse_ratio=0.50,
        min_match_length=8,
        max_entries=100,
        block_size=4,
        enable_bathtub=True,
        model_arch="llama",
    )


@pytest.fixture
def pipeline() -> _StubPipeline:
    return _StubPipeline()


@pytest.fixture
def adapter(config, pipeline) -> SemBlendProviderAdapter:
    return SemBlendProviderAdapter(config=config, pipeline=pipeline)


# ---------------------------------------------------------------------
# register_donor
# ---------------------------------------------------------------------


class TestRegisterDonor:
    def test_adds_donor_to_store(self, adapter, pipeline):
        ok = adapter.register_donor(
            request_id="req-1",
            token_ids=list(range(16)),
            kv_cache=list(range(100, 116)),
            cache_start_pos=0,
            cache_end_pos=16,
            prompt_text="hello world",
        )
        assert ok is True
        assert len(pipeline._donor_store.donors) == 1
        node = pipeline._donor_store.donors[0]
        assert node.request_id == "req-1"
        assert node.token_ids == list(range(16))
        assert adapter.donor_count() == 1

    def test_rejects_short_sequences(self, adapter, pipeline):
        # min_match_length is 8 in fixture; 4 tokens is too short.
        ok = adapter.register_donor(
            request_id="req-short",
            token_ids=list(range(4)),
            kv_cache=list(range(100, 104)),
            cache_start_pos=0,
            cache_end_pos=4,
            prompt_text="short",
        )
        assert ok is False
        assert pipeline._donor_store.donors == []
        stats = adapter.stats()
        assert stats["register_rejected"] == 1
        assert stats["register_ok"] == 0

    def test_rejects_when_no_prompt_text(self, adapter):
        # Adapter refuses to synthesize text — SGLang wrapper owns the
        # tokenizer, so without prompt_text we can't embed.
        ok = adapter.register_donor(
            request_id="req-notext",
            token_ids=list(range(16)),
            kv_cache=list(range(100, 116)),
            cache_start_pos=0,
            cache_end_pos=16,
            prompt_text=None,
        )
        assert ok is False


# ---------------------------------------------------------------------
# match — miss paths
# ---------------------------------------------------------------------


class TestMatchMisses:
    def test_miss_when_remaining_too_short(self, adapter, pipeline):
        result = adapter.match(prompt_token_ids=list(range(4)), already_matched_len=0)
        assert result is None
        # Should not even hit the pipeline
        assert pipeline.find_donor_calls == []

    def test_miss_when_pipeline_not_found(self, adapter, pipeline):
        pipeline.next_result = _StubPipelineResult(found=False)
        result = adapter.match(
            prompt_token_ids=list(range(32)),
            already_matched_len=0,
            prompt_text="query",
        )
        assert result is None
        stats = adapter.stats()
        assert stats["match_misses"] == 1

    def test_miss_when_similarity_below_threshold(self, adapter, pipeline):
        pipeline.next_result = _StubPipelineResult(
            found=True,
            donor_id="donor-low",
            similarity=0.40,  # below fixture threshold 0.60
            reuse_ratio=0.8,
        )
        result = adapter.match(
            prompt_token_ids=list(range(32)),
            already_matched_len=0,
            prompt_text="query",
        )
        assert result is None

    def test_miss_when_reuse_ratio_below_floor(self, adapter, pipeline):
        pipeline.next_result = _StubPipelineResult(
            found=True,
            donor_id="donor-low-reuse",
            similarity=0.85,
            reuse_ratio=0.20,  # below fixture floor 0.50
        )
        result = adapter.match(
            prompt_token_ids=list(range(32)),
            already_matched_len=0,
            prompt_text="query",
        )
        assert result is None
        stats = adapter.stats()
        assert stats["match_rejected_low_reuse"] == 1

    def test_miss_when_donor_kv_not_registered(self, adapter, pipeline):
        # Pipeline says found, but adapter never registered this donor_id
        # so the handle is missing — must treat as miss, not crash.
        pipeline.next_result = _StubPipelineResult(
            found=True,
            donor_id="donor-phantom",
            similarity=0.85,
            reuse_ratio=0.8,
            position_map=_StubPosMap(donor_positions=[0, 1], target_positions=[0, 1]),
        )
        result = adapter.match(
            prompt_token_ids=list(range(32)),
            already_matched_len=0,
            prompt_text="query",
        )
        assert result is None
        stats = adapter.stats()
        assert stats["match_rejected_no_kv"] == 1


# ---------------------------------------------------------------------
# match — hit paths
# ---------------------------------------------------------------------


class TestMatchHits:
    def _register_donor(self, adapter, request_id="donor-A", kv=None, nt=16):
        adapter.register_donor(
            request_id=request_id,
            token_ids=list(range(nt)),
            kv_cache=kv if kv is not None else list(range(100, 100 + nt)),
            cache_start_pos=0,
            cache_end_pos=nt,
            prompt_text="registration text",
        )

    def test_contiguous_single_segment_collapses_to_legacy_shape(self, adapter, pipeline):
        self._register_donor(adapter, nt=16)
        # Perfectly contiguous alignment: one segment, both deltas = 1.
        pipeline.next_result = _StubPipelineResult(
            found=True,
            donor_id="donor-A",
            similarity=0.90,
            reuse_ratio=0.85,
            donor_tokens=list(range(16)),
            position_map=_StubPosMap(
                donor_positions=list(range(16)),
                target_positions=list(range(16)),
            ),
            layer_deviations=[
                {"layerIdx": i, "deviationScore": 0.1, "shouldRecompute": False}
                for i in range(4)
            ],
            confidence_tier="exact",
        )

        result = adapter.match(
            prompt_token_ids=list(range(32)),
            already_matched_len=0,
            prompt_text="query",
        )
        assert isinstance(result, FuzzyMatchResult)
        # Single-segment cases collapse to the legacy contiguous path.
        assert result.segments is None
        # Layer mask is populated even in single-segment mode.
        assert result.layer_recompute_mask == [False] * 4
        assert result.cached_start_pos == 0
        assert result.cached_token_count == 16
        assert result.cached_token_ids == list(range(16))
        assert result.position_offset == 0
        # kv_cache_indices slices the donor's handle.
        assert list(result.kv_cache_indices) == list(range(100, 116))
        # Quality signals preserved.
        assert isinstance(result.quality_signals, QualitySignals)
        assert result.quality_signals.cosine_similarity == pytest.approx(0.90)
        assert result.quality_signals.reuse_ratio == pytest.approx(0.85)
        assert result.quality_signals.confidence_tier == "exact"

    def test_reorder_picks_longest_run_as_match_block(self, adapter, pipeline):
        """match_block: the adapter surfaces the LONGEST contiguous monotonic run
        from the position_map as a single match_block. The block may start at
        any target position (not necessarily 0); SGLang's two-pass
        forward_extend handles the surrounding lead-in / main extends.
        """
        self._register_donor(adapter, nt=16)
        # Two contiguous runs with a reorder gap between them.
        # donor: 4..7 placed at target 0..3, then donor 0..3 at target 4..7.
        # Both runs are length 4; the adapter picks the first (head) one.
        pipeline.next_result = _StubPipelineResult(
            found=True,
            donor_id="donor-A",
            similarity=0.80,
            reuse_ratio=0.7,
            donor_tokens=list(range(16)),
            position_map=_StubPosMap(
                donor_positions=[4, 5, 6, 7, 0, 1, 2, 3],
                target_positions=[0, 1, 2, 3, 4, 5, 6, 7],
            ),
            layer_deviations=[],
            confidence_tier="fuzzy",
        )

        result = adapter.match(
            prompt_token_ids=list(range(32)),
            already_matched_len=0,
            prompt_text="query",
        )
        assert isinstance(result, FuzzyMatchResult)
        assert result.segments is None
        # match_block is set to the head run (donor[4..7] -> target[0..3]).
        assert result.match_block is not None
        assert result.match_block.target_start_in_prompt == 0
        assert result.match_block.length == 4
        assert result.match_block.donor_start == 4
        assert list(result.match_block.donor_kv_indices) == [104, 105, 106, 107]
        # cached_token_count == match_block length.
        assert result.cached_token_count == 4
        # cached_start_pos == donor's absolute position (for the legacy path).
        assert result.cached_start_pos == 4

    def test_match_block_at_non_prefix_position_accepted(
        self, adapter, pipeline
    ):
        """match_block: alignments whose longest run does NOT start at target 0
        are still accepted as a match_block. SGLang handles the lead-in via
        a pass-1 cold prefill before placing the block. This is the
        instruction-before-context paraphrase workload's primary path.
        """
        self._register_donor(adapter, nt=16)
        # Longest run starts at target=4 (lead-in needed for target 0..3).
        pipeline.next_result = _StubPipelineResult(
            found=True,
            donor_id="donor-A",
            similarity=0.85,
            reuse_ratio=0.6,
            donor_tokens=list(range(16)),
            position_map=_StubPosMap(
                donor_positions=[4, 5, 6, 7, 8],
                target_positions=[4, 5, 6, 7, 8],
            ),
            layer_deviations=[],
            confidence_tier="fuzzy",
        )

        result = adapter.match(
            prompt_token_ids=list(range(32)),
            already_matched_len=0,
            prompt_text="query",
        )
        assert result is not None
        assert result.match_block is not None
        # Block at absolute prompt position 4..8 (already_matched_len=0).
        assert result.match_block.target_start_in_prompt == 4
        assert result.match_block.length == 5
        assert result.match_block.donor_start == 4

    def test_match_block_dropped_below_min_length(self, adapter, pipeline):
        """match_block min-block threshold: tiny blocks aren't worth the two-pass
        orchestration overhead. Provider drops them.
        """
        self._register_donor(adapter, nt=16)
        # Only 2 contiguous matches; below MIN_BLOCK_LENGTH=4.
        pipeline.next_result = _StubPipelineResult(
            found=True,
            donor_id="donor-A",
            similarity=0.85,
            reuse_ratio=0.6,
            donor_tokens=list(range(16)),
            position_map=_StubPosMap(
                donor_positions=[4, 5],
                target_positions=[4, 5],
            ),
            layer_deviations=[],
            confidence_tier="fuzzy",
        )
        result = adapter.match(
            prompt_token_ids=list(range(32)),
            already_matched_len=0,
            prompt_text="query",
        )
        assert result is None
        stats = adapter.stats()
        assert stats["match_hits_dropped_no_prefix_run"] == 1

    def test_match_block_already_matched_len_offsets_to_absolute(
        self, adapter, pipeline
    ):
        """target positions in segments are slice-frame (0-based within
        remaining); the provider must translate to absolute prompt position
        by adding already_matched_len."""
        self._register_donor(adapter, nt=16)
        pipeline.next_result = _StubPipelineResult(
            found=True,
            donor_id="donor-A",
            similarity=0.9,
            reuse_ratio=0.7,
            donor_tokens=list(range(16)),
            position_map=_StubPosMap(
                donor_positions=[0, 1, 2, 3, 4],
                target_positions=[0, 1, 2, 3, 4],
            ),
            layer_deviations=[],
            confidence_tier="fuzzy",
        )
        result = adapter.match(
            prompt_token_ids=list(range(40)),
            already_matched_len=8,  # exact prefix already covered 8 tokens
            prompt_text="query",
        )
        assert result is not None
        assert result.match_block is not None
        # Block in absolute prompt frame: 8 + 0 = 8 .. 8 + 4 = 12.
        assert result.match_block.target_start_in_prompt == 8
        assert result.match_block.length == 5

    def test_node_id_carries_through_collapsed_head_segment(self, adapter, pipeline):
        """When on_donor_inserted has run for a donor, the collapsed result
        still surfaces the donor's ``donor_last_node_id`` so RadixCache can
        inc_lock_ref the donor node.
        """
        self._register_donor(adapter, nt=16)
        adapter.on_donor_inserted(request_id="donor-A", donor_last_node_id=4242)

        pipeline.next_result = _StubPipelineResult(
            found=True,
            donor_id="donor-A",
            similarity=0.80,
            reuse_ratio=0.7,
            donor_tokens=list(range(16)),
            position_map=_StubPosMap(
                donor_positions=[4, 5, 6, 7, 0, 1, 2, 3],
                target_positions=[0, 1, 2, 3, 4, 5, 6, 7],
            ),
            layer_deviations=[],
            confidence_tier="fuzzy",
        )

        result = adapter.match(
            prompt_token_ids=list(range(32)),
            already_matched_len=0,
            prompt_text="query",
        )
        assert result is not None
        assert result.donor_last_node_id == 4242
        # Collapsed to contiguous shape; segments field is None.
        assert result.segments is None

    def test_bathtub_mask_disabled_when_config_off(self, config, pipeline):
        cfg = SemBlendProviderConfig(**{**config.__dict__, "enable_bathtub": False})
        adapter = SemBlendProviderAdapter(config=cfg, pipeline=pipeline)
        adapter.register_donor(
            request_id="donor-A",
            token_ids=list(range(16)),
            kv_cache=list(range(100, 116)),
            cache_start_pos=0,
            cache_end_pos=16,
            prompt_text="hi",
        )
        pipeline.next_result = _StubPipelineResult(
            found=True,
            donor_id="donor-A",
            similarity=0.9,
            reuse_ratio=0.85,
            donor_tokens=list(range(16)),
            position_map=_StubPosMap(
                donor_positions=list(range(16)),
                target_positions=list(range(16)),
            ),
            layer_deviations=[
                {"layerIdx": i, "deviationScore": 0.9, "shouldRecompute": True}
                for i in range(4)
            ],
        )
        result = adapter.match(
            prompt_token_ids=list(range(32)),
            already_matched_len=0,
            prompt_text="q",
        )
        assert result is not None
        assert result.layer_recompute_mask is None

    def test_discovery_only_zeros_kv_but_keeps_quality_signals(self, config, pipeline):
        """discovery_only=True: hit happens, but SGLang sees cached_token_count=0."""
        cfg = SemBlendProviderConfig(**{**config.__dict__, "discovery_only": True})
        adapter = SemBlendProviderAdapter(config=cfg, pipeline=pipeline)
        adapter.register_donor(
            request_id="donor-A",
            token_ids=list(range(16)),
            kv_cache=list(range(100, 116)),
            cache_start_pos=0,
            cache_end_pos=16,
            prompt_text="hi",
        )
        pipeline.next_result = _StubPipelineResult(
            found=True,
            donor_id="donor-A",
            similarity=0.92,
            reuse_ratio=0.85,
            donor_tokens=list(range(16)),
            position_map=_StubPosMap(
                donor_positions=list(range(16)),
                target_positions=list(range(16)),
            ),
            confidence_tier="exact",
        )
        result = adapter.match(
            prompt_token_ids=list(range(32)),
            already_matched_len=0,
            prompt_text="q",
        )
        assert result is not None
        # Discovery-only: SGLang side sees no KV reuse...
        assert result.cached_token_count == 0
        assert result.prompt_token_count == 0
        assert result.cached_token_ids == []
        assert result.segments is None
        assert result.layer_recompute_mask is None
        # ...but quality signals are preserved for telemetry.
        assert result.quality_signals.cosine_similarity == pytest.approx(0.92)
        assert result.quality_signals.reuse_ratio == pytest.approx(0.85)
        # And the discovery hit is counted separately.
        stats = adapter.stats()
        assert stats["match_hits_discovery_only"] == 1
        assert stats["match_hits"] == 0

    def test_on_donor_inserted_sets_donor_last_node_id_in_match_result(
        self, adapter, pipeline
    ):
        """Adapter records donor_last_node_id and surfaces it at match time.

        Validates the plumbing for the SGLang RadixCache pool-leak fix:
        on_donor_inserted writes the TreeNode id into _DonorKVHandle, and
        the next match() call returns it via FuzzyMatchResult.donor_last_node_id.
        Without this id, RadixCache.match_prefix can't inc_lock_ref the donor
        and donor KV slots are LRU-evicted mid-request.
        """
        adapter.register_donor(
            request_id="donor-A",
            token_ids=list(range(16)),
            kv_cache=list(range(100, 116)),
            cache_start_pos=0,
            cache_end_pos=16,
            prompt_text="hi",
        )
        # Simulate RadixCache.cache_finished_req calling the new hook.
        adapter.on_donor_inserted(
            request_id="donor-A",
            donor_last_node_id=42,
        )

        pipeline.next_result = _StubPipelineResult(
            found=True,
            donor_id="donor-A",
            similarity=0.92,
            reuse_ratio=0.85,
            donor_tokens=list(range(16)),
            position_map=_StubPosMap(
                donor_positions=list(range(16)),
                target_positions=list(range(16)),
            ),
            confidence_tier="exact",
        )
        result = adapter.match(
            prompt_token_ids=list(range(32)),
            already_matched_len=0,
            prompt_text="q",
        )
        assert result is not None
        assert result.donor_last_node_id == 42

    def test_on_donor_inserted_silent_when_donor_unknown(self, adapter):
        """No error if RadixCache calls on_donor_inserted for a request we
        rejected at register_donor time (e.g. too short, no embedding)."""
        # Donor was never registered; calling the hook should be a no-op.
        adapter.on_donor_inserted(request_id="never-registered", donor_last_node_id=42)
        # No exception, no state mutation. Counter sanity check.
        assert adapter.donor_count() == 0

    def test_match_returns_none_donor_id_when_handle_missing(self, adapter, pipeline):
        """A registered-but-no-handle donor still produces a match with
        donor_last_node_id=None (default). The radix-side fix gates the
        inc_lock_ref on donor_last_node_id is not None."""
        adapter.register_donor(
            request_id="donor-B",
            token_ids=list(range(16)),
            kv_cache=list(range(200, 216)),
            cache_start_pos=0,
            cache_end_pos=16,
            prompt_text="hi",
        )
        # Note: on_donor_inserted is NOT called -- handle.last_node_id stays None.
        pipeline.next_result = _StubPipelineResult(
            found=True,
            donor_id="donor-B",
            similarity=0.95,
            reuse_ratio=0.90,
            donor_tokens=list(range(16)),
            position_map=_StubPosMap(
                donor_positions=list(range(16)),
                target_positions=list(range(16)),
            ),
            confidence_tier="exact",
        )
        result = adapter.match(
            prompt_token_ids=list(range(32)),
            already_matched_len=0,
            prompt_text="q",
        )
        assert result is not None
        assert result.donor_last_node_id is None

    def test_position_offset_respects_already_matched_len(self, adapter, pipeline):
        self._register_donor(adapter, nt=16)
        pipeline.next_result = _StubPipelineResult(
            found=True,
            donor_id="donor-A",
            similarity=0.9,
            reuse_ratio=0.85,
            donor_tokens=list(range(16)),
            position_map=_StubPosMap(
                donor_positions=list(range(16)),
                target_positions=list(range(16)),
            ),
        )
        result = adapter.match(
            prompt_token_ids=list(range(40)),
            already_matched_len=8,
            prompt_text="q",
        )
        assert result is not None
        assert result.position_offset == 8
        # Remaining tokens passed to pipeline are after already_matched_len.
        assert pipeline.find_donor_calls[0]["token_ids"] == list(range(8, 40))


# ---------------------------------------------------------------------
# Public package exports
# ---------------------------------------------------------------------


def test_public_exports():
    from semblend.integration.sglang import (
        FuzzyMatchResult as FMR,
    )
    from semblend.integration.sglang import (
        FuzzyMatchSegment as FMS,
    )
    from semblend.integration.sglang import (
        QualitySignals as QS,
    )
    from semblend.integration.sglang import (
        SemBlendProviderAdapter as Adapter,
    )
    from semblend.integration.sglang import (
        SemBlendProviderConfig as Cfg,
    )

    # Sanity: symbols are the same as the direct import targets.
    from semblend.integration.sglang import provider, types

    assert Adapter is provider.SemBlendProviderAdapter
    assert FMR is types.FuzzyMatchResult
    assert FMS is types.FuzzyMatchSegment
    assert QS is types.QualitySignals
    assert Cfg is not None

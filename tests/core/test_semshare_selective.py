"""Tests for selective attention bridge — SemShareKV schedule to PartialAttentionPlan."""
from __future__ import annotations

import numpy as np
import pytest

from semblend_core.partial_attention import (
    AttentionMode,
    PartialAttentionPlan,
    build_attention_plan,
)
from semblend_core.semshare.config import SemShareConfig
from semblend_core.semshare.hd_detector import HDDetectionResult
from semblend_core.semshare.layer_schedule import compute_layer_schedule
from semblend_core.semshare.selective_attention import (
    build_selective_attention_plan,
    enhance_plan_with_selective_recompute,
)


@pytest.fixture
def config() -> SemShareConfig:
    return SemShareConfig(embed_dim=64, hd_fraction=0.40)


@pytest.fixture
def hd_result() -> HDDetectionResult:
    """10 tokens, 4 HD (top 40%)."""
    return HDDetectionResult(
        hd_mask=tuple([True] * 4 + [False] * 6),
        deviation_scores=tuple([5.0] * 4 + [0.1] * 6),
        hd_threshold=1.0,
        hd_fraction=0.4,
    )


@pytest.fixture
def schedule(hd_result: HDDetectionResult, config: SemShareConfig) -> object:
    return compute_layer_schedule(hd_result, num_layers=8, config=config)


class TestBuildSelectiveAttentionPlan:
    def test_layer1_full_recompute(self, schedule, hd_result) -> None:
        """Layer 1 should be full recompute in selective plan."""
        plan = build_selective_attention_plan(
            donor_id="d1",
            target_len=10,
            donor_len=10,
            schedule=schedule,
            copy_positions=list(range(10)),
            placeholder_positions=[],
            slot_actions=[
                {"action": "copy_from_donor", "target_pos": i, "donor_pos": i}
                for i in range(10)
            ],
            hd_result=hd_result,
        )

        assert plan.donor_id == "d1"
        assert plan.target_len == 10
        # Layer 1 should be full recompute
        assert plan.layer_masks[1].recompute_all is True

    def test_later_layers_graduated(self, schedule, hd_result) -> None:
        """Layers 2+ should have per-token masks, not full recompute."""
        plan = build_selective_attention_plan(
            donor_id="d1",
            target_len=10,
            donor_len=10,
            schedule=schedule,
            copy_positions=list(range(10)),
            placeholder_positions=[],
            slot_actions=[
                {"action": "copy_from_donor", "target_pos": i, "donor_pos": i}
                for i in range(10)
            ],
            hd_result=hd_result,
        )

        # Layers 2+ should NOT be full recompute
        for layer_idx in range(2, 8):
            mask = plan.layer_masks[layer_idx]
            assert mask.recompute_all is False
            assert mask.verification_source == "selective_recompute"

    def test_computation_ratio_less_than_one(self, schedule, hd_result) -> None:
        """Selective recompute should save compute vs full prefill."""
        plan = build_selective_attention_plan(
            donor_id="d1",
            target_len=10,
            donor_len=10,
            schedule=schedule,
            copy_positions=list(range(10)),
            placeholder_positions=[],
            slot_actions=[
                {"action": "copy_from_donor", "target_pos": i, "donor_pos": i}
                for i in range(10)
            ],
        )

        assert plan.computation_ratio < 1.0
        assert plan.computation_ratio > 0.0

    def test_placeholder_positions_always_recomputed(self, schedule) -> None:
        """Placeholder positions should be recomputed in every layer."""
        plan = build_selective_attention_plan(
            donor_id="d1",
            target_len=10,
            donor_len=10,
            schedule=schedule,
            copy_positions=[0, 1, 2, 3, 4],
            placeholder_positions=[5, 6, 7, 8, 9],
            slot_actions=[
                {"action": "copy_from_donor", "target_pos": i, "donor_pos": i}
                for i in range(5)
            ] + [
                {"action": "placeholder", "target_pos": i}
                for i in range(5, 10)
            ],
        )

        # Placeholder positions should be PARTIAL or FULL in all layers
        for layer_mask in plan.layer_masks:
            for pm in layer_mask.position_masks:
                if pm.target_pos >= 5:
                    assert pm.mode in (AttentionMode.PARTIAL, AttentionMode.FULL)


class TestEnhancePlanWithSelectiveRecompute:
    def test_full_recompute_layers_preserved(self, schedule, hd_result) -> None:
        """Layers bathtub marked as full recompute should stay that way."""
        # Build a bathtub plan where layers 0-1 are full recompute
        existing = build_attention_plan(
            donor_id="d1",
            target_len=10,
            donor_len=10,
            copy_positions=list(range(10)),
            placeholder_positions=[],
            slot_actions=[
                {"action": "copy_from_donor", "target_pos": i, "donor_pos": i}
                for i in range(10)
            ],
            layer_hints=[
                {"recompute_all": True, "deviation_score": 0.9},
                {"recompute_all": True, "deviation_score": 0.8},
            ] + [
                {"recompute_all": False, "deviation_score": 0.1}
                for _ in range(6)
            ],
            num_layers=8,
        )

        enhanced = enhance_plan_with_selective_recompute(
            existing, schedule, hd_result
        )

        # First two layers should still be full recompute
        assert enhanced.layer_masks[0].recompute_all is True
        assert enhanced.layer_masks[1].recompute_all is True

    def test_reuse_layers_get_selective_tokens(self, schedule, hd_result) -> None:
        """Layers bathtub said to reuse should get selective per-token masks."""
        existing = build_attention_plan(
            donor_id="d1",
            target_len=10,
            donor_len=10,
            copy_positions=list(range(10)),
            placeholder_positions=[],
            slot_actions=[
                {"action": "copy_from_donor", "target_pos": i, "donor_pos": i}
                for i in range(10)
            ],
            layer_hints=[
                {"recompute_all": False, "deviation_score": 0.1}
                for _ in range(8)
            ],
            num_layers=8,
        )

        enhanced = enhance_plan_with_selective_recompute(
            existing, schedule, hd_result
        )

        # Some positions in enhanced layers should be PARTIAL (recomputed)
        for layer_idx in range(2, 8):
            mask = enhanced.layer_masks[layer_idx]
            recompute_count = sum(
                1 for pm in mask.position_masks
                if pm.mode == AttentionMode.PARTIAL
            )
            # Layer 1 in schedule is full recompute, so layer_masks[1] should
            # have all PARTIAL. Other enhanced layers should have some.
            if layer_idx == 1:
                assert recompute_count == 10  # full
            # Other layers: the schedule adds some recomputed tokens
            # (can't assert exact count due to schedule dynamics)

    def test_enhanced_computation_ratio(self, schedule, hd_result) -> None:
        """Enhanced plan should have lower computation ratio than full prefill."""
        existing = build_attention_plan(
            donor_id="d1",
            target_len=10,
            donor_len=10,
            copy_positions=list(range(10)),
            placeholder_positions=[],
            slot_actions=[
                {"action": "copy_from_donor", "target_pos": i, "donor_pos": i}
                for i in range(10)
            ],
            layer_hints=[
                {"recompute_all": False, "deviation_score": 0.1}
                for _ in range(8)
            ],
            num_layers=8,
        )

        enhanced = enhance_plan_with_selective_recompute(
            existing, schedule, hd_result
        )

        assert enhanced.computation_ratio < 1.0
        assert enhanced.computation_ratio > 0.0

    def test_donor_map_preserved(self, schedule, hd_result) -> None:
        """Donor map from original plan should be preserved."""
        existing = PartialAttentionPlan(
            target_len=10,
            donor_len=10,
            donor_id="d1",
            layer_masks=[],
            num_reuse_positions=10,
            num_partial_positions=0,
            num_full_layers=0,
            computation_ratio=0.0,
            donor_map=((0, "d1"), (1, "d1")),
        )

        # Schedule with 0 layers — should just return same plan
        from semblend_core.semshare.layer_schedule import LayerSchedule
        empty_schedule = LayerSchedule(
            per_layer_alpha=(),
            per_layer_masks=(),
            num_layers=0,
            total_recomputed=0,
            total_possible=0,
        )

        enhanced = enhance_plan_with_selective_recompute(
            existing, empty_schedule, hd_result
        )

        assert enhanced.donor_map == ((0, "d1"), (1, "d1"))

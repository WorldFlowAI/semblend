"""Unit tests for the PartialAttention scaffold."""

from __future__ import annotations

import numpy as np

from synapse_kv_connector.partial_attention import (
    AttentionMode,
    LayerMask,
    PartialAttentionPlan,
    PositionMask,
    build_attention_plan,
    compute_attention_mask,
    compute_donor_kv_indices,
)


class TestBuildAttentionPlan:
    """Tests for build_attention_plan."""

    def test_basic_plan_no_layer_hints(self) -> None:
        """Build plan with copy and placeholder positions."""
        plan = build_attention_plan(
            donor_id="d1",
            target_len=5,
            donor_len=5,
            copy_positions=[0, 1, 3, 4],
            placeholder_positions=[2],
            slot_actions=[
                {"action": "copy_from_donor", "donorPos": 0, "targetPos": 0},
                {"action": "copy_from_donor", "donorPos": 1, "targetPos": 1},
                {"action": "placeholder", "targetPos": 2},
                {"action": "copy_from_donor", "donorPos": 3, "targetPos": 3},
                {"action": "copy_from_donor", "donorPos": 4, "targetPos": 4},
            ],
            num_layers=4,
        )

        assert plan.target_len == 5
        assert plan.donor_len == 5
        assert plan.donor_id == "d1"
        assert plan.num_reuse_positions == 4
        assert plan.num_partial_positions == 1
        assert plan.num_full_layers == 0
        assert len(plan.layer_masks) == 4

        # Computation ratio: 1/5 of positions per layer = 0.2
        assert abs(plan.computation_ratio - 0.2) < 1e-6

    def test_plan_with_layer_hints(self) -> None:
        """Build plan with CacheBlend layer recomputation hints."""
        plan = build_attention_plan(
            donor_id="d2",
            target_len=4,
            donor_len=4,
            copy_positions=[0, 1, 2],
            placeholder_positions=[3],
            slot_actions=[
                {"action": "copy_from_donor", "donorPos": 0, "targetPos": 0},
                {"action": "copy_from_donor", "donorPos": 1, "targetPos": 1},
                {"action": "copy_from_donor", "donorPos": 2, "targetPos": 2},
                {"action": "placeholder", "targetPos": 3},
            ],
            layer_hints=[
                {"recomputeAll": False, "deviationScore": 0.1},
                {"recomputeAll": True, "deviationScore": 0.5},
                {"recomputeAll": False, "deviationScore": 0.2},
                {"recomputeAll": True, "deviationScore": 0.8},
            ],
            num_layers=4,
        )

        assert plan.num_full_layers == 2
        assert not plan.layer_masks[0].recompute_all
        assert plan.layer_masks[1].recompute_all
        assert not plan.layer_masks[2].recompute_all
        assert plan.layer_masks[3].recompute_all

        # Computation ratio:
        # 2 partial layers × 1 placeholder = 2
        # 2 full layers × 4 positions = 8
        # Total possible = 4 layers × 4 positions = 16
        # Ratio = (2 + 8) / 16 = 0.625
        assert abs(plan.computation_ratio - 0.625) < 1e-6

    def test_full_recompute_plan(self) -> None:
        """All positions are placeholders."""
        plan = build_attention_plan(
            donor_id="d3",
            target_len=3,
            donor_len=3,
            copy_positions=[],
            placeholder_positions=[0, 1, 2],
            slot_actions=[
                {"action": "placeholder", "targetPos": 0},
                {"action": "placeholder", "targetPos": 1},
                {"action": "placeholder", "targetPos": 2},
            ],
            num_layers=2,
        )

        assert plan.num_reuse_positions == 0
        assert plan.num_partial_positions == 3
        assert abs(plan.computation_ratio - 1.0) < 1e-6

    def test_full_reuse_plan(self) -> None:
        """All positions copy from donor."""
        plan = build_attention_plan(
            donor_id="d4",
            target_len=3,
            donor_len=3,
            copy_positions=[0, 1, 2],
            placeholder_positions=[],
            slot_actions=[
                {"action": "copy_from_donor", "donorPos": 0, "targetPos": 0},
                {"action": "copy_from_donor", "donorPos": 1, "targetPos": 1},
                {"action": "copy_from_donor", "donorPos": 2, "targetPos": 2},
            ],
            num_layers=2,
        )

        assert plan.num_reuse_positions == 3
        assert plan.num_partial_positions == 0
        assert abs(plan.computation_ratio - 0.0) < 1e-6


class TestComputeAttentionMask:
    """Tests for compute_attention_mask."""

    def test_partial_layer_mask(self) -> None:
        """Mask shows only placeholder positions as True."""
        plan = build_attention_plan(
            donor_id="d",
            target_len=5,
            donor_len=5,
            copy_positions=[0, 1, 3, 4],
            placeholder_positions=[2],
            slot_actions=[
                {"action": "copy_from_donor", "donorPos": 0, "targetPos": 0},
                {"action": "copy_from_donor", "donorPos": 1, "targetPos": 1},
                {"action": "placeholder", "targetPos": 2},
                {"action": "copy_from_donor", "donorPos": 3, "targetPos": 3},
                {"action": "copy_from_donor", "donorPos": 4, "targetPos": 4},
            ],
            num_layers=2,
        )

        mask = compute_attention_mask(plan, layer_idx=0)
        assert mask.shape == (5,)
        assert mask.dtype == bool
        np.testing.assert_array_equal(
            mask, [False, False, True, False, False]
        )

    def test_full_recompute_layer_mask(self) -> None:
        """Full recompute layer has all True."""
        plan = build_attention_plan(
            donor_id="d",
            target_len=3,
            donor_len=3,
            copy_positions=[0, 1],
            placeholder_positions=[2],
            slot_actions=[
                {"action": "copy_from_donor", "donorPos": 0, "targetPos": 0},
                {"action": "copy_from_donor", "donorPos": 1, "targetPos": 1},
                {"action": "placeholder", "targetPos": 2},
            ],
            layer_hints=[
                {"recomputeAll": True, "deviationScore": 0.9},
            ],
            num_layers=1,
        )

        mask = compute_attention_mask(plan, layer_idx=0)
        np.testing.assert_array_equal(mask, [True, True, True])

    def test_out_of_range_layer_defaults_to_full(self) -> None:
        """Out-of-range layer index defaults to full computation."""
        plan = build_attention_plan(
            donor_id="d",
            target_len=3,
            donor_len=3,
            copy_positions=[0],
            placeholder_positions=[1, 2],
            slot_actions=[
                {"action": "copy_from_donor", "donorPos": 0, "targetPos": 0},
                {"action": "placeholder", "targetPos": 1},
                {"action": "placeholder", "targetPos": 2},
            ],
            num_layers=1,
        )

        mask = compute_attention_mask(plan, layer_idx=99)
        np.testing.assert_array_equal(mask, [True, True, True])


class TestComputeDonorKvIndices:
    """Tests for compute_donor_kv_indices."""

    def test_returns_copy_pairs(self) -> None:
        """Returns (donor_pos, target_pos) for REUSE positions."""
        plan = build_attention_plan(
            donor_id="d",
            target_len=4,
            donor_len=4,
            copy_positions=[0, 2, 3],
            placeholder_positions=[1],
            slot_actions=[
                {"action": "copy_from_donor", "donorPos": 0, "targetPos": 0},
                {"action": "placeholder", "targetPos": 1},
                {"action": "copy_from_donor", "donorPos": 2, "targetPos": 2},
                {"action": "copy_from_donor", "donorPos": 3, "targetPos": 3},
            ],
            num_layers=2,
        )

        pairs = compute_donor_kv_indices(plan, layer_idx=0)
        assert pairs == [(0, 0), (2, 2), (3, 3)]

    def test_full_recompute_layer_returns_empty(self) -> None:
        """Full recompute layer returns no copy pairs."""
        plan = build_attention_plan(
            donor_id="d",
            target_len=3,
            donor_len=3,
            copy_positions=[0, 1],
            placeholder_positions=[2],
            slot_actions=[
                {"action": "copy_from_donor", "donorPos": 0, "targetPos": 0},
                {"action": "copy_from_donor", "donorPos": 1, "targetPos": 1},
                {"action": "placeholder", "targetPos": 2},
            ],
            layer_hints=[
                {"recomputeAll": True, "deviationScore": 0.9},
            ],
            num_layers=1,
        )

        pairs = compute_donor_kv_indices(plan, layer_idx=0)
        assert pairs == []

    def test_out_of_range_layer_returns_empty(self) -> None:
        """Out-of-range layer returns empty list."""
        plan = build_attention_plan(
            donor_id="d",
            target_len=2,
            donor_len=2,
            copy_positions=[0],
            placeholder_positions=[1],
            slot_actions=[
                {"action": "copy_from_donor", "donorPos": 0, "targetPos": 0},
                {"action": "placeholder", "targetPos": 1},
            ],
            num_layers=1,
        )

        assert compute_donor_kv_indices(plan, layer_idx=5) == []


class TestPositionMask:
    """Tests for PositionMask dataclass."""

    def test_reuse_mask(self) -> None:
        mask = PositionMask(
            target_pos=0, mode=AttentionMode.REUSE, donor_pos=3
        )
        assert mask.target_pos == 0
        assert mask.mode == AttentionMode.REUSE
        assert mask.donor_pos == 3

    def test_partial_mask(self) -> None:
        mask = PositionMask(target_pos=1, mode=AttentionMode.PARTIAL)
        assert mask.donor_pos is None

    def test_frozen(self) -> None:
        mask = PositionMask(target_pos=0, mode=AttentionMode.FULL)
        try:
            mask.target_pos = 1  # type: ignore[misc]
            assert False, "Should have raised"
        except AttributeError:
            pass


class TestAttentionMode:
    """Tests for AttentionMode enum."""

    def test_values(self) -> None:
        assert AttentionMode.FULL.value == "full"
        assert AttentionMode.REUSE.value == "reuse"
        assert AttentionMode.PARTIAL.value == "partial"
        assert AttentionMode.SKIP_LAYER.value == "skip_layer"

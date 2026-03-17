"""Tests for the vLLM attention kernel patch module."""

from __future__ import annotations

import numpy as np
import pytest

from synapse_kv_connector.attention_patch import (
    PrefillPatchResult,
    apply_kv_patch,
    get_compute_mask,
    summarize_patch,
)
from synapse_kv_connector.partial_attention import (
    AttentionMode,
    LayerMask,
    PartialAttentionPlan,
    PositionMask,
    build_attention_plan,
)


def _make_plan(
    target_len: int = 10,
    donor_len: int = 8,
    copy_positions: list[int] | None = None,
    placeholder_positions: list[int] | None = None,
    num_layers: int = 4,
) -> PartialAttentionPlan:
    """Helper to build a PartialAttentionPlan from positions."""
    if copy_positions is None:
        copy_positions = [0, 1, 2, 3, 4]
    if placeholder_positions is None:
        placeholder_positions = [5, 6, 7, 8, 9]

    slot_actions = [
        {
            "action": "copy_from_donor",
            "donor_pos": p,
            "target_pos": p,
        }
        for p in copy_positions
    ] + [
        {"action": "placeholder", "target_pos": p}
        for p in placeholder_positions
    ]

    return build_attention_plan(
        donor_id="test-donor",
        target_len=target_len,
        donor_len=donor_len,
        copy_positions=copy_positions,
        placeholder_positions=placeholder_positions,
        slot_actions=slot_actions,
        num_layers=num_layers,
    )


class TestApplyKvPatch:
    """Tests for apply_kv_patch."""

    def test_copies_donor_kv_at_reuse_positions(self):
        plan = _make_plan(
            target_len=6,
            donor_len=6,
            copy_positions=[0, 1, 2],
            placeholder_positions=[3, 4, 5],
            num_layers=2,
        )

        num_heads = 4
        head_dim = 8

        # Donor KV has known values at positions 0,1,2
        donor_kv = np.ones((2, num_heads, 6, head_dim), dtype=np.float16)
        donor_kv[:, :, 0, :] = 1.0
        donor_kv[:, :, 1, :] = 2.0
        donor_kv[:, :, 2, :] = 3.0

        # Target KV starts as zeros
        kv_cache = np.zeros((2, num_heads, 6, head_dim), dtype=np.float16)

        result = apply_kv_patch(plan, 0, kv_cache, donor_kv)

        # Positions 0,1,2 should have donor values
        np.testing.assert_array_equal(
            result[0, 0, 0, :], np.full(head_dim, 1.0, dtype=np.float16)
        )
        np.testing.assert_array_equal(
            result[0, 0, 1, :], np.full(head_dim, 2.0, dtype=np.float16)
        )
        np.testing.assert_array_equal(
            result[0, 0, 2, :], np.full(head_dim, 3.0, dtype=np.float16)
        )

        # Positions 3,4,5 should remain zeros (placeholders)
        np.testing.assert_array_equal(
            result[0, 0, 3, :], np.zeros(head_dim, dtype=np.float16)
        )

    def test_no_copy_for_full_recompute_layer(self):
        plan = _make_plan(num_layers=2)

        # Layer with recompute_all = True should return empty pairs
        full_plan = PartialAttentionPlan(
            target_len=10,
            donor_len=8,
            donor_id="test",
            layer_masks=[
                LayerMask(
                    layer_idx=0,
                    recompute_all=True,
                    deviation_score=0.9,
                )
            ],
            num_reuse_positions=0,
            num_partial_positions=10,
            num_full_layers=1,
            computation_ratio=1.0,
        )

        kv_cache = np.zeros((2, 4, 10, 8), dtype=np.float16)
        donor_kv = np.ones((2, 4, 8, 8), dtype=np.float16)

        result = apply_kv_patch(full_plan, 0, kv_cache, donor_kv)

        # Nothing should be copied
        np.testing.assert_array_equal(result, kv_cache)

    def test_handles_out_of_range_layer(self):
        plan = _make_plan(num_layers=2)
        kv_cache = np.zeros((2, 4, 10, 8), dtype=np.float16)
        donor_kv = np.ones((2, 4, 8, 8), dtype=np.float16)

        # Layer 5 is out of range (only 2 layers in plan)
        result = apply_kv_patch(plan, 5, kv_cache, donor_kv)
        np.testing.assert_array_equal(result, kv_cache)


class TestGetComputeMask:
    """Tests for get_compute_mask."""

    def test_partial_layer_mask(self):
        plan = _make_plan(
            target_len=6,
            copy_positions=[0, 1, 2],
            placeholder_positions=[3, 4, 5],
            num_layers=2,
        )

        mask = get_compute_mask(plan, 0)

        # Placeholder positions should be True (need computation)
        assert mask[3]
        assert mask[4]
        assert mask[5]

        # Reuse positions should be False
        assert not mask[0]
        assert not mask[1]
        assert not mask[2]

    def test_full_recompute_layer(self):
        plan = PartialAttentionPlan(
            target_len=6,
            donor_len=6,
            donor_id="test",
            layer_masks=[
                LayerMask(
                    layer_idx=0,
                    recompute_all=True,
                    deviation_score=0.9,
                )
            ],
            num_reuse_positions=0,
            num_partial_positions=6,
            num_full_layers=1,
            computation_ratio=1.0,
        )

        mask = get_compute_mask(plan, 0)
        assert mask.all()  # All positions need computation


class TestSummarizePatch:
    """Tests for summarize_patch."""

    def test_basic_summary(self):
        plan = _make_plan(
            target_len=10,
            copy_positions=[0, 1, 2, 3, 4, 5, 6],
            placeholder_positions=[7, 8, 9],
            num_layers=4,
        )

        result = summarize_patch(plan)

        assert isinstance(result, PrefillPatchResult)
        assert result.positions_reused == 7
        assert result.positions_computed == 3
        assert result.computation_ratio < 1.0
        assert result.layers_fully_recomputed == 0

    def test_full_reuse_ratio(self):
        plan = _make_plan(
            target_len=8,
            copy_positions=list(range(8)),
            placeholder_positions=[],
            num_layers=4,
        )

        result = summarize_patch(plan)
        assert result.computation_ratio == 0.0
        assert result.positions_reused == 8
        assert result.positions_computed == 0

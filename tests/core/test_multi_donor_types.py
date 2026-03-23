"""Tests for multi-donor frozen data structures and attention plan builder."""
import numpy as np
import pytest

from semblend_core.multi_donor_types import (
    ChunkAssignment,
    CompositeKVPlan,
    MatchType,
    MultiDonorAlignmentResult,
    MultiDonorPositionMapping,
    MultiDonorSlotAction,
)


class TestMatchType:
    """MatchType enum values."""

    def test_enum_values(self):
        assert MatchType.EXACT.value == "exact"
        assert MatchType.FUZZY.value == "fuzzy"
        assert MatchType.RECOMPUTE.value == "recompute"


class TestChunkAssignment:
    """ChunkAssignment frozen dataclass."""

    def test_frozen(self):
        asgn = ChunkAssignment(
            target_chunk_idx=0,
            donor_id="d1",
            donor_chunk_idx=5,
            match_type=MatchType.EXACT,
            confidence=1.0,
        )
        with pytest.raises(AttributeError):
            asgn.donor_id = "d2"

    def test_defaults(self):
        asgn = ChunkAssignment(target_chunk_idx=3)
        assert asgn.donor_id is None
        assert asgn.donor_chunk_idx is None
        assert asgn.match_type == MatchType.RECOMPUTE
        assert asgn.confidence == 0.0


class TestMultiDonorSlotAction:
    """MultiDonorSlotAction frozen dataclass."""

    def test_frozen(self):
        sa = MultiDonorSlotAction(
            action="copy_from_donor",
            target_pos=10,
            donor_pos=5,
            donor_id="d1",
        )
        with pytest.raises(AttributeError):
            sa.action = "recompute"

    def test_recompute_has_no_donor(self):
        sa = MultiDonorSlotAction(
            action="recompute",
            target_pos=42,
        )
        assert sa.donor_pos is None
        assert sa.donor_id is None


class TestMultiDonorPositionMapping:
    """MultiDonorPositionMapping frozen dataclass."""

    def test_frozen(self):
        mapping = MultiDonorPositionMapping(
            donor_ids=("d1", "d2"),
            donor_positions=(0, 256),
            target_positions=(0, 256),
        )
        with pytest.raises(AttributeError):
            mapping.donor_ids = ()

    def test_num_pairs(self):
        mapping = MultiDonorPositionMapping(
            donor_ids=("d1", "d1", "d2"),
            donor_positions=(0, 1, 100),
            target_positions=(0, 1, 200),
        )
        assert mapping.num_pairs == 3

    def test_needs_correction(self):
        # No correction needed when donor_pos == target_pos
        no_corr = MultiDonorPositionMapping(
            donor_ids=("d1",),
            donor_positions=(5,),
            target_positions=(5,),
        )
        assert not no_corr.needs_correction

        # Correction needed when positions differ
        corr = MultiDonorPositionMapping(
            donor_ids=("d1",),
            donor_positions=(5,),
            target_positions=(10,),
        )
        assert corr.needs_correction

    def test_for_donor_filter(self):
        mapping = MultiDonorPositionMapping(
            donor_ids=("d1", "d2", "d1", "d2"),
            donor_positions=(0, 100, 1, 101),
            target_positions=(0, 200, 1, 201),
        )
        d1_only = mapping.for_donor("d1")
        assert d1_only.num_pairs == 2
        assert d1_only.donor_ids == ("d1", "d1")
        assert d1_only.donor_positions == (0, 1)
        assert d1_only.target_positions == (0, 1)

    def test_empty_mapping(self):
        mapping = MultiDonorPositionMapping()
        assert mapping.num_pairs == 0
        assert not mapping.needs_correction


class TestCompositeKVPlan:
    """CompositeKVPlan frozen dataclass."""

    def test_frozen(self):
        plan = CompositeKVPlan(
            donor_ids=("d1", "d2"),
            total_reuse_ratio=0.75,
            donors_per_composite=2,
        )
        with pytest.raises(AttributeError):
            plan.total_reuse_ratio = 0.5

    def test_actions_for_donor(self):
        plan = CompositeKVPlan(
            donor_ids=("d1", "d2"),
            slot_actions=(
                MultiDonorSlotAction("copy_from_donor", 0, 0, "d1"),
                MultiDonorSlotAction("copy_from_donor", 1, 100, "d2"),
                MultiDonorSlotAction("recompute", 2),
                MultiDonorSlotAction("copy_from_donor", 3, 1, "d1"),
            ),
            total_reuse_ratio=0.75,
            donors_per_composite=2,
        )
        d1_actions = plan.actions_for_donor("d1")
        assert len(d1_actions) == 2
        assert all(a.donor_id == "d1" for a in d1_actions)

        d2_actions = plan.actions_for_donor("d2")
        assert len(d2_actions) == 1

    def test_recompute_positions(self):
        plan = CompositeKVPlan(
            slot_actions=(
                MultiDonorSlotAction("copy_from_donor", 0, 0, "d1"),
                MultiDonorSlotAction("recompute", 1),
                MultiDonorSlotAction("recompute", 2),
                MultiDonorSlotAction("copy_from_donor", 3, 0, "d2"),
            ),
            total_reuse_ratio=0.5,
            donors_per_composite=2,
        )
        recompute = plan.recompute_positions()
        assert recompute == [1, 2]


class TestMultiDonorAlignmentResult:
    """MultiDonorAlignmentResult frozen dataclass."""

    def test_frozen(self):
        result = MultiDonorAlignmentResult(
            reuse_ratio=0.8,
            chunk_assignments=(),
            composite_plan=CompositeKVPlan(),
            donor_ids=("d1",),
            exact_chunks=5,
        )
        with pytest.raises(AttributeError):
            result.reuse_ratio = 0.9

    def test_defaults(self):
        result = MultiDonorAlignmentResult(
            reuse_ratio=0.0,
            chunk_assignments=(),
            composite_plan=CompositeKVPlan(),
            donor_ids=(),
        )
        assert result.exact_chunks == 0
        assert result.fuzzy_chunks == 0
        assert result.recompute_chunks == 0
        assert result.chunk_index_hits == 0


class TestBuildMultiDonorAttentionPlan:
    """Tests for build_multi_donor_attention_plan."""

    def test_basic_multi_donor_plan(self):
        from semblend_core.partial_attention import build_multi_donor_attention_plan

        slot_actions = [
            {"action": "copy_from_donor", "targetPos": 0, "donorPos": 0, "donorId": "d1"},
            {"action": "copy_from_donor", "targetPos": 1, "donorPos": 1, "donorId": "d1"},
            {"action": "copy_from_donor", "targetPos": 2, "donorPos": 10, "donorId": "d2"},
            {"action": "recompute", "targetPos": 3},
        ]
        plan = build_multi_donor_attention_plan(
            target_len=4,
            slot_actions=slot_actions,
            num_layers=4,
        )
        assert plan.target_len == 4
        assert plan.num_reuse_positions == 3
        assert plan.num_partial_positions == 1
        assert plan.computation_ratio < 1.0

    def test_donor_map_is_immutable_tuple(self):
        from semblend_core.partial_attention import build_multi_donor_attention_plan

        slot_actions = [
            {"action": "copy_from_donor", "targetPos": 0, "donorPos": 0, "donorId": "d1"},
            {"action": "copy_from_donor", "targetPos": 1, "donorPos": 5, "donorId": "d2"},
        ]
        plan = build_multi_donor_attention_plan(
            target_len=2,
            slot_actions=slot_actions,
            num_layers=2,
        )
        # donor_map should be a tuple of (pos, donor_id) pairs
        assert isinstance(plan.donor_map, tuple)
        donor_dict = dict(plan.donor_map)
        assert donor_dict[0] == "d1"
        assert donor_dict[1] == "d2"

    def test_primary_donor_most_frequent(self):
        from semblend_core.partial_attention import build_multi_donor_attention_plan

        # 3 positions from d1, 1 from d2 → primary = d1
        slot_actions = [
            {"action": "copy_from_donor", "targetPos": 0, "donorPos": 0, "donorId": "d1"},
            {"action": "copy_from_donor", "targetPos": 1, "donorPos": 1, "donorId": "d1"},
            {"action": "copy_from_donor", "targetPos": 2, "donorPos": 2, "donorId": "d1"},
            {"action": "copy_from_donor", "targetPos": 3, "donorPos": 10, "donorId": "d2"},
        ]
        plan = build_multi_donor_attention_plan(
            target_len=4, slot_actions=slot_actions, num_layers=2,
        )
        assert plan.donor_id == "d1"

    def test_layer_hints_full_recompute(self):
        from semblend_core.partial_attention import build_multi_donor_attention_plan

        slot_actions = [
            {"action": "copy_from_donor", "targetPos": 0, "donorPos": 0, "donorId": "d1"},
            {"action": "recompute", "targetPos": 1},
        ]
        layer_hints = [
            {"recompute_all": True, "deviation_score": 0.9},
            {"recompute_all": False, "deviation_score": 0.1},
        ]
        plan = build_multi_donor_attention_plan(
            target_len=2,
            slot_actions=slot_actions,
            layer_hints=layer_hints,
            num_layers=2,
        )
        assert plan.num_full_layers == 1
        assert plan.layer_masks[0].recompute_all is True
        assert plan.layer_masks[1].recompute_all is False


class TestExtendSegments:
    """Tests for PQSegmentStore.extend_segments."""

    def test_extend_pre_training(self):
        from semblend_core.pq_segment_store import PQSegmentStore

        store = PQSegmentStore(max_entries=100, train_threshold=1000)
        rng = np.random.RandomState(42)

        # Add initial segments
        initial = rng.randn(5, 384).astype(np.float32)
        initial /= np.linalg.norm(initial, axis=1, keepdims=True)
        store.add_segments("d1", initial)
        assert store.size == 1

        # Extend with more segments
        extra = rng.randn(3, 384).astype(np.float32)
        extra /= np.linalg.norm(extra, axis=1, keepdims=True)
        store.extend_segments("d1", extra)

        # Buffer should have 8 segments for d1
        assert "d1" in store._buffer
        assert store._buffer["d1"].shape[0] == 8

    def test_extend_new_donor(self):
        from semblend_core.pq_segment_store import PQSegmentStore

        store = PQSegmentStore(max_entries=100, train_threshold=1000)
        rng = np.random.RandomState(42)

        segs = rng.randn(3, 384).astype(np.float32)
        segs /= np.linalg.norm(segs, axis=1, keepdims=True)
        store.extend_segments("new_donor", segs)

        assert "new_donor" in store._buffer
        assert store._buffer["new_donor"].shape[0] == 3

    def test_extend_respects_max_segments(self):
        from semblend_core.pq_segment_store import PQSegmentStore

        store = PQSegmentStore(
            max_entries=100, max_segments_per_entry=10, train_threshold=1000,
        )
        rng = np.random.RandomState(42)

        initial = rng.randn(8, 384).astype(np.float32)
        initial /= np.linalg.norm(initial, axis=1, keepdims=True)
        store.add_segments("d1", initial)

        # Try to extend by 5 (would exceed max of 10)
        extra = rng.randn(5, 384).astype(np.float32)
        extra /= np.linalg.norm(extra, axis=1, keepdims=True)
        store.extend_segments("d1", extra)

        # Should cap at 10
        assert store._buffer["d1"].shape[0] == 10

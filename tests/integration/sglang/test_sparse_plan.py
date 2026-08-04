"""Sparse-plan construction from gated alignment segments (H18 unit 1).

The plan orders the remaining-prompt window into alternating novel/donor
spans so the consumption engine can steer chunk boundaries to donor-span
edges: novel spans are computed (attending all earlier KV), donor spans are
consumed as boundary-anchored contiguous joins. Every position in
[0, remaining_len) must be covered exactly once.
"""

from semblend.integration.sglang.sparse_plan import (
    SparsePlanSpan,
    build_sparse_plan,
)
from semblend.integration.sglang.types import FuzzyMatchSegment


def _seg(t_start, length, d_start):
    return FuzzyMatchSegment(
        target_positions=list(range(t_start, t_start + length)),
        donor_positions=list(range(d_start, d_start + length)),
        length=length,
    )


def _covers(plan, remaining_len):
    pos = 0
    for span in plan:
        assert span.target_start == pos, f"gap/overlap at {pos}: {span}"
        assert span.target_end > span.target_start
        pos = span.target_end
    return pos == remaining_len


class TestBuildSparsePlan:
    def test_interior_donor_span_yields_novel_donor_novel(self):
        # Production shape: novel head, huge donor interior, novel tail.
        plan = build_sparse_plan([_seg(247, 21000, 500)], remaining_len=21500)
        assert [s.kind for s in plan] == ["novel", "donor", "novel"]
        assert _covers(plan, 21500)
        donor = plan[1]
        assert (donor.target_start, donor.target_end) == (247, 21247)
        assert donor.donor_start == 500
        assert donor.segment_index == 0

    def test_boundary_anchored_donor_has_no_leading_novel(self):
        plan = build_sparse_plan([_seg(0, 100, 40)], remaining_len=150)
        assert [s.kind for s in plan] == ["donor", "novel"]
        assert _covers(plan, 150)

    def test_multiple_donors_sorted_and_gap_filled(self):
        segs = [_seg(300, 50, 900), _seg(100, 80, 700)]
        plan = build_sparse_plan(segs, remaining_len=400)
        assert [s.kind for s in plan] == [
            "novel",
            "donor",
            "novel",
            "donor",
            "novel",
        ]
        assert _covers(plan, 400)
        assert plan[1].target_start == 100 and plan[1].segment_index == 1
        assert plan[3].target_start == 300 and plan[3].segment_index == 0

    def test_short_donor_span_folds_into_novel(self):
        plan = build_sparse_plan(
            [_seg(100, 8, 700)], remaining_len=200, min_donor_span=16
        )
        assert plan is None  # sole donor folded away -> no plan

    def test_overlapping_later_segment_dropped(self):
        segs = [_seg(100, 100, 700), _seg(150, 100, 900)]
        plan = build_sparse_plan(segs, remaining_len=400)
        donors = [s for s in plan if s.kind == "donor"]
        assert len(donors) == 1
        assert (donors[0].target_start, donors[0].target_end) == (100, 200)
        assert _covers(plan, 400)

    def test_no_segments_returns_none(self):
        assert build_sparse_plan([], remaining_len=100) is None

    def test_donor_span_at_exact_tail(self):
        plan = build_sparse_plan([_seg(50, 50, 0)], remaining_len=100)
        assert [s.kind for s in plan] == ["novel", "donor"]
        assert _covers(plan, 100)

    def test_edge_shave_moves_span_edges_into_novel(self):
        plan = build_sparse_plan(
            [_seg(100, 1000, 700)], remaining_len=1200, edge_shave=32
        )
        donor = [s for s in plan if s.kind == "donor"][0]
        assert (donor.target_start, donor.target_end) == (132, 1068)
        assert donor.donor_start == 732  # advanced with the start shave
        assert _covers(plan, 1200)

    def test_edge_shave_below_min_span_folds(self):
        plan = build_sparse_plan(
            [_seg(100, 60, 700)], remaining_len=300, min_donor_span=16, edge_shave=32
        )
        assert plan is None  # 60 - 64 < min -> folded away

    def test_gap_interleave_splits_long_spans(self):
        plan = build_sparse_plan(
            [_seg(0, 5000, 100)],
            remaining_len=5000,
            gap_period=2000,
            gap_size=64,
        )
        kinds = [s.kind for s in plan]
        # donor 2000, gap 64, donor 2000, gap 64, donor 872 (>= min 16)
        assert kinds == ["donor", "novel", "donor", "novel", "donor"]
        d = [s for s in plan if s.kind == "donor"]
        assert (d[0].target_start, d[0].target_end, d[0].donor_start) == (0, 2000, 100)
        assert (d[1].target_start, d[1].donor_start) == (2064, 2164)
        assert _covers(plan, 5000)

    def test_gap_interleave_leaves_short_spans_whole(self):
        plan = build_sparse_plan(
            [_seg(0, 1500, 100)], remaining_len=1600, gap_period=2000, gap_size=64
        )
        assert [s.kind for s in plan] == ["donor", "novel"]

    def test_gap_remainder_below_min_folds_into_novel(self):
        plan = build_sparse_plan(
            [_seg(0, 2070, 100)],
            remaining_len=2070,
            min_donor_span=16,
            gap_period=2000,
            gap_size=64,
        )
        # remainder after donor(2000)+gap(64) is 6 tokens < min -> novel
        assert [s.kind for s in plan] == ["donor", "novel"]
        assert plan[0].target_end == 2000
        assert _covers(plan, 2070)

    def test_span_exceeding_window_is_clamped(self):
        plan = build_sparse_plan([_seg(50, 100, 0)], remaining_len=100)
        donors = [s for s in plan if s.kind == "donor"]
        assert donors[0].target_end == 100
        assert _covers(plan, 100)

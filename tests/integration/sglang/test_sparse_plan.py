"""Sparse-plan construction from gated alignment segments (plan-steered unit 1).

The plan orders the remaining-prompt window into alternating novel/donor
spans so the consumption engine can steer chunk boundaries to donor-span
edges: novel spans are computed (attending all earlier KV), donor spans are
consumed as boundary-anchored contiguous joins. Every position in
[0, remaining_len) must be covered exactly once.
"""

from semblend.integration.sglang.sparse_plan import (
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


class TestCanonicalAugmentation:
    def test_reformat_donor_yields_verified_segments(self, monkeypatch):
        """End to end at the adapter: token alignment starving + canonical runs
        present -> augmented segments carry engine-verified token identity."""
        monkeypatch.setenv("SEMBLEND_CANONICAL_MATCH", "1")
        from types import SimpleNamespace

        from semblend.integration.sglang.provider import SemBlendProviderAdapter

        adapter = SemBlendProviderAdapter.__new__(SemBlendProviderAdapter)
        adapter._config = SimpleNamespace(segment_min_tokens=4, model_arch="x")

        # Fake offsets tokenizer: BPE-like words (whitespace attaches).
        import re as _re

        def tok(text, add_special_tokens=False, return_offsets_mapping=True):
            ids, offs = [], []
            for m in _re.finditer(r"\s*\S+", text):
                ids.append(hash(m.group(0)) & 0x7FFFFFFF)
                offs.append((m.start(), m.end()))
            return {"input_ids": ids, "offset_mapping": offs}

        adapter._offsets_tok = tok

        base = [f"tok{i}" for i in range(40)]
        donor_text = " ".join(base)
        target_text = "\n".join(
            " ".join(base[k : k + 10]) for k in range(0, 40, 10)
        )
        d_ids = tok(donor_text)["input_ids"]
        t_ids = tok(target_text)["input_ids"]

        handle = SimpleNamespace(
            prompt_text=donor_text,
            kv_indices=list(range(1000, 1000 + len(d_ids))),
            last_node_id=7,
        )
        segs = adapter._canonical_augment_segments(
            remaining=t_ids,
            remaining_text=target_text,
            donor_id="d1",
            handle=handle,
            donor_tokens=d_ids,
        )
        assert segs, "expected augmented segments"
        covered = sum(s.length for s in segs)
        assert covered >= 25  # most of 40 words minus rewrap bubbles
        for s in segs:
            # engine-tokenization identity was re-verified inside
            assert t_ids[s.target_positions[0] : s.target_positions[0] + s.length] == \
                d_ids[s.donor_positions[0] : s.donor_positions[0] + s.length]

    def test_engine_window_convention(self, monkeypatch):
        """Live-flow contract: the engine decodes the POST-HEAD window and
        passes that text, so run indices are window-relative and verify
        directly against the window ids — no rebasing. (A rebase here
        shifted every run into silent guard death in live serving.)"""
        monkeypatch.setenv("SEMBLEND_CANONICAL_MATCH", "1")
        from types import SimpleNamespace

        from semblend.integration.sglang.provider import SemBlendProviderAdapter

        adapter = SemBlendProviderAdapter.__new__(SemBlendProviderAdapter)
        adapter._config = SimpleNamespace(segment_min_tokens=4, model_arch="x")

        import re as _re

        def tok(text, add_special_tokens=False, return_offsets_mapping=True):
            ids, offs = [], []
            for m in _re.finditer(r"\s*\S+", text):
                ids.append(hash(m.group(0)) & 0x7FFFFFFF)
                offs.append((m.start(), m.end()))
            return {"input_ids": ids, "offset_mapping": offs}

        adapter._offsets_tok = tok

        base = [f"tok{i}" for i in range(40)]
        donor_text = " ".join(base)
        # engine already served the 12-token head; the adapter receives
        # ONLY the window's text and ids
        window_text = "\n".join(
            " ".join(base[k : k + 10]) for k in range(12, 40, 10)
        )
        d_ids = tok(donor_text)["input_ids"]
        window_ids = tok(window_text)["input_ids"]

        handle = SimpleNamespace(
            prompt_text=donor_text,
            kv_indices=list(range(1000, 1000 + len(d_ids))),
            last_node_id=7,
        )
        segs = adapter._canonical_augment_segments(
            remaining=window_ids,
            remaining_text=window_text,
            donor_id="d1",
            handle=handle,
            donor_tokens=d_ids,
        )
        assert segs, "window-relative runs must survive"
        covered = sum(s.length for s in segs)
        assert covered >= 18
        for s in segs:
            assert window_ids[s.target_positions[0] : s.target_positions[0] + s.length] == \
                d_ids[s.donor_positions[0] : s.donor_positions[0] + s.length]

    def test_rescue_resolves_unregistered_composite_donor(self, monkeypatch):
        """Composite (multi-donor) results can carry a donor_id that is not
        a registry key; the rescue previously skipped canonical alignment
        SILENTLY (observed live as cover=0.000 with zero aligner runs).
        With one registered donor the resolver must fall back to it."""
        monkeypatch.setenv("SEMBLEND_CANONICAL_MATCH", "1")
        from collections import OrderedDict
        from types import SimpleNamespace

        from semblend.integration.sglang.provider import SemBlendProviderAdapter

        adapter = SemBlendProviderAdapter.__new__(SemBlendProviderAdapter)
        sole = SimpleNamespace(token_ids=[1, 2, 3], prompt_text="a b c")
        adapter._donor_kv = OrderedDict([("real-donor", sole)])

        result = SimpleNamespace(donor_id="composite-xyz", donor_ids=None)
        donor_id, handle = adapter._resolve_canon_handle(result)
        assert donor_id == "real-donor"
        assert handle is sole

        # composite donor ids take precedence over the sole fallback
        other = SimpleNamespace(token_ids=[9], prompt_text="z")
        adapter._donor_kv["other"] = other
        result2 = SimpleNamespace(donor_id="composite-xyz", donor_ids=["other"])
        donor_id2, handle2 = adapter._resolve_canon_handle(result2)
        assert (donor_id2, handle2) == ("other", other)

        # unresolvable: multiple donors, none referenced
        result3 = SimpleNamespace(donor_id="composite-xyz", donor_ids=None)
        _did, handle3 = adapter._resolve_canon_handle(result3)
        assert handle3 is None

    def test_disabled_without_env(self):
        from types import SimpleNamespace

        from semblend.integration.sglang.provider import SemBlendProviderAdapter

        adapter = SemBlendProviderAdapter.__new__(SemBlendProviderAdapter)
        adapter._config = SimpleNamespace(segment_min_tokens=4, model_arch="x")
        adapter._offsets_tok = None
        assert adapter._canonical_augment_segments(
            remaining=[1, 2],
            remaining_text="a b",
            donor_id="d",
            handle=SimpleNamespace(prompt_text="a b", kv_indices=[], last_node_id=None),
            donor_tokens=[1, 2],
        ) == []

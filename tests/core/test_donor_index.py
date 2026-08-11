"""Vector-index seam: exact reference behavior and the delta/graph
lifecycle (rebuild folding, tombstones, re-ranking) with a fake graph."""

import numpy as np
import pytest

from semblend_core.donor_index import BruteForceIndex, DeltaGraphIndex


def _vec(*xs):
    return np.asarray(xs, dtype=np.float32)


def _fake_graph_factory(built):
    """Records builds; searches the built matrix exactly (stands in for
    the ANN graph, whose approximation is out of scope here)."""

    def factory(ids, matrix):
        built.append(list(ids))

        def search(query, top_k):
            sims = matrix @ query
            order = np.argsort(-sims)[:top_k]
            return [(ids[i], float(sims[i])) for i in order]

        return search

    return factory


class TestBruteForceIndex:
    def test_top_k_orders_by_cosine(self):
        ix = BruteForceIndex(dim=2)
        ix.add("a", _vec(1, 0))
        ix.add("b", _vec(0.9, 0.1))
        ix.add("c", _vec(0, 1))
        out = ix.search(_vec(1, 0), top_k=2)
        assert [d for d, _ in out] == ["a", "b"]

    def test_remove_and_reinsert(self):
        ix = BruteForceIndex(dim=2)
        ix.add("a", _vec(1, 0))
        ix.add("b", _vec(0, 1))
        ix.remove("a")
        assert ix.size() == 1
        assert [d for d, _ in ix.search(_vec(1, 0), top_k=2)] == ["b"]
        ix.add("a", _vec(1, 0))
        assert ix.search(_vec(1, 0), top_k=1)[0][0] == "a"


class TestDeltaGraphIndex:
    def test_delta_serves_before_any_rebuild(self):
        ix = DeltaGraphIndex(2, _fake_graph_factory([]), rebuild_threshold=100)
        ix.add("a", _vec(1, 0))
        assert ix.search(_vec(1, 0), top_k=1)[0][0] == "a"

    def test_threshold_triggers_rebuild_and_folds_delta(self):
        built = []
        ix = DeltaGraphIndex(2, _fake_graph_factory(built), rebuild_threshold=2)
        ix.add("a", _vec(1, 0))
        ix.add("b", _vec(0, 1))  # hits threshold -> rebuild
        assert built == [["a", "b"]]
        assert ix._delta.size() == 0
        # graph now serves both; delta empty
        assert ix.search(_vec(0, 1), top_k=1)[0][0] == "b"

    def test_tombstone_hides_built_entry_until_fold(self):
        built = []
        ix = DeltaGraphIndex(2, _fake_graph_factory(built), rebuild_threshold=2)
        ix.add("a", _vec(1, 0))
        ix.add("b", _vec(0, 1))
        ix.remove("a")
        assert [d for d, _ in ix.search(_vec(1, 0), top_k=2)] == ["b"]
        assert ix.size() == 1
        ix.add("c", _vec(1, 0.1))
        ix.add("d", _vec(0.5, 0.5))  # rebuild folds tombstone
        assert "a" not in ix._built_vec
        assert ix.size() == 3

    def test_union_reranks_graph_and_delta(self):
        built = []
        ix = DeltaGraphIndex(2, _fake_graph_factory(built), rebuild_threshold=2)
        ix.add("far", _vec(0, 1))
        ix.add("mid", _vec(0.7, 0.7))  # rebuild
        ix.add("near", _vec(1, 0))     # delta only
        out = ix.search(_vec(1, 0), top_k=3)
        assert [d for d, _ in out] == ["near", "mid", "far"]

    def test_readd_after_remove_clears_tombstone(self):
        ix = DeltaGraphIndex(2, _fake_graph_factory([]), rebuild_threshold=2)
        ix.add("a", _vec(1, 0))
        ix.add("b", _vec(0, 1))
        ix.remove("a")
        ix.add("a", _vec(1, 0))
        assert ix.search(_vec(1, 0), top_k=1)[0][0] == "a"


class TestRecallTelemetry:
    def test_sampled_audit_records_perfect_recall_for_exact_fake(self):
        built = []
        ix = DeltaGraphIndex(
            2, _fake_graph_factory(built), rebuild_threshold=2,
            recall_sample_every=1,
        )
        ix.add("a", _vec(1, 0))
        ix.add("b", _vec(0, 1))
        ix.search(_vec(1, 0), top_k=2)
        assert ix.recall_samples == 1
        assert ix.observed_recall == 1.0

    def test_lossy_graph_shows_degraded_recall(self):
        def lossy_factory(ids, matrix):
            def search(query, top_k):
                return []  # graph loses everything

            return search

        ix = DeltaGraphIndex(2, lossy_factory, rebuild_threshold=2,
                             recall_sample_every=1)
        ix.add("a", _vec(1, 0))
        ix.add("b", _vec(0, 1))  # rebuild -> both only in (lossy) graph
        ix.search(_vec(1, 0), top_k=2)
        assert ix.observed_recall == 0.0

    def test_sampling_disabled(self):
        ix = DeltaGraphIndex(2, _fake_graph_factory([]), rebuild_threshold=100,
                             recall_sample_every=0)
        ix.add("a", _vec(1, 0))
        ix.search(_vec(1, 0), top_k=1)
        assert ix.recall_samples == 0

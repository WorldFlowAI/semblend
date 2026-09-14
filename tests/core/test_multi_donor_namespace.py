"""Namespace isolation for multi-donor composite alignment.

A composite plan is assembled from process-global indexes (ChunkIndex, PQ
segment store) that hold every donor the process ever registered. These
tests pin the property that a donor outside the requesting namespace can
never appear in a plan — in the exact-grid phase and in the PQ semantic
phase — while same-namespace reuse is left untouched.
"""

import logging
import time

import numpy as np
import pytest

from semblend_core.chunk_index import ChunkIndex
from semblend_core.donor_store import DonorNode, DonorStore
from semblend_core.multi_donor_alignment import (
    compute_cdc_alignment,
    compute_multi_donor_alignment,
)
from semblend_core.pq_segment_store import PQSegmentStore

CHUNK_SIZE = 16

# Hashed-salt shaped namespaces, as the engine connectors build them.
NS_A = "semblend:ns:salt:1111111111111111"
NS_FOREIGN = "semblend:ns:salt:2222222222222222"


@pytest.fixture(autouse=True)
def _grid_chunking(monkeypatch):
    """Pin the default fixed-grid path; CDC chunking has its own test below."""
    monkeypatch.delenv("SEMBLEND_CDC_CHUNKS", raising=False)


def _tokens(n: int, offset: int = 0) -> list[int]:
    return list(range(offset, offset + n))


def _node(request_id: str, token_ids: list[int], extra_key: str | None) -> DonorNode:
    return DonorNode(
        request_id=request_id,
        token_ids=token_ids,
        embedding=np.ones(4, dtype=np.float32) / 2.0,
        timestamp=time.monotonic(),
        extra_key=extra_key,
    )


def _store() -> DonorStore:
    return DonorStore(max_entries=100, embedding_dim=4, chunk_size=CHUNK_SIZE)


def _copied_donor_ids(result) -> set[str]:
    return {
        sa.donor_id for sa in result.composite_plan.slot_actions if sa.action == "copy_from_donor"
    }


class _FixedEmbedder:
    """Embedder stub — the PQ phase only needs some query segment vectors."""

    dimension = 4

    def embed(self, text: str):
        return np.ones(4, dtype=np.float32)


class _ForeignPQStore(PQSegmentStore):
    """PQ store whose best per-chunk match is always the foreign donor.

    Subclassed rather than faked: the alignment only consults a store that
    passes an isinstance check against the real type.
    """

    def __init__(self, donor_id: str) -> None:
        super().__init__(max_entries=4, max_segments_per_entry=4)
        self._foreign_donor_id = donor_id

    @property
    def size(self) -> int:
        return 1

    def find_best_donor_per_chunk(self, query_segments, min_similarity: float = 0.85):
        return [(self._foreign_donor_id, 0, 0.99)] * query_segments.shape[0]


class TestExactGridPhaseIsolation:
    """Phase 1 reads ChunkIndex locations for every donor in the process."""

    def test_foreign_donor_chunks_never_enter_the_composite(self):
        store = _store()
        shared = _tokens(CHUNK_SIZE * 2, offset=100)
        # Identical content in both namespaces, foreign registered first:
        # it then owns the first ChunkLocation of every shared hash, which
        # is the one unfiltered selection picks.
        store.add_donor(_node("d-foreign", list(shared), NS_FOREIGN))
        store.add_donor(_node("d-a", list(shared), NS_A))

        result = store.find_multi_donor(
            query_tokens=list(shared),
            min_reuse_ratio=0.5,
            extra_key=NS_A,
        )

        assert result is not None
        assert result.donor_ids == ("d-a",)
        assert _copied_donor_ids(result) == {"d-a"}
        assert set(result.composite_plan.position_map.donor_ids) == {"d-a"}
        assert all(
            asgn.donor_id in (None, "d-a") for asgn in result.composite_plan.chunk_assignments
        )
        # Same-namespace reuse is untouched by the filter.
        assert result.reuse_ratio == 1.0

    def test_salted_request_cannot_read_an_unsalted_donor(self):
        store = _store()
        shared = _tokens(CHUNK_SIZE * 2, offset=300)
        store.add_donor(_node("d-unsalted", list(shared), None))
        store.add_donor(_node("d-a", list(shared), NS_A))

        result = store.find_multi_donor(
            query_tokens=list(shared),
            min_reuse_ratio=0.5,
            extra_key=NS_A,
        )

        assert result is not None
        assert result.donor_ids == ("d-a",)
        assert _copied_donor_ids(result) == {"d-a"}

    def test_withheld_chunks_are_counted_and_reported(self):
        store = _store()
        shared = _tokens(CHUNK_SIZE * 2, offset=500)
        store.add_donor(_node("d-foreign", list(shared), NS_FOREIGN))
        store.add_donor(_node("d-a", list(shared), NS_A))

        logger_name = "semblend_core.multi_donor_alignment"
        records: list[logging.LogRecord] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        handler = _Capture(level=logging.WARNING)
        logger = logging.getLogger(logger_name)
        logger.addHandler(handler)
        try:
            result = store.find_multi_donor(
                query_tokens=list(shared),
                min_reuse_ratio=0.5,
                extra_key=NS_A,
            )
        finally:
            logger.removeHandler(handler)

        assert result is not None
        # One withheld donor chunk per target chunk, both reported.
        messages = [record.getMessage() for record in records]
        assert any("withheld 2 cross-namespace donor chunk candidates" in m for m in messages)

    def test_same_namespace_donors_still_compose(self):
        store = _store()
        chunk_a = _tokens(CHUNK_SIZE, offset=100)
        chunk_b = _tokens(CHUNK_SIZE, offset=200)
        store.add_donor(_node("d-a1", chunk_a + _tokens(CHUNK_SIZE, offset=1000), NS_A))
        store.add_donor(_node("d-a2", chunk_b + _tokens(CHUNK_SIZE, offset=2000), NS_A))

        result = store.find_multi_donor(
            query_tokens=chunk_a + chunk_b,
            min_reuse_ratio=0.5,
            extra_key=NS_A,
        )

        assert result is not None
        assert set(result.donor_ids) == {"d-a1", "d-a2"}
        assert result.reuse_ratio == 1.0

    def test_no_isolation_key_sees_every_donor(self):
        store = _store()
        shared = _tokens(CHUNK_SIZE * 2, offset=100)
        store.add_donor(_node("d-foreign", list(shared), NS_FOREIGN))
        store.add_donor(_node("d-a", list(shared), NS_A))

        result = store.find_multi_donor(query_tokens=list(shared), min_reuse_ratio=0.5)

        assert result is not None
        assert result.reuse_ratio == 1.0


class TestPQSemanticPhaseIsolation:
    """Phase 1.5 takes PQ winners, which are searched across all donors."""

    def test_foreign_pq_match_never_enters_the_composite(self):
        store = _store()
        shared = _tokens(CHUNK_SIZE * 2, offset=100)
        store.add_donor(_node("d-a", list(shared), NS_A))
        store.add_donor(_node("d-foreign", _tokens(CHUNK_SIZE * 2, offset=700), NS_FOREIGN))

        # Two chunks the exact phase matches against d-a, then one chunk
        # only the PQ phase can place — and PQ places it on the foreign donor.
        target = list(shared) + _tokens(CHUNK_SIZE, offset=5000)

        result = store.find_multi_donor(
            query_tokens=target,
            min_reuse_ratio=0.5,
            pq_store=_ForeignPQStore("d-foreign"),
            target_text="target prompt text " * 40,
            embedder=_FixedEmbedder(),
            extra_key=NS_A,
        )

        assert result is not None
        assert "d-foreign" not in result.donor_ids
        assert result.donor_ids == ("d-a",)
        assert _copied_donor_ids(result) == {"d-a"}
        assert all(
            asgn.donor_id in (None, "d-a") for asgn in result.composite_plan.chunk_assignments
        )
        # The same-namespace chunks are still reused.
        assert result.reuse_ratio > 0.5

    def test_same_namespace_pq_match_is_kept(self):
        store = _store()
        shared = _tokens(CHUNK_SIZE * 2, offset=100)
        store.add_donor(_node("d-a", list(shared), NS_A))

        target = list(shared) + _tokens(CHUNK_SIZE, offset=5000)

        result = store.find_multi_donor(
            query_tokens=target,
            min_reuse_ratio=0.5,
            pq_store=_ForeignPQStore("d-a"),
            target_text="target prompt text " * 40,
            embedder=_FixedEmbedder(),
            extra_key=NS_A,
        )

        assert result is not None
        assert result.donor_ids == ("d-a",)
        assert result.fuzzy_chunks == 1


class TestAllowedSetIsIndependentOfTheTokenStore:
    """The allowed set is threaded down, not inferred from a filtered store."""

    def test_allowed_set_gates_an_unfiltered_donor_token_store(self):
        index = ChunkIndex(max_donors=10, chunk_size=CHUNK_SIZE)
        shared = _tokens(CHUNK_SIZE * 2, offset=100)
        index.add_donor_chunks("d-foreign", list(shared))
        index.add_donor_chunks("d-a", list(shared))

        result = compute_multi_donor_alignment(
            target_tokens=list(shared),
            chunk_index=index,
            donor_token_store={"d-foreign": list(shared), "d-a": list(shared)},
            chunk_size=CHUNK_SIZE,
            allowed_donor_ids={"d-a"},
        )

        assert result is not None
        assert result.donor_ids == ("d-a",)
        assert _copied_donor_ids(result) == {"d-a"}


class TestCdcPhaseIsolation:
    """CDC chunking reads the same process-global ChunkIndex."""

    def test_foreign_donor_chunks_never_enter_a_cdc_plan(self, monkeypatch):
        monkeypatch.setenv("SEMBLEND_CDC_CHUNKS", "1")
        shared = [2000 + (i * 41) % 900 for i in range(600)]
        index = ChunkIndex(max_donors=10, chunk_size=CHUNK_SIZE)
        index.add_donor_chunks("d-foreign", list(shared))
        index.add_donor_chunks("d-a", list(shared))

        result = compute_cdc_alignment(
            list(shared),
            index,
            {"d-foreign": list(shared), "d-a": list(shared)},
            allowed_donor_ids={"d-a"},
        )

        assert result is not None
        assert result.donor_ids == ("d-a",)
        assert _copied_donor_ids(result) == {"d-a"}

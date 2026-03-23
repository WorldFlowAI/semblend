"""Tests for multi-donor composite alignment."""
import pytest

from semblend_core.chunk_index import ChunkIndex
from semblend_core.multi_donor_alignment import compute_multi_donor_alignment

CHUNK_SIZE = 16  # Small chunk size for fast tests


def _make_tokens(n: int, offset: int = 0) -> list[int]:
    return list(range(offset, offset + n))


@pytest.fixture
def chunk_index():
    return ChunkIndex(max_donors=1000, chunk_size=CHUNK_SIZE)


class TestSingleDonorFallback:
    """Multi-donor with only one donor should behave like single-donor."""

    def test_single_donor_exact_match(self, chunk_index):
        donor = _make_tokens(CHUNK_SIZE * 4)
        chunk_index.add_donor_chunks("d1", donor)

        # Target shares first 3 chunks with donor
        target = donor[:CHUNK_SIZE * 3] + _make_tokens(CHUNK_SIZE, offset=9000)

        result = compute_multi_donor_alignment(
            target_tokens=target,
            chunk_index=chunk_index,
            donor_token_store={"d1": donor},
            chunk_size=CHUNK_SIZE,
        )

        assert result is not None
        assert result.reuse_ratio > 0.5
        assert len(result.donor_ids) == 1
        assert result.donor_ids[0] == "d1"
        assert result.exact_chunks == 3
        assert result.recompute_chunks == 1

    def test_no_match_returns_none(self, chunk_index):
        donor = _make_tokens(CHUNK_SIZE * 2)
        chunk_index.add_donor_chunks("d1", donor)

        target = _make_tokens(CHUNK_SIZE * 2, offset=50000)

        result = compute_multi_donor_alignment(
            target_tokens=target,
            chunk_index=chunk_index,
            donor_token_store={"d1": donor},
            chunk_size=CHUNK_SIZE,
        )
        assert result is None


class TestComplementaryDonors:
    """Two donors each covering different parts of the target."""

    def test_two_donors_complementary(self, chunk_index):
        # Donor 1 has chunks A, B
        chunk_a = _make_tokens(CHUNK_SIZE, offset=100)
        chunk_b = _make_tokens(CHUNK_SIZE, offset=200)
        # Donor 2 has chunks C, D
        chunk_c = _make_tokens(CHUNK_SIZE, offset=300)
        chunk_d = _make_tokens(CHUNK_SIZE, offset=400)

        donor1 = chunk_a + chunk_b
        donor2 = chunk_c + chunk_d

        chunk_index.add_donor_chunks("d1", donor1)
        chunk_index.add_donor_chunks("d2", donor2)

        # Target: A, B, C, D — first half from d1, second from d2
        target = chunk_a + chunk_b + chunk_c + chunk_d

        result = compute_multi_donor_alignment(
            target_tokens=target,
            chunk_index=chunk_index,
            donor_token_store={"d1": donor1, "d2": donor2},
            chunk_size=CHUNK_SIZE,
        )

        assert result is not None
        assert result.reuse_ratio == 1.0
        assert len(result.donor_ids) == 2
        assert set(result.donor_ids) == {"d1", "d2"}
        assert result.exact_chunks == 4
        assert result.composite_plan.donors_per_composite == 2

    def test_three_donors_rag(self, chunk_index):
        """3-document RAG scenario: each doc cached as separate donor."""
        doc1 = _make_tokens(CHUNK_SIZE * 3, offset=1000)
        doc2 = _make_tokens(CHUNK_SIZE * 3, offset=2000)
        doc3 = _make_tokens(CHUNK_SIZE * 3, offset=3000)

        chunk_index.add_donor_chunks("doc1", doc1)
        chunk_index.add_donor_chunks("doc2", doc2)
        chunk_index.add_donor_chunks("doc3", doc3)

        # RAG query with pieces from all 3 docs + new query
        query_chunk = _make_tokens(CHUNK_SIZE, offset=9999)
        target = (
            doc1[:CHUNK_SIZE * 2]  # 2 chunks from doc1
            + doc2[CHUNK_SIZE:CHUNK_SIZE * 3]  # 2 chunks from doc2
            + doc3[:CHUNK_SIZE]  # 1 chunk from doc3
            + query_chunk  # new query
        )

        result = compute_multi_donor_alignment(
            target_tokens=target,
            chunk_index=chunk_index,
            donor_token_store={"doc1": doc1, "doc2": doc2, "doc3": doc3},
            chunk_size=CHUNK_SIZE,
        )

        assert result is not None
        assert len(result.donor_ids) == 3
        assert result.exact_chunks == 5  # 2 + 2 + 1
        assert result.recompute_chunks == 1  # query chunk


class TestContextGate:
    """Context gate rejects isolated cross-donor matches."""

    def test_isolated_match_rejected(self, chunk_index):
        """A single matching chunk surrounded by non-matches should be rejected."""
        shared_chunk = _make_tokens(CHUNK_SIZE, offset=100)
        donor = shared_chunk + _make_tokens(CHUNK_SIZE * 3, offset=5000)
        chunk_index.add_donor_chunks("d1", donor)

        # Target: different chunk, shared chunk, different chunk, different chunk
        target = (
            _make_tokens(CHUNK_SIZE, offset=7000)
            + shared_chunk
            + _make_tokens(CHUNK_SIZE, offset=8000)
            + _make_tokens(CHUNK_SIZE, offset=9000)
        )

        result = compute_multi_donor_alignment(
            target_tokens=target,
            chunk_index=chunk_index,
            donor_token_store={"d1": donor},
            chunk_size=CHUNK_SIZE,
            context_gate=True,
        )

        # Should return None because the single match is isolated
        assert result is None

    def test_context_gate_disabled(self, chunk_index):
        """With context gate off, isolated matches are kept."""
        shared_chunk = _make_tokens(CHUNK_SIZE, offset=100)
        donor = shared_chunk + _make_tokens(CHUNK_SIZE, offset=5000)
        chunk_index.add_donor_chunks("d1", donor)

        target = (
            _make_tokens(CHUNK_SIZE, offset=7000)
            + shared_chunk
            + _make_tokens(CHUNK_SIZE, offset=8000)
        )

        result = compute_multi_donor_alignment(
            target_tokens=target,
            chunk_index=chunk_index,
            donor_token_store={"d1": donor},
            chunk_size=CHUNK_SIZE,
            context_gate=False,
        )

        assert result is not None
        assert result.exact_chunks == 1

    def test_adjacent_matches_pass_gate(self, chunk_index):
        """Two adjacent matching chunks should pass context gate."""
        chunk_a = _make_tokens(CHUNK_SIZE, offset=100)
        chunk_b = _make_tokens(CHUNK_SIZE, offset=200)
        donor = chunk_a + chunk_b + _make_tokens(CHUNK_SIZE, offset=5000)
        chunk_index.add_donor_chunks("d1", donor)

        target = chunk_a + chunk_b + _make_tokens(CHUNK_SIZE, offset=9000)

        result = compute_multi_donor_alignment(
            target_tokens=target,
            chunk_index=chunk_index,
            donor_token_store={"d1": donor},
            chunk_size=CHUNK_SIZE,
            context_gate=True,
        )

        assert result is not None
        assert result.exact_chunks == 2


class TestCompositeSlotActions:
    """Verify per-position slot actions carry correct donor_id."""

    def test_slot_actions_have_donor_ids(self, chunk_index):
        chunk_a = _make_tokens(CHUNK_SIZE, offset=100)
        chunk_b = _make_tokens(CHUNK_SIZE, offset=200)
        d1 = chunk_a + _make_tokens(CHUNK_SIZE, offset=1000)
        d2 = chunk_b + _make_tokens(CHUNK_SIZE, offset=2000)

        chunk_index.add_donor_chunks("d1", d1)
        chunk_index.add_donor_chunks("d2", d2)

        target = chunk_a + chunk_b

        result = compute_multi_donor_alignment(
            target_tokens=target,
            chunk_index=chunk_index,
            donor_token_store={"d1": d1, "d2": d2},
            chunk_size=CHUNK_SIZE,
        )

        assert result is not None
        plan = result.composite_plan

        # First CHUNK_SIZE actions should be from d1
        d1_actions = [sa for sa in plan.slot_actions if sa.donor_id == "d1"]
        d2_actions = [sa for sa in plan.slot_actions if sa.donor_id == "d2"]
        assert len(d1_actions) == CHUNK_SIZE
        assert len(d2_actions) == CHUNK_SIZE

    def test_position_map_per_donor(self, chunk_index):
        chunk_a = _make_tokens(CHUNK_SIZE, offset=100)
        chunk_b = _make_tokens(CHUNK_SIZE, offset=200)
        d1 = chunk_a + _make_tokens(CHUNK_SIZE, offset=1000)
        d2 = chunk_b + _make_tokens(CHUNK_SIZE, offset=2000)

        chunk_index.add_donor_chunks("d1", d1)
        chunk_index.add_donor_chunks("d2", d2)

        target = chunk_a + chunk_b

        result = compute_multi_donor_alignment(
            target_tokens=target,
            chunk_index=chunk_index,
            donor_token_store={"d1": d1, "d2": d2},
            chunk_size=CHUNK_SIZE,
        )

        assert result is not None
        pos_map = result.composite_plan.position_map

        d1_map = pos_map.for_donor("d1")
        d2_map = pos_map.for_donor("d2")
        assert d1_map.num_pairs == CHUNK_SIZE
        assert d2_map.num_pairs == CHUNK_SIZE


class TestMultiTurnScenario:
    """Simulated multi-turn conversation via ChunkIndex."""

    def test_turn_n_plus_1_reuses_prefix(self, chunk_index):
        """Turn 5 shares prefix chunks with Turn 4."""
        prefix = _make_tokens(CHUNK_SIZE * 5, offset=0)
        turn4_suffix = _make_tokens(CHUNK_SIZE * 2, offset=10000)
        turn5_suffix = _make_tokens(CHUNK_SIZE * 2, offset=20000)

        turn4 = prefix + turn4_suffix
        turn5 = prefix + turn5_suffix

        chunk_index.add_donor_chunks("turn4", turn4)

        result = compute_multi_donor_alignment(
            target_tokens=turn5,
            chunk_index=chunk_index,
            donor_token_store={"turn4": turn4},
            chunk_size=CHUNK_SIZE,
        )

        assert result is not None
        assert result.exact_chunks == 5  # All 5 prefix chunks
        assert result.recompute_chunks == 2  # New suffix
        assert result.chunk_index_hits == 5
        assert result.reuse_ratio == pytest.approx(5 / 7, abs=0.01)

"""Tests for SemShareKV LSH index."""
from __future__ import annotations

import numpy as np
import pytest

from semblend_core.semshare.config import SemShareConfig
from semblend_core.semshare.lsh_index import LSHIndex, LSHTable, _first_n_primes


@pytest.fixture
def config() -> SemShareConfig:
    return SemShareConfig(embed_dim=64, lsh_seed=42, max_donors=100, ttl_seconds=3600)


@pytest.fixture
def lsh_index(config: SemShareConfig) -> LSHIndex:
    return LSHIndex(config)


class TestLSHTable:
    def test_hash_deterministic(self) -> None:
        rng = np.random.RandomState(42)
        table = LSHTable(
            projections=rng.randn(8, 64).astype(np.float32),
            offsets=rng.uniform(0, 4.0, size=8).astype(np.float32),
            w=4.0,
        )
        vec = rng.randn(5, 64).astype(np.float32)
        codes1 = table.hash_vectors(vec)
        codes2 = table.hash_vectors(vec)
        np.testing.assert_array_equal(codes1, codes2)

    def test_identical_vectors_same_hash(self) -> None:
        rng = np.random.RandomState(42)
        table = LSHTable(
            projections=rng.randn(8, 64).astype(np.float32),
            offsets=rng.uniform(0, 4.0, size=8).astype(np.float32),
            w=4.0,
        )
        vec = rng.randn(1, 64).astype(np.float32)
        batch = np.vstack([vec, vec, vec])
        codes = table.hash_vectors(batch)
        assert codes[0] == codes[1] == codes[2]

    def test_output_shape(self) -> None:
        rng = np.random.RandomState(42)
        table = LSHTable(
            projections=rng.randn(16, 64).astype(np.float32),
            offsets=rng.uniform(0, 4.0, size=16).astype(np.float32),
            w=4.0,
        )
        vec = rng.randn(100, 64).astype(np.float32)
        codes = table.hash_vectors(vec)
        assert codes.shape == (100,)
        assert codes.dtype == np.int64


class TestLSHIndex:
    def test_register_and_query_identical(self, lsh_index: LSHIndex) -> None:
        """Identical embeddings should match with high similarity."""
        rng = np.random.RandomState(123)
        emb = rng.randn(10, 64).astype(np.float32)

        lsh_index.register_donor("donor1", emb)
        results = lsh_index.query(emb)

        assert len(results) == 10
        # Each target token should find its matching donor token
        matched_count = sum(1 for r in results if len(r) > 0)
        assert matched_count >= 8  # high match rate for identical vectors

    def test_orthogonal_no_match(self, lsh_index: LSHIndex) -> None:
        """Orthogonal vectors should rarely match."""
        # Create orthogonal basis vectors
        eye = np.eye(64, dtype=np.float32)[:10]
        donor = eye * 10.0  # scale up to separate buckets
        target = -eye * 10.0  # opposite direction

        lsh_index.register_donor("donor1", donor)
        results = lsh_index.query(target)

        # Most tokens should have no match above threshold
        high_sim_count = sum(
            1
            for r in results
            if any(m.similarity >= 0.5 for m in r)
        )
        assert high_sim_count <= 3  # very few high-similarity matches

    def test_similar_vectors_probabilistic_match(self) -> None:
        """Similar vectors should match with decent probability."""
        # Use wider buckets and lower similarity threshold for noisy matches
        cfg = SemShareConfig(
            embed_dim=64, lsh_seed=42, max_donors=100, ttl_seconds=3600,
            lsh_w=8.0, min_similarity=0.125,  # 1/8 tables need to agree
        )
        index = LSHIndex(cfg)
        rng = np.random.RandomState(456)
        donor = rng.randn(20, 64).astype(np.float32)
        # Target = donor + small noise
        noise = rng.randn(20, 64).astype(np.float32) * 0.3
        target = donor + noise

        index.register_donor("donor1", donor)
        results = index.query(target)

        matched = sum(1 for r in results if len(r) > 0)
        assert matched >= 5  # reasonable match rate for similar vectors

    def test_multiple_donors(self, lsh_index: LSHIndex) -> None:
        """Can register multiple donors and find best match."""
        rng = np.random.RandomState(789)
        donor1 = rng.randn(10, 64).astype(np.float32)
        donor2 = rng.randn(10, 64).astype(np.float32) * 5.0  # very different

        lsh_index.register_donor("d1", donor1)
        lsh_index.register_donor("d2", donor2)

        # Query with donor1's embeddings — should match d1
        results = lsh_index.query(donor1)
        d1_matches = sum(
            1
            for r in results
            for m in r
            if m.donor_id == "d1"
        )
        d2_matches = sum(
            1
            for r in results
            for m in r
            if m.donor_id == "d2"
        )
        assert d1_matches > d2_matches

    def test_remove_donor(self, lsh_index: LSHIndex) -> None:
        rng = np.random.RandomState(101)
        emb = rng.randn(5, 64).astype(np.float32)

        lsh_index.register_donor("d1", emb)
        assert lsh_index.num_donors == 1

        lsh_index.remove_donor("d1")
        assert lsh_index.num_donors == 0

        # Query should return empty
        results = lsh_index.query(emb)
        total_matches = sum(len(r) for r in results)
        assert total_matches == 0

    def test_capacity_eviction(self, config: SemShareConfig) -> None:
        small_config = SemShareConfig(
            embed_dim=64, max_donors=3, ttl_seconds=3600, lsh_seed=42
        )
        index = LSHIndex(small_config)
        rng = np.random.RandomState(202)

        for i in range(5):
            index.register_donor(f"d{i}", rng.randn(3, 64).astype(np.float32))

        assert index.num_donors <= 3

    def test_candidate_filter(self, lsh_index: LSHIndex) -> None:
        rng = np.random.RandomState(303)
        emb = rng.randn(5, 64).astype(np.float32)

        lsh_index.register_donor("d1", emb)
        lsh_index.register_donor("d2", emb + 0.01)

        results = lsh_index.query(emb, candidate_donor_ids=["d1"])
        for r in results:
            for m in r:
                assert m.donor_id == "d1"

    def test_duplicate_register_ignored(self, lsh_index: LSHIndex) -> None:
        rng = np.random.RandomState(404)
        emb = rng.randn(5, 64).astype(np.float32)

        lsh_index.register_donor("d1", emb)
        lsh_index.register_donor("d1", emb * 2)  # should be ignored
        assert lsh_index.num_donors == 1

    def test_empty_query(self, lsh_index: LSHIndex) -> None:
        rng = np.random.RandomState(505)
        results = lsh_index.query(rng.randn(5, 64).astype(np.float32))
        assert len(results) == 5
        assert all(len(r) == 0 for r in results)


class TestFirstNPrimes:
    def test_first_5(self) -> None:
        assert _first_n_primes(5) == [2, 3, 5, 7, 11]

    def test_first_1(self) -> None:
        assert _first_n_primes(1) == [2]

    def test_first_0(self) -> None:
        assert _first_n_primes(0) == []

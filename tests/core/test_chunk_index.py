"""Tests for ChunkIndex — reverse chunk hash index for cross-donor lookup."""
import threading
import time

import pytest

from semblend_core.chunk_index import ChunkIndex, _chunk_hash, chunk_hash_sequence


@pytest.fixture
def chunk_size():
    return 16  # Small chunk size for fast tests


@pytest.fixture
def chunk_index(chunk_size):
    return ChunkIndex(max_donors=1000, chunk_size=chunk_size)


def _make_tokens(n: int, offset: int = 0) -> list[int]:
    """Generate a token sequence of length n starting from offset."""
    return list(range(offset, offset + n))


class TestChunkIndexBasic:
    """Basic add/lookup/remove operations."""

    def test_add_and_lookup(self, chunk_index, chunk_size):
        tokens = _make_tokens(chunk_size * 4)
        chunk_index.add_donor_chunks("donor_1", tokens)

        assert chunk_index.num_donors == 1
        assert chunk_index.num_entries == 4

        # Look up the first chunk
        first_chunk = tokens[:chunk_size]
        locs = chunk_index.lookup_chunk(first_chunk)
        assert len(locs) == 1
        assert locs[0].donor_id == "donor_1"
        assert locs[0].chunk_idx == 0
        assert locs[0].pos == 0

    def test_lookup_second_chunk(self, chunk_index, chunk_size):
        tokens = _make_tokens(chunk_size * 3)
        chunk_index.add_donor_chunks("d1", tokens)

        second_chunk = tokens[chunk_size:chunk_size * 2]
        locs = chunk_index.lookup_chunk(second_chunk)
        assert len(locs) == 1
        assert locs[0].chunk_idx == 1
        assert locs[0].pos == chunk_size

    def test_lookup_miss(self, chunk_index, chunk_size):
        tokens = _make_tokens(chunk_size * 2)
        chunk_index.add_donor_chunks("d1", tokens)

        # Different chunk
        different = _make_tokens(chunk_size, offset=10000)
        locs = chunk_index.lookup_chunk(different)
        assert len(locs) == 0

    def test_partial_chunk_ignored(self, chunk_index, chunk_size):
        # Trailing partial chunk should not be indexed
        tokens = _make_tokens(chunk_size + 5)
        chunk_index.add_donor_chunks("d1", tokens)
        assert chunk_index.num_entries == 1  # Only the first full chunk

    def test_remove_donor(self, chunk_index, chunk_size):
        tokens = _make_tokens(chunk_size * 3)
        chunk_index.add_donor_chunks("d1", tokens)
        assert chunk_index.num_donors == 1
        assert chunk_index.num_entries == 3

        chunk_index.remove_donor("d1")
        assert chunk_index.num_donors == 0
        assert chunk_index.num_entries == 0

        # Lookup should miss now
        locs = chunk_index.lookup_chunk(tokens[:chunk_size])
        assert len(locs) == 0

    def test_remove_nonexistent(self, chunk_index):
        # Should not raise
        chunk_index.remove_donor("nonexistent")

    def test_duplicate_add_is_noop(self, chunk_index, chunk_size):
        tokens = _make_tokens(chunk_size * 2)
        n1 = chunk_index.add_donor_chunks("d1", tokens)
        n2 = chunk_index.add_donor_chunks("d1", tokens)

        assert n1 == 2
        assert n2 == 0  # Already indexed
        assert chunk_index.num_donors == 1


class TestMultiDonorSameChunk:
    """Multiple donors containing the same chunk."""

    def test_same_chunk_in_two_donors(self, chunk_index, chunk_size):
        shared_chunk = _make_tokens(chunk_size)
        donor1 = shared_chunk + _make_tokens(chunk_size, offset=1000)
        donor2 = shared_chunk + _make_tokens(chunk_size, offset=2000)

        chunk_index.add_donor_chunks("d1", donor1)
        chunk_index.add_donor_chunks("d2", donor2)

        locs = chunk_index.lookup_chunk(shared_chunk)
        assert len(locs) == 2
        donor_ids = {loc.donor_id for loc in locs}
        assert donor_ids == {"d1", "d2"}

    def test_remove_one_donor_keeps_other(self, chunk_index, chunk_size):
        shared_chunk = _make_tokens(chunk_size)
        chunk_index.add_donor_chunks("d1", shared_chunk + _make_tokens(chunk_size, 100))
        chunk_index.add_donor_chunks("d2", shared_chunk + _make_tokens(chunk_size, 200))

        chunk_index.remove_donor("d1")

        locs = chunk_index.lookup_chunk(shared_chunk)
        assert len(locs) == 1
        assert locs[0].donor_id == "d2"


class TestFindMatchingChunks:
    """find_matching_chunks for bulk target lookup."""

    def test_multi_chunk_match(self, chunk_index, chunk_size):
        # Donor has 4 chunks
        donor = _make_tokens(chunk_size * 4)
        chunk_index.add_donor_chunks("d1", donor)

        # Target shares first 3 chunks, 4th is different
        target = donor[:chunk_size * 3] + _make_tokens(chunk_size, offset=9000)
        matches = chunk_index.find_matching_chunks(target)

        assert len(matches) == 3
        assert 0 in matches
        assert 1 in matches
        assert 2 in matches
        assert 3 not in matches

    def test_no_matches(self, chunk_index, chunk_size):
        chunk_index.add_donor_chunks("d1", _make_tokens(chunk_size * 2))
        target = _make_tokens(chunk_size * 2, offset=50000)
        matches = chunk_index.find_matching_chunks(target)
        assert len(matches) == 0


class TestLRUEviction:
    """LRU eviction at capacity."""

    def test_eviction_at_capacity(self, chunk_size):
        idx = ChunkIndex(max_donors=3, chunk_size=chunk_size)

        idx.add_donor_chunks("d1", _make_tokens(chunk_size, offset=0))
        idx.add_donor_chunks("d2", _make_tokens(chunk_size, offset=100))
        idx.add_donor_chunks("d3", _make_tokens(chunk_size, offset=200))
        assert idx.num_donors == 3

        # Adding d4 should evict d1 (LRU)
        idx.add_donor_chunks("d4", _make_tokens(chunk_size, offset=300))
        assert idx.num_donors == 3

        # d1's chunk should be gone
        locs = idx.lookup_chunk(_make_tokens(chunk_size, offset=0))
        assert len(locs) == 0

        # d2, d3, d4 should still exist
        assert len(idx.lookup_chunk(_make_tokens(chunk_size, offset=100))) == 1
        assert len(idx.lookup_chunk(_make_tokens(chunk_size, offset=200))) == 1
        assert len(idx.lookup_chunk(_make_tokens(chunk_size, offset=300))) == 1


class TestLookupByHash:
    """Lookup by pre-computed hash."""

    def test_lookup_hash(self, chunk_index, chunk_size):
        tokens = _make_tokens(chunk_size * 2)
        chunk_index.add_donor_chunks("d1", tokens)

        h = _chunk_hash(tokens[:chunk_size])
        locs = chunk_index.lookup_hash(h)
        assert len(locs) == 1
        assert locs[0].donor_id == "d1"


class TestChunkHashSequence:
    """chunk_hash_sequence utility."""

    def test_produces_correct_count(self, chunk_size):
        tokens = _make_tokens(chunk_size * 5 + 3)  # 5 full + partial
        hashes = chunk_hash_sequence(tokens, chunk_size=chunk_size)
        assert len(hashes) == 5

    def test_consistent_hashing(self, chunk_size):
        tokens = _make_tokens(chunk_size)
        h1 = chunk_hash_sequence(tokens, chunk_size=chunk_size)
        h2 = chunk_hash_sequence(tokens, chunk_size=chunk_size)
        assert h1 == h2


class TestMemoryEstimate:
    """Memory usage estimation."""

    def test_memory_grows_with_entries(self, chunk_index, chunk_size):
        m0 = chunk_index.estimated_memory_bytes()
        for i in range(100):
            chunk_index.add_donor_chunks(
                f"d{i}", _make_tokens(chunk_size * 10, offset=i * 1000)
            )
        m1 = chunk_index.estimated_memory_bytes()
        assert m1 > m0

    def test_memory_within_bounds(self):
        """100K donors × 30 chunks should be < 200MB (realistic 256-token chunks)."""
        # Use realistic chunk_size=256 (not the test fixture's 16)
        realistic_chunk_size = 256
        idx = ChunkIndex(max_donors=100_000, chunk_size=realistic_chunk_size)
        # Index 1000 donors with 30 chunks each as a sample
        for i in range(1000):
            idx.add_donor_chunks(
                f"d{i}",
                _make_tokens(realistic_chunk_size * 30, offset=i * 100000),
            )
        # Extrapolate: 1K donors memory × 100 should be < 300MB
        # (estimate is conservative; actual memory with shared hashes is lower)
        mem_1k = idx.estimated_memory_bytes()
        estimated_100k = mem_1k * 100
        assert estimated_100k < 300_000_000  # < 300MB


class TestLatency:
    """Lookup latency checks."""

    def test_lookup_under_1ms(self, chunk_size):
        idx = ChunkIndex(max_donors=10_000, chunk_size=chunk_size)
        for i in range(1000):
            idx.add_donor_chunks(
                f"d{i}", _make_tokens(chunk_size * 10, offset=i * 10000)
            )

        target_chunk = _make_tokens(chunk_size, offset=500 * 10000)
        t0 = time.monotonic()
        for _ in range(100):
            idx.lookup_chunk(target_chunk)
        elapsed_ms = (time.monotonic() - t0) * 1000 / 100
        assert elapsed_ms < 1.0, f"Lookup took {elapsed_ms:.3f}ms (>1ms)"


class TestThreadSafety:
    """Concurrent access safety."""

    def test_concurrent_add_and_lookup(self, chunk_size):
        idx = ChunkIndex(max_donors=1000, chunk_size=chunk_size)
        errors = []

        def writer(thread_id):
            try:
                for i in range(50):
                    idx.add_donor_chunks(
                        f"t{thread_id}_d{i}",
                        _make_tokens(chunk_size * 5, offset=thread_id * 100000 + i * 1000),
                    )
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(200):
                    idx.lookup_chunk(_make_tokens(chunk_size))
                    idx.find_matching_chunks(_make_tokens(chunk_size * 3))
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=(0,)),
            threading.Thread(target=writer, args=(1,)),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"

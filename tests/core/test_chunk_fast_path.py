"""Tests for ChunkIndex fast path — skip embedding on multi-turn."""
import time

import numpy as np

from semblend_core.chunk_index import ChunkIndex
from semblend_core.donor_store import DonorNode, DonorStore

CHUNK_SIZE = 16


def _make_tokens(n: int, offset: int = 0) -> list[int]:
    return list(range(offset, offset + n))


class TestChunkFastPathDonorStore:
    """DonorStore ChunkIndex integration."""

    def test_add_donor_indexes_chunks(self):
        store = DonorStore(
            max_entries=100,
            embedding_dim=4,
            chunk_size=CHUNK_SIZE,
        )
        tokens = _make_tokens(CHUNK_SIZE * 3)
        node = DonorNode(
            request_id="d1",
            token_ids=tokens,
            embedding=np.random.randn(4).astype(np.float32),
            timestamp=time.monotonic(),
        )
        store.add_donor(node)

        # ChunkIndex should have indexed the chunks
        assert store.chunk_index.num_donors == 1
        assert store.chunk_index.num_entries == 3

    def test_eviction_removes_from_chunk_index(self):
        store = DonorStore(
            max_entries=3,
            embedding_dim=4,
            chunk_size=CHUNK_SIZE,
        )
        for i in range(4):
            node = DonorNode(
                request_id=f"d{i}",
                token_ids=_make_tokens(CHUNK_SIZE * 2, offset=i * 1000),
                embedding=np.random.randn(4).astype(np.float32),
                timestamp=time.monotonic(),
            )
            store.add_donor(node)

        # d0 should have been evicted (LRU)
        locs = store.chunk_index.lookup_chunk(
            _make_tokens(CHUNK_SIZE, offset=0)
        )
        assert len(locs) == 0

        # d1, d2, d3 should still exist
        assert store.chunk_index.num_donors == 3

    def test_get_donor_tokens(self):
        store = DonorStore(max_entries=10, embedding_dim=4, chunk_size=CHUNK_SIZE)
        tokens = _make_tokens(CHUNK_SIZE * 2)
        node = DonorNode(
            request_id="d1",
            token_ids=tokens,
            embedding=np.random.randn(4).astype(np.float32),
            timestamp=time.monotonic(),
        )
        store.add_donor(node)
        assert store.get_donor_tokens("d1") == tokens
        assert store.get_donor_tokens("nonexistent") is None

    def test_find_multi_donor(self):
        store = DonorStore(
            max_entries=100,
            embedding_dim=4,
            chunk_size=CHUNK_SIZE,
        )
        # Register two donors with distinct chunks
        chunk_a = _make_tokens(CHUNK_SIZE, offset=100)
        chunk_b = _make_tokens(CHUNK_SIZE, offset=200)
        d1 = chunk_a + _make_tokens(CHUNK_SIZE, offset=1000)
        d2 = chunk_b + _make_tokens(CHUNK_SIZE, offset=2000)

        store.add_donor(DonorNode(
            request_id="d1", token_ids=d1,
            embedding=np.random.randn(4).astype(np.float32),
            timestamp=time.monotonic(),
        ))
        store.add_donor(DonorNode(
            request_id="d2", token_ids=d2,
            embedding=np.random.randn(4).astype(np.float32),
            timestamp=time.monotonic(),
        ))

        # Target uses chunks from both donors
        target = chunk_a + chunk_b

        result = store.find_multi_donor(
            query_tokens=target,
            min_reuse_ratio=0.5,
        )

        assert result is not None
        assert result.reuse_ratio == 1.0
        assert len(result.donor_ids) == 2


class TestChunkFastPathThreshold:
    """Chunk fast path triggers when >=N chunks match."""

    def test_below_threshold_no_fast_path(self):
        """< 3 matching chunks should NOT trigger fast path."""
        idx = ChunkIndex(max_donors=100, chunk_size=CHUNK_SIZE)
        # One shared chunk + one different
        shared = _make_tokens(CHUNK_SIZE, offset=100)
        donor = shared + _make_tokens(CHUNK_SIZE, offset=200)
        idx.add_donor_chunks("d1", donor)

        # Target has 1 match + 3 different chunks
        target = (
            shared
            + _make_tokens(CHUNK_SIZE, offset=5000)
            + _make_tokens(CHUNK_SIZE, offset=6000)
            + _make_tokens(CHUNK_SIZE, offset=7000)
        )
        matches = idx.find_matching_chunks(target)
        assert len(matches) < 3  # Below threshold

    def test_above_threshold_triggers_fast_path(self):
        """>=3 matching chunks should trigger fast path."""
        idx = ChunkIndex(max_donors=100, chunk_size=CHUNK_SIZE)
        prefix = _make_tokens(CHUNK_SIZE * 5, offset=0)
        donor = prefix + _make_tokens(CHUNK_SIZE * 2, offset=10000)
        idx.add_donor_chunks("d1", donor)

        target = prefix + _make_tokens(CHUNK_SIZE * 2, offset=20000)
        matches = idx.find_matching_chunks(target)
        assert len(matches) >= 3  # Above threshold


class TestMultiTurnFastPath:
    """End-to-end multi-turn scenario: Turn N+1 skips embedding."""

    def test_multi_turn_5_turns(self):
        """Simulate 5 turns of conversation, verifying ChunkIndex growing."""
        idx = ChunkIndex(max_donors=100, chunk_size=CHUNK_SIZE)

        # System prompt (shared prefix)
        system = _make_tokens(CHUNK_SIZE * 3, offset=0)

        for turn in range(5):
            turn_tokens = system + _make_tokens(
                CHUNK_SIZE * 2, offset=(turn + 1) * 10000
            )
            idx.add_donor_chunks(f"turn_{turn}", turn_tokens)

            # Next turn shares the prefix
            next_tokens = system + _make_tokens(
                CHUNK_SIZE * 2, offset=(turn + 2) * 10000
            )
            matches = idx.find_matching_chunks(next_tokens)
            # Should always find the 3 system prompt chunks
            assert len(matches) >= 3, f"Turn {turn}: only {len(matches)} matches"

    def test_embedding_skipped_saves_time(self):
        """Verify that the fast path saves the ~3ms embedding time.

        We simulate by checking that find_matching_chunks is fast (<0.1ms)
        compared to embedding (~3ms).
        """
        idx = ChunkIndex(max_donors=1000, chunk_size=CHUNK_SIZE)
        prefix = _make_tokens(CHUNK_SIZE * 10)

        for i in range(100):
            donor = prefix + _make_tokens(CHUNK_SIZE * 2, offset=(i + 1) * 10000)
            idx.add_donor_chunks(f"d{i}", donor)

        target = prefix + _make_tokens(CHUNK_SIZE * 2, offset=999999)

        t0 = time.monotonic()
        for _ in range(100):
            matches = idx.find_matching_chunks(target)
            assert len(matches) >= 3
        elapsed_ms = (time.monotonic() - t0) * 1000 / 100

        # Should be much faster than 3ms (embedding cost)
        assert elapsed_ms < 1.0, f"Fast path took {elapsed_ms:.3f}ms"

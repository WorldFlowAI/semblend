"""Reverse chunk hash index for O(1) cross-donor chunk lookup.

Maps chunk_hash → list[ChunkLocation] for instant lookup of which
donors contain a given chunk. This enables:
  1. Multi-turn fast path: skip MiniLM embedding when ChunkIndex finds
     >=3 matching chunks (shared prefix from prior turn).
  2. Multi-donor composite: find the best donor for EACH chunk
     independently, assembling KV from multiple donors.

Memory: 100K donors × 30 chunks = 3M entries ≈ 150MB
Thread-safe with RW lock for concurrent register/find.
"""
from __future__ import annotations

import hashlib
import logging
import struct
import threading
from collections import OrderedDict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkLocation:
    """Where a chunk lives in the donor store."""
    donor_id: str
    chunk_idx: int
    pos: int  # Token position in the donor sequence (chunk_idx * chunk_size)


class ChunkIndex:
    """Reverse index: chunk_hash → list[ChunkLocation].

    O(1) lookup per target chunk across ALL donors. Uses interned
    donor_id strings and compact ChunkLocation for memory efficiency.

    Thread-safe: writes use a lock, reads are protected by the same
    lock (RW pattern via threading.Lock — good enough for Python GIL).

    Args:
        max_donors: Maximum number of donors to index.
        chunk_size: Token chunk size for hashing (must match engine).
    """

    def __init__(
        self,
        max_donors: int = 100_000,
        chunk_size: int = 256,
    ) -> None:
        self._max_donors = max_donors
        self._chunk_size = chunk_size
        self._lock = threading.Lock()

        # chunk_hash → list[ChunkLocation]
        self._index: dict[str, list[ChunkLocation]] = {}
        # donor_id → set of chunk_hashes (for fast eviction)
        self._donor_hashes: OrderedDict[str, set[str]] = OrderedDict()
        # Interned donor_id strings (avoid duplicate string objects)
        self._interned: dict[str, str] = {}

    @property
    def num_donors(self) -> int:
        with self._lock:
            return len(self._donor_hashes)

    @property
    def num_entries(self) -> int:
        with self._lock:
            return sum(len(locs) for locs in self._index.values())

    @property
    def num_unique_hashes(self) -> int:
        with self._lock:
            return len(self._index)

    def estimated_memory_bytes(self) -> int:
        """Estimate memory usage of the index."""
        with self._lock:
            n_entries = sum(len(locs) for locs in self._index.values())
            # ~50 bytes per ChunkLocation (dataclass overhead + string ref + 2 ints)
            # ~40 bytes per hash key (32 hex chars + dict overhead)
            # ~100 bytes per donor_hashes entry
            return n_entries * 50 + len(self._index) * 40 + len(self._donor_hashes) * 100

    def add_donor_chunks(
        self,
        donor_id: str,
        token_ids: list[int],
    ) -> int:
        """Index all full-size chunks from a donor's token sequence.

        Hashing is done outside the lock for better concurrency.

        Args:
            donor_id: Donor request ID.
            token_ids: Full token sequence of the donor.

        Returns:
            Number of chunks indexed.
        """
        # Phase 1: hash outside the lock (CPU-bound, no shared state)
        chunk_data: list[tuple[str, int]] = []  # (hash, chunk_start)
        for chunk_start in range(0, len(token_ids), self._chunk_size):
            chunk = token_ids[chunk_start:chunk_start + self._chunk_size]
            if len(chunk) == self._chunk_size:
                chunk_data.append((_chunk_hash(chunk), chunk_start))

        if not chunk_data:
            return 0

        # Phase 2: insert under lock
        chunks_indexed = 0

        with self._lock:
            # Evict LRU donors if at capacity
            while len(self._donor_hashes) >= self._max_donors:
                self._evict_lru()

            # Intern the donor_id string
            if donor_id not in self._interned:
                self._interned[donor_id] = donor_id
            interned_id = self._interned[donor_id]

            # Skip if already indexed
            if interned_id in self._donor_hashes:
                self._donor_hashes.move_to_end(interned_id)
                return 0

            chunk_hashes: set[str] = set()

            for h, chunk_start in chunk_data:
                chunk_hashes.add(h)

                loc = ChunkLocation(
                    donor_id=interned_id,
                    chunk_idx=chunk_start // self._chunk_size,
                    pos=chunk_start,
                )

                if h in self._index:
                    self._index[h].append(loc)
                else:
                    self._index[h] = [loc]

                chunks_indexed += 1

            self._donor_hashes[interned_id] = chunk_hashes

        if chunks_indexed > 0:
            logger.debug(
                "ChunkIndex: indexed %d chunks for donor %s",
                chunks_indexed, donor_id,
            )

        return chunks_indexed

    def remove_donor(self, donor_id: str) -> None:
        """Remove all chunk entries for a donor.

        Args:
            donor_id: Donor to remove.
        """
        with self._lock:
            interned_id = self._interned.get(donor_id, donor_id)
            hashes = self._donor_hashes.pop(interned_id, None)
            if hashes is None:
                return

            for h in hashes:
                locs = self._index.get(h)
                if locs is not None:
                    self._index[h] = [
                        loc for loc in locs if loc.donor_id != interned_id
                    ]
                    if not self._index[h]:
                        del self._index[h]

            self._interned.pop(donor_id, None)

    def lookup_chunk(self, chunk_tokens: list[int]) -> list[ChunkLocation]:
        """Find all donors containing this exact chunk.

        Args:
            chunk_tokens: Token IDs of the chunk to look up.

        Returns:
            List of ChunkLocations, empty if no match.
        """
        if len(chunk_tokens) != self._chunk_size:
            return []

        h = _chunk_hash(chunk_tokens)

        with self._lock:
            return list(self._index.get(h, []))

    def lookup_hash(self, chunk_hash: str) -> list[ChunkLocation]:
        """Look up by pre-computed hash (avoids re-hashing).

        Args:
            chunk_hash: MD5 hex digest of packed chunk tokens.

        Returns:
            List of ChunkLocations, empty if no match.
        """
        with self._lock:
            return list(self._index.get(chunk_hash, []))

    def find_matching_chunks(
        self,
        target_tokens: list[int],
        min_matches: int = 1,
    ) -> dict[int, list[ChunkLocation]]:
        """Find ChunkIndex matches for all chunks in a target sequence.

        Args:
            target_tokens: Target token sequence.
            min_matches: Minimum matches to include a chunk.

        Returns:
            Dict of target_chunk_idx → list[ChunkLocation] for matched chunks.
        """
        matches: dict[int, list[ChunkLocation]] = {}

        with self._lock:
            for chunk_start in range(0, len(target_tokens), self._chunk_size):
                chunk = target_tokens[chunk_start:chunk_start + self._chunk_size]
                if len(chunk) != self._chunk_size:
                    continue

                h = _chunk_hash(chunk)
                locs = self._index.get(h)
                if locs and len(locs) >= min_matches:
                    chunk_idx = chunk_start // self._chunk_size
                    matches[chunk_idx] = list(locs)

        return matches

    def _evict_lru(self) -> None:
        """Evict the least recently used donor. Caller holds lock."""
        if not self._donor_hashes:
            return

        evicted_id, hashes = self._donor_hashes.popitem(last=False)

        for h in hashes:
            locs = self._index.get(h)
            if locs is not None:
                self._index[h] = [
                    loc for loc in locs if loc.donor_id != evicted_id
                ]
                if not self._index[h]:
                    del self._index[h]

        self._interned.pop(evicted_id, None)


def _chunk_hash(tokens: list[int]) -> str:
    """Hash a chunk of token IDs.

    Uses SHA-256 truncated to 32 hex chars (128 bits) for FIPS compliance.
    Matches alignment._chunk_hash output format.
    """
    data = struct.pack(f"<{len(tokens)}I", *tokens)
    return hashlib.sha256(data).hexdigest()[:32]


def chunk_hash_sequence(
    token_ids: list[int],
    chunk_size: int = 256,
) -> list[str]:
    """Compute chunk hashes for an entire token sequence.

    Args:
        token_ids: Full token sequence.
        chunk_size: Tokens per chunk.

    Returns:
        List of MD5 hex digests, one per full-size chunk.
    """
    hashes = []
    for i in range(0, len(token_ids), chunk_size):
        chunk = token_ids[i:i + chunk_size]
        if len(chunk) == chunk_size:
            hashes.append(_chunk_hash(chunk))
    return hashes

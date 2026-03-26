"""Token-level inverted index for scalable fuzzy chunk matching.

Maps token_id → list[ChunkRef] for fast identification of donor chunks
that share tokens with a target chunk. Enables O(chunk_size) fuzzy
candidate discovery instead of O(N_donors × N_chunks) brute force.

Architecture:
  - Each donor chunk is registered with all its unique token IDs
  - For a target chunk, look up its tokens → get candidate donor chunks
  - Count co-occurring candidates → chunks with ≥ threshold shared tokens
    are fuzzy match candidates
  - Only these candidates need full overlap verification

Memory: ~50 bytes per entry. 100K donors × 30 chunks × 256 tokens
  = 768M entries but most tokens appear in many chunks, so the actual
  index is much smaller (each token maps to a set of chunk locations).

Thread-safe with the same RW lock pattern as ChunkIndex.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict, defaultdict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkRef:
    """Reference to a specific chunk within a donor."""

    donor_id: str
    chunk_idx: int


class TokenIndex:
    """Inverted index: token_id → set[ChunkRef] for fuzzy chunk discovery.

    For each target chunk, finds donor chunks that share many tokens —
    candidates for 90%+ token overlap fuzzy matching.

    Args:
        max_donors: Maximum donors to index.
        chunk_size: Token chunk size (must match engine).
        min_shared_tokens: Minimum shared unique tokens for a candidate.
    """

    def __init__(
        self,
        max_donors: int = 100_000,
        chunk_size: int = 256,
        min_shared_fraction: float = 0.50,
    ) -> None:
        self._max_donors = max_donors
        self._chunk_size = chunk_size
        self._min_shared_fraction = min_shared_fraction
        self._lock = threading.Lock()

        # token_id → set[ChunkRef]
        self._index: dict[int, set[ChunkRef]] = defaultdict(set)
        # donor_id → list of chunk token sets (for eviction cleanup)
        self._donor_tokens: OrderedDict[str, list[set[int]]] = OrderedDict()

    @property
    def num_donors(self) -> int:
        with self._lock:
            return len(self._donor_tokens)

    def add_donor(self, donor_id: str, token_ids: list[int]) -> int:
        """Index all chunks from a donor's token sequence.

        Returns number of chunks indexed.
        """
        # Pre-compute chunk token sets outside the lock
        chunk_sets: list[set[int]] = []
        chunk_refs: list[tuple[set[int], ChunkRef]] = []

        for chunk_idx in range(0, len(token_ids), self._chunk_size):
            chunk = token_ids[chunk_idx : chunk_idx + self._chunk_size]
            if len(chunk) < self._chunk_size // 2:
                continue  # Skip very short trailing chunks
            token_set = set(chunk)
            ref = ChunkRef(donor_id=donor_id, chunk_idx=chunk_idx // self._chunk_size)
            chunk_sets.append(token_set)
            chunk_refs.append((token_set, ref))

        if not chunk_refs:
            return 0

        with self._lock:
            # Evict LRU if at capacity
            while len(self._donor_tokens) >= self._max_donors:
                self._evict_lru()

            # Skip if already indexed
            if donor_id in self._donor_tokens:
                self._donor_tokens.move_to_end(donor_id)
                return 0

            # Index all tokens for each chunk
            for token_set, ref in chunk_refs:
                for tok in token_set:
                    self._index[tok].add(ref)

            self._donor_tokens[donor_id] = [ts for ts, _ in chunk_refs]

        return len(chunk_refs)

    def remove_donor(self, donor_id: str) -> None:
        """Remove a donor's chunks from the index."""
        with self._lock:
            chunk_sets = self._donor_tokens.pop(donor_id, None)
            if chunk_sets is None:
                return
            # Remove all ChunkRefs for this donor
            for token_set in chunk_sets:
                for tok in token_set:
                    refs = self._index.get(tok)
                    if refs:
                        refs.discard(ChunkRef(donor_id=donor_id, chunk_idx=0))
                        # Actually need to remove all refs for this donor
                        to_remove = {r for r in refs if r.donor_id == donor_id}
                        refs -= to_remove
                        if not refs:
                            del self._index[tok]

    def find_fuzzy_candidates(
        self,
        target_chunk: list[int],
        min_shared: int | None = None,
    ) -> list[tuple[ChunkRef, int]]:
        """Find donor chunks that share many tokens with the target chunk.

        Returns list of (ChunkRef, shared_token_count) sorted by count descending.
        Only returns candidates above the minimum shared token threshold.
        """
        target_set = set(target_chunk)
        if min_shared is None:
            min_shared = max(
                10,
                int(len(target_set) * self._min_shared_fraction),
            )

        # Count how many target tokens each donor chunk shares
        candidate_counts: dict[ChunkRef, int] = defaultdict(int)

        with self._lock:
            for tok in target_set:
                refs = self._index.get(tok)
                if refs:
                    for ref in refs:
                        candidate_counts[ref] += 1

        # Filter by minimum shared tokens and sort
        candidates = [
            (ref, count) for ref, count in candidate_counts.items() if count >= min_shared
        ]
        candidates.sort(key=lambda x: x[1], reverse=True)

        return candidates

    def _evict_lru(self) -> None:
        """Evict least recently used donor. Caller holds lock."""
        if not self._donor_tokens:
            return
        evicted_id, chunk_sets = self._donor_tokens.popitem(last=False)
        for token_set in chunk_sets:
            for tok in token_set:
                refs = self._index.get(tok)
                if refs:
                    to_remove = {r for r in refs if r.donor_id == evicted_id}
                    refs -= to_remove
                    if not refs:
                        del self._index[tok]

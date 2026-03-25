"""Token-level Locality-Sensitive Hashing index for SemShareKV.

Implements the LSH matching from the paper: h(x) = floor((x·r + b) / w)
where r is a random projection vector, b is a random offset, w is bucket width.

Multiple hash tables increase recall. Only hash codes are stored (not full
embeddings) to keep memory bounded at scale.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from semblend_core.semshare.config import SemShareConfig


@dataclass(frozen=True)
class LSHTable:
    """A single LSH hash table with random projections."""

    projections: np.ndarray  # shape [num_bits, embed_dim]
    offsets: np.ndarray  # shape [num_bits]
    w: float

    def hash_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Hash a batch of vectors. Returns int64 hash codes.

        Args:
            vectors: shape [n, embed_dim]

        Returns:
            hash_codes: shape [n] int64
        """
        # h(x) = floor((x · r + b) / w) for each bit
        projected = vectors @ self.projections.T + self.offsets  # [n, num_bits]
        quantized = np.floor(projected / self.w).astype(np.int64)  # [n, num_bits]

        # Combine bits into a single hash via polynomial hashing
        # Use prime multipliers to reduce collisions
        primes = np.array(
            [np.int64(p) for p in _first_n_primes(quantized.shape[1])],
            dtype=np.int64,
        )
        codes = np.sum(quantized * primes, axis=1)
        return codes


@dataclass(frozen=True)
class TokenLocation:
    """Location of a token in a donor's sequence."""

    donor_id: str
    token_pos: int


@dataclass
class DonorEntry:
    """Per-donor data in the LSH index."""

    donor_id: str
    hash_codes: list[np.ndarray]  # one per table, each shape [seq_len]
    seq_len: int
    registered_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class LSHMatch:
    """A single token-level match between target and donor."""

    target_pos: int
    donor_id: str
    donor_pos: int
    similarity: float


class LSHIndex:
    """Multi-table LSH index for token-level matching across donors.

    Stores only hash codes (not full embeddings) to bound memory usage.
    At query time, finds donor tokens in the same hash bucket and computes
    similarity from hash distance.
    """

    def __init__(self, config: SemShareConfig) -> None:
        self._config = config
        self._rng = np.random.RandomState(config.lsh_seed)

        # Initialize hash tables with random projections
        self._tables: tuple[LSHTable, ...] = tuple(
            LSHTable(
                projections=self._rng.randn(
                    config.lsh_num_bits, config.embed_dim
                ).astype(np.float32),
                offsets=self._rng.uniform(
                    0, config.lsh_w, size=config.lsh_num_bits
                ).astype(np.float32),
                w=config.lsh_w,
            )
            for _ in range(config.lsh_num_tables)
        )

        # Donor storage: donor_id -> DonorEntry
        self._donors: dict[str, DonorEntry] = {}

        # Inverted index: (table_idx, hash_code) -> list[TokenLocation]
        self._buckets: dict[tuple[int, int], list[TokenLocation]] = {}

    @property
    def num_donors(self) -> int:
        return len(self._donors)

    @property
    def config(self) -> SemShareConfig:
        return self._config

    def register_donor(
        self,
        donor_id: str,
        embeddings: np.ndarray,
    ) -> None:
        """Register a donor's token embeddings in the LSH index.

        Args:
            donor_id: unique identifier for this donor
            embeddings: shape [seq_len, embed_dim] token embeddings
        """
        if donor_id in self._donors:
            return

        self._evict_expired()
        self._evict_if_over_capacity()

        seq_len = embeddings.shape[0]
        all_codes: list[np.ndarray] = []

        for table_idx, table in enumerate(self._tables):
            codes = table.hash_vectors(embeddings)
            all_codes.append(codes)

            # Index each token in this table's buckets
            for pos in range(seq_len):
                code = int(codes[pos])
                key = (table_idx, code)
                bucket = self._buckets.get(key)
                if bucket is None:
                    bucket = []
                    self._buckets[key] = bucket
                bucket.append(TokenLocation(donor_id=donor_id, token_pos=pos))

        self._donors[donor_id] = DonorEntry(
            donor_id=donor_id,
            hash_codes=all_codes,
            seq_len=seq_len,
        )

    def remove_donor(self, donor_id: str) -> None:
        """Remove a donor from the index."""
        entry = self._donors.pop(donor_id, None)
        if entry is None:
            return

        # Remove from buckets
        for table_idx, codes in enumerate(entry.hash_codes):
            for pos in range(entry.seq_len):
                code = int(codes[pos])
                key = (table_idx, code)
                bucket = self._buckets.get(key)
                if bucket is not None:
                    self._buckets[key] = [
                        loc
                        for loc in bucket
                        if not (loc.donor_id == donor_id and loc.token_pos == pos)
                    ]
                    if not self._buckets[key]:
                        del self._buckets[key]

    def query(
        self,
        target_embeddings: np.ndarray,
        candidate_donor_ids: Sequence[str] | None = None,
    ) -> list[list[LSHMatch]]:
        """Find matching donor tokens for each target token.

        Args:
            target_embeddings: shape [seq_len, embed_dim]
            candidate_donor_ids: if provided, only match against these donors

        Returns:
            List of length seq_len, where each element is a list of LSHMatch
            candidates for that target position, sorted by similarity descending.
        """
        seq_len = target_embeddings.shape[0]
        donor_filter = set(candidate_donor_ids) if candidate_donor_ids else None

        # Hash target tokens in all tables
        target_codes: list[np.ndarray] = [
            table.hash_vectors(target_embeddings) for table in self._tables
        ]

        results: list[list[LSHMatch]] = []

        for pos in range(seq_len):
            # Collect candidates from all tables (union for higher recall)
            candidates: dict[tuple[str, int], int] = {}  # (donor_id, donor_pos) -> vote count

            for table_idx in range(len(self._tables)):
                code = int(target_codes[table_idx][pos])
                key = (table_idx, code)
                bucket = self._buckets.get(key)
                if bucket is None:
                    continue

                for loc in bucket:
                    if donor_filter is not None and loc.donor_id not in donor_filter:
                        continue
                    cand_key = (loc.donor_id, loc.token_pos)
                    candidates[cand_key] = candidates.get(cand_key, 0) + 1

            # Convert vote counts to similarity scores
            # More tables agreeing = higher similarity
            matches: list[LSHMatch] = []
            num_tables = len(self._tables)
            for (did, dpos), votes in candidates.items():
                similarity = votes / num_tables
                if similarity >= self._config.min_similarity:
                    matches.append(
                        LSHMatch(
                            target_pos=pos,
                            donor_id=did,
                            donor_pos=dpos,
                            similarity=similarity,
                        )
                    )

            matches.sort(key=lambda m: m.similarity, reverse=True)
            results.append(matches)

        return results

    def _evict_expired(self) -> None:
        """Remove donors older than TTL."""
        now = time.time()
        expired = [
            did
            for did, entry in self._donors.items()
            if now - entry.registered_at > self._config.ttl_seconds
        ]
        for did in expired:
            self.remove_donor(did)

    def _evict_if_over_capacity(self) -> None:
        """Evict oldest donors if over capacity."""
        while len(self._donors) >= self._config.max_donors:
            oldest = min(self._donors.values(), key=lambda e: e.registered_at)
            self.remove_donor(oldest.donor_id)


def _first_n_primes(n: int) -> list[int]:
    """Generate first n primes for polynomial hashing."""
    primes: list[int] = []
    candidate = 2
    while len(primes) < n:
        if all(candidate % p != 0 for p in primes):
            primes.append(candidate)
        candidate += 1
    return primes

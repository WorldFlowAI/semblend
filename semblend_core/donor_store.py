"""Numpy-based in-process donor store for SemBlend.

Stores donor embeddings in a numpy matrix [N, dim] for vectorized cosine
similarity search. No SimHash pre-filter — embedding search is the only
index, which is correct for all overlap patterns (REORDER, PARTIAL, etc.)
and fast enough at N<=10K via brute-force matrix multiply (~2ms).

For scale beyond 10K, the search path should be replaced with LSH on
embeddings or CAGRA ANN — NOT token-level hashing (SimHash).

Performance targets:
  - Lookup at N=1000: <3ms (cosine + alignment)
  - Lookup at N=10000: <5ms
  - Add donor: O(1) append + O(chunks) ChunkIndex indexing
  - LRU eviction at capacity
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from semblend_core.alignment import (
    DEFAULT_CHUNK_SIZE,
    AlignmentResult,
    compute_alignment,
    estimate_reuse_ratio,
)
from semblend_core.chunk_index import ChunkIndex
from semblend_core.token_index import TokenIndex

logger = logging.getLogger(__name__)


@dataclass
class DonorNode:
    """A cached donor entry with embedding and token data."""

    request_id: str
    token_ids: list[int]
    embedding: np.ndarray | None  # [dim] normalized
    timestamp: float
    prompt_text: str = ""
    segment_embeddings: object | None = None  # SegmentEmbeddings (transient, pre-PQ)


@dataclass(frozen=True)
class DonorMatch:
    """Result of a donor lookup with alignment."""

    donor: DonorNode
    similarity: float
    alignment: AlignmentResult


class DonorStore:
    """In-process donor store with numpy vectorized cosine search.

    Uses brute-force cosine similarity on a numpy embedding matrix.
    At N=10K with 384-dim embeddings, a single matrix multiply takes ~2ms.
    For N>10K, replace with LSH on embeddings or CAGRA ANN index.

    Integrates ChunkIndex for O(1) cross-donor chunk lookup, enabling
    multi-turn fast path and multi-donor composite KV assembly.

    Args:
        max_entries: Maximum number of donors to store.
        embedding_dim: Dimension of embeddings (384 for MiniLM).
        min_similarity: Minimum cosine similarity for donor candidates.
        chunk_size: KV block size for alignment (from backend).
        chunk_index: Optional pre-configured ChunkIndex. If None, creates one.
    """

    def __init__(
        self,
        max_entries: int = 10_000,
        embedding_dim: int = 384,
        min_similarity: float = 0.60,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_index: ChunkIndex | None = None,
    ) -> None:
        self._max_entries = max_entries
        self._embedding_dim = embedding_dim
        self._min_similarity = min_similarity
        self._chunk_size = chunk_size

        # Ordered dict for LRU eviction
        self._entries: OrderedDict[str, DonorNode] = OrderedDict()

        # Numpy matrix for vectorized cosine search
        self._embeddings = np.zeros((max_entries, embedding_dim), dtype=np.float32)
        self._valid_mask = np.zeros(max_entries, dtype=bool)
        self._id_to_idx: dict[str, int] = {}
        self._next_idx = 0

        # ChunkIndex for O(1) cross-donor chunk lookup (exact hash)
        self._chunk_index = chunk_index or ChunkIndex(
            max_donors=max_entries,
            chunk_size=chunk_size,
        )

        # TokenIndex for scalable fuzzy cross-donor chunk matching
        self._token_index = TokenIndex(
            max_donors=max_entries,
            chunk_size=chunk_size,
        )

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def token_index(self) -> TokenIndex:
        """Access the TokenIndex for fuzzy cross-donor chunk lookup."""
        return self._token_index

    @property
    def chunk_index(self) -> ChunkIndex:
        """Access the ChunkIndex for cross-donor chunk lookup."""
        return self._chunk_index

    def add_donor(self, node: DonorNode) -> None:
        """Add a donor to the store with O(1) append + O(chunks) indexing.

        If at capacity, evicts the least recently used entry.
        Also indexes the donor's chunks in the ChunkIndex for O(1)
        cross-donor lookup.
        """
        if node.request_id in self._entries:
            self._entries.move_to_end(node.request_id)
            return

        # Evict LRU if at capacity
        while len(self._entries) >= self._max_entries:
            evicted_id, _ = self._entries.popitem(last=False)
            evicted_idx = self._id_to_idx.pop(evicted_id, None)
            if evicted_idx is not None:
                self._valid_mask[evicted_idx] = False
            # Remove from indexes
            self._chunk_index.remove_donor(evicted_id)
            self._token_index.remove_donor(evicted_id)

        # Assign storage index (reuse evicted slots or append)
        idx = self._next_idx % self._max_entries
        self._next_idx += 1

        # Store embedding if available
        if node.embedding is not None and len(node.embedding) == self._embedding_dim:
            self._embeddings[idx] = node.embedding
        else:
            self._embeddings[idx] = 0.0

        self._valid_mask[idx] = True
        self._id_to_idx[node.request_id] = idx
        self._entries[node.request_id] = node

        # Index chunks in ChunkIndex (exact hash) and TokenIndex (fuzzy)
        if node.token_ids:
            self._chunk_index.add_donor_chunks(node.request_id, node.token_ids)
            self._token_index.add_donor(node.request_id, node.token_ids)

    def find_donor(
        self,
        query_embedding: np.ndarray | None,
        query_tokens: list[int],
        top_k: int = 5,
        min_reuse_ratio: float = 0.5,
    ) -> DonorMatch | None:
        """Find the best donor for a query via embedding cosine + alignment.

        Two-phase scoring:
          1. Cosine similarity → top-k candidates (threshold >= min_similarity)
          2. Fast Jaccard pre-filter + estimate_reuse_ratio (no SlotActions)
          3. Full alignment only for the winner

        Args:
            query_embedding: Query embedding vector [dim]. Required.
            query_tokens: Query token IDs for alignment.
            top_k: Number of top cosine candidates to align.
            min_reuse_ratio: Minimum alignment reuse ratio.

        Returns:
            DonorMatch with alignment, or None if no suitable donor found.
        """
        if not self._entries:
            return None

        t0 = time.monotonic()

        n_valid = int(self._valid_mask.sum())
        if n_valid == 0:
            return None

        if query_embedding is None or len(query_embedding) != self._embedding_dim:
            return None

        valid_indices = np.where(self._valid_mask)[0]

        # Step 1: Cosine similarity (vectorized matrix multiply)
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        similarities = self._embeddings[valid_indices] @ query_norm

        # Filter by threshold
        cos_mask = similarities >= self._min_similarity
        if not cos_mask.any():
            max_sim = float(similarities.max()) if len(similarities) > 0 else 0.0
            logger.debug(
                "DonorStore: no candidates above threshold (max_sim=%.4f < %.2f)",
                max_sim,
                self._min_similarity,
            )
            return None

        candidate_indices = valid_indices[cos_mask]
        similarities = similarities[cos_mask]

        # Top-k by similarity
        if len(candidate_indices) > top_k:
            topk_idx = np.argpartition(similarities, -top_k)[-top_k:]
            candidate_indices = candidate_indices[topk_idx]
            similarities = similarities[topk_idx]

        # Step 2: Two-phase scoring
        idx_to_id = {v: k for k, v in self._id_to_idx.items()}

        # Pre-compute query token set for fast Jaccard pre-filter
        query_set = set(query_tokens)

        # Phase 1: Fast scoring with estimate_reuse_ratio (no SlotActions)
        best_candidate: tuple[float, float, DonorNode] | None = None
        best_score = 0.0

        for i, storage_idx in enumerate(candidate_indices):
            donor_id = idx_to_id.get(int(storage_idx))
            if donor_id is None:
                continue

            donor = self._entries.get(donor_id)
            if donor is None or donor.token_ids == query_tokens:
                continue

            # Fast Jaccard pre-filter
            donor_set = set(donor.token_ids)
            intersection = len(query_set & donor_set)
            union = len(query_set | donor_set)
            jaccard = intersection / union if union > 0 else 0.0
            if jaccard < min_reuse_ratio:
                continue

            # Fast reuse estimation (no SlotAction creation)
            reuse = estimate_reuse_ratio(
                donor.token_ids,
                query_tokens,
                chunk_size=self._chunk_size,
            )
            if reuse < min_reuse_ratio:
                continue

            score = float(similarities[i]) * reuse
            if score > best_score:
                best_score = score
                best_candidate = (score, float(similarities[i]), donor)

        # Phase 2: Full alignment only for the winner
        best_match: DonorMatch | None = None
        if best_candidate is not None:
            _, sim, donor = best_candidate
            alignment = compute_alignment(
                donor.token_ids,
                query_tokens,
                chunk_size=self._chunk_size,
            )
            if alignment.reuse_ratio >= min_reuse_ratio:
                best_match = DonorMatch(
                    donor=donor,
                    similarity=sim,
                    alignment=alignment,
                )

        elapsed_ms = (time.monotonic() - t0) * 1000
        if elapsed_ms > 5:
            logger.debug(
                "DonorStore.find_donor: %.1fms (N=%d, candidates=%d)",
                elapsed_ms,
                n_valid,
                len(candidate_indices),
            )

        return best_match

    def find_donors(
        self,
        query_embedding: np.ndarray | None,
        query_tokens: list[int],
        top_k: int = 5,
        min_reuse_ratio: float = 0.5,
    ) -> list[DonorMatch]:
        """Find multiple donor candidates ranked by score with recency bias.

        Returns up to top_k candidates sorted by score (descending).
        Score = similarity * reuse_ratio + recency_bonus.
        This allows the connector to try each candidate until it finds
        one whose KV is still in the engine cache.
        """
        if not self._entries or query_embedding is None:
            return []

        if len(query_embedding) != self._embedding_dim:
            return []

        n_valid = int(self._valid_mask.sum())
        if n_valid == 0:
            return []

        valid_indices = np.where(self._valid_mask)[0]

        # Cosine similarity
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        similarities = self._embeddings[valid_indices] @ query_norm

        cos_mask = similarities >= self._min_similarity
        if not cos_mask.any():
            return []

        candidate_indices = valid_indices[cos_mask]
        similarities = similarities[cos_mask]

        # Widen top-k to give more candidates for fallback
        fetch_k = min(len(candidate_indices), top_k * 2)
        if len(candidate_indices) > fetch_k:
            topk_idx = np.argpartition(similarities, -fetch_k)[-fetch_k:]
            candidate_indices = candidate_indices[topk_idx]
            similarities = similarities[topk_idx]

        # Scoring with fast reuse estimation + recency tiebreaker
        idx_to_id = {v: k for k, v in self._id_to_idx.items()}
        now = time.monotonic()
        scored_candidates: list[tuple[float, float, DonorNode]] = []

        # Pre-compute query token set for fast Jaccard pre-filter
        query_set = set(query_tokens)

        for i, storage_idx in enumerate(candidate_indices):
            donor_id = idx_to_id.get(int(storage_idx))
            if donor_id is None:
                continue

            donor = self._entries.get(donor_id)
            if donor is None or donor.token_ids == query_tokens:
                continue

            # Fast Jaccard pre-filter
            donor_set = set(donor.token_ids)
            intersection = len(query_set & donor_set)
            union = len(query_set | donor_set)
            jaccard = intersection / union if union > 0 else 0.0
            if jaccard < min_reuse_ratio:
                continue

            # Fast reuse estimation (no SlotAction creation)
            reuse = estimate_reuse_ratio(
                donor.token_ids,
                query_tokens,
                chunk_size=self._chunk_size,
            )
            if reuse < min_reuse_ratio:
                continue

            sim = float(similarities[i])
            age = now - donor.timestamp
            recency_bonus = max(0.0, 0.01 * (1.0 - age / 60.0))
            score = sim * reuse + recency_bonus

            scored_candidates.append((score, sim, donor))

        # Sort by score descending, take top_k
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        top_candidates = scored_candidates[:top_k]

        # Full alignment only for winners (builds SlotAction lists)
        matches: list[DonorMatch] = []
        for score, sim, donor in top_candidates:
            alignment = compute_alignment(
                donor.token_ids,
                query_tokens,
                chunk_size=self._chunk_size,
            )
            if alignment.reuse_ratio < min_reuse_ratio:
                continue

            matches.append(
                DonorMatch(
                    donor=donor,
                    similarity=sim,
                    alignment=alignment,
                )
            )

        return matches

    def find_candidates_jaccard(
        self,
        query_tokens: list[int],
        top_k: int = 5,
        min_jaccard: float = 0.30,
        min_reuse_ratio: float = 0.5,
    ) -> list[DonorMatch]:
        """Find donor candidates using Jaccard token-set similarity.

        No embedding is needed -- operates purely on token IDs.
        """
        if not self._entries or not query_tokens:
            return []

        query_set = set(query_tokens)
        scored: list[tuple[float, DonorNode]] = []

        for donor in self._entries.values():
            if donor.token_ids == query_tokens:
                continue

            donor_set = set(donor.token_ids)
            intersection = len(query_set & donor_set)
            union = len(query_set | donor_set)
            jaccard = intersection / union if union > 0 else 0.0

            if jaccard >= min_jaccard:
                scored.append((jaccard, donor))

        # Sort by Jaccard descending, take top candidates
        scored.sort(key=lambda x: x[0], reverse=True)
        candidates = scored[: top_k * 2]

        # Run alignment on top candidates
        matches: list[tuple[float, DonorMatch]] = []
        for jaccard, donor in candidates:
            alignment = compute_alignment(
                donor.token_ids,
                query_tokens,
                chunk_size=self._chunk_size,
            )
            if alignment.reuse_ratio < min_reuse_ratio:
                continue

            score = jaccard * alignment.reuse_ratio
            matches.append(
                (
                    score,
                    DonorMatch(
                        donor=donor,
                        similarity=jaccard,
                        alignment=alignment,
                    ),
                )
            )

        matches.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in matches[:top_k]]

    def get_donor(self, donor_id: str) -> DonorNode | None:
        """Get a donor node by request_id."""
        return self._entries.get(donor_id)

    def get_donor_tokens(self, donor_id: str) -> list[int] | None:
        """Get a donor's token IDs by request_id.

        Used by multi_donor_alignment for fuzzy fallback.
        """
        node = self._entries.get(donor_id)
        return node.token_ids if node is not None else None

    def get_all_donor_tokens(self) -> dict[str, list[int]]:
        """Get token IDs for all donors in the store.

        Used by multi_donor_alignment to build the donor_token_store.
        Returns defensive copies to prevent mutation of internal state.
        """
        return {did: list(node.token_ids) for did, node in self._entries.items()}

    def find_multi_donor(
        self,
        query_tokens: list[int],
        min_reuse_ratio: float = 0.5,
        context_gate: bool | None = None,
        min_fuzzy_overlap: float = 0.90,
        pq_store: object | None = None,
        target_text: str = "",
        embedder: object | None = None,
    ) -> object | None:
        """Find multi-donor composite alignment using ChunkIndex + PQ semantics.

        Delegates to multi_donor_alignment.compute_multi_donor_alignment()
        for cross-donor chunk matching. When pq_store and embedder are
        provided, also performs semantic per-chunk matching via PQ ADC.

        Args:
            query_tokens: Target token sequence.
            min_reuse_ratio: Minimum combined reuse ratio.
            context_gate: Override for context gate.
            min_fuzzy_overlap: Minimum token overlap for fuzzy match.
            pq_store: Optional PQSegmentStore for semantic chunk matching.
            target_text: Decoded prompt text for per-chunk embedding.
            embedder: Embedder instance for per-chunk embedding.

        Returns:
            MultiDonorAlignmentResult or None.
        """
        from semblend_core.multi_donor_alignment import (
            compute_multi_donor_alignment,
        )

        donor_token_store = self.get_all_donor_tokens()

        result = compute_multi_donor_alignment(
            target_tokens=query_tokens,
            chunk_index=self._chunk_index,
            donor_token_store=donor_token_store,
            chunk_size=self._chunk_size,
            context_gate=context_gate,
            min_fuzzy_overlap=min_fuzzy_overlap,
            pq_store=pq_store,
            target_text=target_text,
            embedder=embedder,
            token_index=self._token_index,
        )

        if result is not None and result.reuse_ratio < min_reuse_ratio:
            return None

        return result

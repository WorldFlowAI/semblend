"""GPU-accelerated donor store using NVIDIA cuVS CAGRA ANN index.

Replaces the O(N) numpy cosine scan in DonorStore with:
  - cuVS brute_force: exact GPU search (good for N < 64K)
  - cuVS CAGRA: approximate GPU ANN (sub-ms at any N, N >= 64)

Performance targets:
  - Lookup at N=100K:   <0.5ms (CAGRA)
  - Lookup at N=1M:     <1ms   (CAGRA)
  - Lookup at N=10K:    <0.3ms (brute_force or CAGRA)
  - Build at N=100K:    <5s    (one-time, amortized)

Activation:
  Set SEMBLEND_USE_CAGRA=1 to use CAGRADonorStore instead of DonorStore.
  Falls back to numpy DonorStore if cuVS/cupy not available.

Compatibility:
  Presents the same interface as DonorStore:
    - add_donor(node: DonorNode) -> None
    - find_donor(query_embedding, query_tokens, top_k, min_reuse_ratio) -> DonorMatch | None
    - find_donors(query_embedding, query_tokens, top_k, min_reuse_ratio) -> list[DonorMatch]
"""
from __future__ import annotations

import logging
import os
import time
from collections import OrderedDict

import numpy as np

from semblend_core.alignment import AlignmentResult, compute_alignment
from semblend_core.donor_store import DonorMatch, DonorNode

logger = logging.getLogger(__name__)

# CAGRA requires at least this many dataset vectors to build an index.
_CAGRA_MIN_N = 64


def _has_cuvs() -> bool:
    try:
        import cupy  # noqa: F401
        import cuvs.neighbors.brute_force  # noqa: F401
        return True
    except ImportError:
        return False


class CAGRADonorStore:
    """GPU-accelerated donor lookup using NVIDIA cuVS.

    Uses cuVS brute_force for N < _CAGRA_MIN_N, cuVS CAGRA for N >= _CAGRA_MIN_N.
    Falls back to numpy for any GPU error.

    Index is rebuilt lazily on next search after a new donor is added.
    For very high add rates, consider batched adds followed by explicit rebuild.

    Args:
        max_entries: Maximum donors to store (LRU eviction).
        embedding_dim: Embedding dimension (384 for MiniLM).
        min_similarity: Minimum cosine similarity for candidates.
        rebuild_every: Rebuild index after this many adds (0 = on every find).
    """

    def __init__(
        self,
        max_entries: int = 100_000,
        embedding_dim: int = 384,
        min_similarity: float = 0.60,
        rebuild_every: int = 0,
    ) -> None:
        self._max_entries = max_entries
        self._embedding_dim = embedding_dim
        self._min_similarity = min_similarity
        self._rebuild_every = rebuild_every

        self._entries: OrderedDict[str, DonorNode] = OrderedDict()
        self._emb_list: list[np.ndarray] = []
        self._id_list: list[str] = []

        self._index: object | None = None
        self._index_kind: str = ""  # "cagra" | "brute" | ""
        self._index_dataset_gpu: object | None = None  # CRITICAL: keep dataset alive for CAGRA
        self._dirty: bool = True
        self._adds_since_rebuild: int = 0

        self._available = _has_cuvs()
        if not self._available:
            logger.warning(
                "cuVS/cupy not available — CAGRADonorStore will fall back to numpy search"
            )

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def add_donor(self, node: DonorNode) -> None:
        """Add a donor (O(1) append). Index rebuilt lazily."""
        if node.request_id in self._entries:
            self._entries.move_to_end(node.request_id)
            return

        # LRU eviction
        while len(self._entries) >= self._max_entries:
            evicted_id, _ = self._entries.popitem(last=False)
            if evicted_id in self._id_list:
                idx = self._id_list.index(evicted_id)
                self._id_list.pop(idx)
                self._emb_list.pop(idx)
            self._dirty = True

        emb = (
            node.embedding.astype(np.float32)
            if node.embedding is not None and len(node.embedding) == self._embedding_dim
            else np.zeros(self._embedding_dim, dtype=np.float32)
        )
        self._emb_list.append(emb)
        self._id_list.append(node.request_id)
        self._entries[node.request_id] = node
        self._dirty = True
        self._adds_since_rebuild += 1

    def _rebuild_index(self) -> None:
        """Rebuild cuVS index from current embeddings."""
        if not self._dirty or not self._emb_list:
            return
        if not self._available:
            self._dirty = False
            return

        try:
            import cupy as cp
            from cuvs.neighbors import brute_force, cagra

            data = np.stack(self._emb_list).astype(np.float32)
            # Normalize rows for cosine similarity
            norms = np.linalg.norm(data, axis=1, keepdims=True)
            data = data / (norms + 1e-8)
            data_gpu = cp.asarray(data)
            # CRITICAL: keep data_gpu alive on self so CAGRA index doesn't hold
            # a dangling pointer. cagra_donor_store.py dataset lifetime bug
            # (cuvs-cagra-python-dataset-lifetime skill).
            self._index_dataset_gpu = data_gpu

            n = len(data)
            t0 = time.monotonic()
            if n >= _CAGRA_MIN_N:
                params = cagra.IndexParams(metric="cosine")
                self._index = cagra.build(params, data_gpu)
                self._index_kind = "cagra"
            else:
                self._index = brute_force.build(data_gpu, metric="cosine")
                self._index_kind = "brute"

            build_ms = (time.monotonic() - t0) * 1000
            logger.debug(
                "CAGRADonorStore: rebuilt %s index (N=%d, %.1fms)",
                self._index_kind, n, build_ms,
            )
            self._dirty = False
            self._adds_since_rebuild = 0

        except Exception:
            logger.warning("CAGRADonorStore: index build failed, falling back to numpy", exc_info=True)
            self._index = None
            self._index_kind = ""
            self._dirty = False

    def _gpu_search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Search using cuVS index. Returns (donor_id, similarity) pairs."""
        if not self._available or not self._emb_list:
            return []

        self._rebuild_index()

        if self._index is None:
            return self._numpy_fallback(query_embedding, top_k)

        try:
            import cupy as cp
            from cuvs.neighbors import brute_force, cagra

            q = query_embedding.astype(np.float32)
            q = q / (np.linalg.norm(q) + 1e-8)
            q_gpu = cp.asarray(q[np.newaxis])

            k = min(top_k, len(self._emb_list))
            if self._index_kind == "cagra":
                sp = cagra.SearchParams()
                distances, indices = cagra.search(sp, self._index, q_gpu, k)
            else:
                distances, indices = brute_force.search(self._index, q_gpu, k)

            # Cosine distance → similarity: sim = 1 - dist
            results = []
            for dist, idx in zip(distances[0].tolist(), indices[0].tolist()):
                sim = 1.0 - float(dist)
                if sim >= self._min_similarity and 0 <= int(idx) < len(self._id_list):
                    results.append((self._id_list[int(idx)], sim))
            return results

        except Exception:
            logger.warning("CAGRADonorStore: GPU search failed, falling back to numpy", exc_info=True)
            return self._numpy_fallback(query_embedding, top_k)

    def _numpy_fallback(
        self,
        query_embedding: np.ndarray,
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Numpy cosine fallback when cuVS unavailable."""
        if not self._emb_list:
            return []
        data = np.stack(self._emb_list).astype(np.float32)
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        data = data / (norms + 1e-8)
        q = query_embedding.astype(np.float32)
        q = q / (np.linalg.norm(q) + 1e-8)
        sims = data @ q
        k = min(top_k, len(sims))
        if k == 0:
            return []
        top_idx = np.argpartition(sims, -k)[-k:]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        return [
            (self._id_list[i], float(sims[i]))
            for i in top_idx
            if float(sims[i]) >= self._min_similarity
        ]

    def find_donor(
        self,
        query_embedding: np.ndarray | None,
        query_tokens: list[int],
        top_k: int = 5,
        min_reuse_ratio: float = 0.5,
    ) -> DonorMatch | None:
        """Find the best donor via GPU ANN search + alignment."""
        if not self._entries or query_embedding is None:
            return None
        if len(query_embedding) != self._embedding_dim:
            return None

        candidates = self._gpu_search(query_embedding, top_k)
        if not candidates:
            return None

        best_match: DonorMatch | None = None
        best_score = 0.0

        for donor_id, sim in candidates:
            donor = self._entries.get(donor_id)
            if donor is None or donor.token_ids == query_tokens:
                continue
            alignment = compute_alignment(donor.token_ids, query_tokens)
            if alignment.reuse_ratio < min_reuse_ratio:
                continue
            score = sim * alignment.reuse_ratio
            if score > best_score:
                best_score = score
                best_match = DonorMatch(donor=donor, similarity=sim, alignment=alignment)

        return best_match

    def find_donors(
        self,
        query_embedding: np.ndarray | None,
        query_tokens: list[int],
        top_k: int = 5,
        min_reuse_ratio: float = 0.5,
    ) -> list[DonorMatch]:
        """Find multiple donors via GPU ANN search + alignment, sorted by score."""
        if not self._entries or query_embedding is None:
            return []
        if len(query_embedding) != self._embedding_dim:
            return []

        candidates = self._gpu_search(query_embedding, top_k * 2)
        if not candidates:
            return []

        now = time.monotonic()
        matches: list[tuple[float, DonorMatch]] = []

        for donor_id, sim in candidates:
            donor = self._entries.get(donor_id)
            if donor is None or donor.token_ids == query_tokens:
                continue
            alignment = compute_alignment(donor.token_ids, query_tokens)
            if alignment.reuse_ratio < min_reuse_ratio:
                continue
            age = now - donor.timestamp
            recency_bonus = max(0.0, 0.01 * (1.0 - age / 60.0))
            score = sim * alignment.reuse_ratio + recency_bonus
            matches.append((score, DonorMatch(donor=donor, similarity=sim, alignment=alignment)))

        matches.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in matches[:top_k]]

    # ---- Compatibility with DonorStore.find_candidates_jaccard ----

    def find_candidates_jaccard(
        self,
        query_tokens: list[int],
        top_k: int = 5,
        min_jaccard: float = 0.30,
        min_reuse_ratio: float = 0.5,
    ) -> list[DonorMatch]:
        """Jaccard fallback — identical to DonorStore implementation."""
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
        scored.sort(key=lambda x: x[0], reverse=True)
        matches: list[tuple[float, DonorMatch]] = []
        for jaccard, donor in scored[: top_k * 2]:
            alignment = compute_alignment(donor.token_ids, query_tokens)
            if alignment.reuse_ratio < min_reuse_ratio:
                continue
            score = jaccard * alignment.reuse_ratio
            matches.append((score, DonorMatch(donor=donor, similarity=jaccard, alignment=alignment)))
        matches.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in matches[:top_k]]


def make_donor_store(
    max_entries: int = 10_000,
    embedding_dim: int = 384,
    min_similarity: float = 0.60,
    use_cagra: bool | None = None,
) -> "CAGRADonorStore | DonorStore":
    """Factory: return CAGRADonorStore if available and requested, else DonorStore.

    Args:
        use_cagra: Force CAGRA (True), force numpy (False), or auto-detect (None).
                   Auto-detect uses SEMBLEND_USE_CAGRA env var (default: False).
    """
    from semblend_core.donor_store import DonorStore

    if use_cagra is None:
        use_cagra = os.environ.get("SEMBLEND_USE_CAGRA", "0").lower() in ("1", "true", "yes")

    if use_cagra:
        if _has_cuvs():
            logger.info("CAGRADonorStore: using cuVS GPU-accelerated ANN index")
            return CAGRADonorStore(
                max_entries=max_entries,
                embedding_dim=embedding_dim,
                min_similarity=min_similarity,
            )
        else:
            logger.warning(
                "SEMBLEND_USE_CAGRA=1 but cuVS not available — falling back to numpy DonorStore"
            )

    return DonorStore(
        max_entries=max_entries,
        embedding_dim=embedding_dim,
        min_similarity=min_similarity,
    )

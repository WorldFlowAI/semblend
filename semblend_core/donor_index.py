"""Pluggable vector-index backends for donor-candidate proposal.

The index only proposes candidates; token-level verification downstream
decides what is reused, so an approximate backend can trade recall for
speed without affecting correctness. `BruteForceIndex` is the exact
reference; `DeltaGraphIndex` wraps a batch-built ANN graph with a
brute-forced delta buffer and tombstones so additions stay O(1) between
rebuilds.
"""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np


class DonorVectorIndex(Protocol):
    def add(self, donor_id: str, vector: np.ndarray) -> None: ...
    def remove(self, donor_id: str) -> None: ...
    def search(
        self,
        query: np.ndarray,
        top_k: int,
        allowed: set[str] | None = None,
    ) -> list[tuple[str, float]]: ...
    def size(self) -> int: ...
    def clear(self) -> None: ...


class BruteForceIndex:
    """Exact cosine search over a dense matrix (the reference backend)."""

    def __init__(self, dim: int):
        self._dim = dim
        self._ids: list[str] = []
        self._pos: dict[str, int] = {}
        self._matrix = np.empty((0, dim), dtype=np.float32)

    def add(self, donor_id: str, vector: np.ndarray) -> None:
        if donor_id in self._pos:
            self._matrix[self._pos[donor_id]] = _unit(vector)
            return
        self._pos[donor_id] = len(self._ids)
        self._ids.append(donor_id)
        self._matrix = np.vstack([self._matrix, _unit(vector)[None, :]])

    def remove(self, donor_id: str) -> None:
        idx = self._pos.pop(donor_id, None)
        if idx is None:
            return
        self._ids.pop(idx)
        self._matrix = np.delete(self._matrix, idx, axis=0)
        for i in range(idx, len(self._ids)):
            self._pos[self._ids[i]] = i

    def search(
        self,
        query: np.ndarray,
        top_k: int,
        allowed: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        if not self._ids:
            return []
        sims = self._matrix @ _unit(query)
        if allowed is not None:
            mask = np.fromiter(
                (d in allowed for d in self._ids), dtype=bool, count=len(self._ids)
            )
            if not mask.any():
                return []
            idxs = np.where(mask)[0]
            order = idxs[np.argsort(-sims[idxs])][:top_k]
        else:
            order = np.argsort(-sims)[:top_k]
        return [(self._ids[i], float(sims[i])) for i in order]

    def size(self) -> int:
        return len(self._ids)

    def clear(self) -> None:
        self._ids = []
        self._pos = {}
        self._matrix = np.empty((0, self._dim), dtype=np.float32)


class DeltaGraphIndex:
    """Batch-built graph backend with a brute-forced delta buffer.

    ``graph_factory`` builds the immutable graph from (ids, matrix) and
    returns a callable ``search(query, top_k) -> list[(id, score)]``; the
    real GPU graph and the test fake both fit this shape. Additions land
    in the delta (exact) until ``rebuild_threshold`` triggers a rebuild;
    removals tombstone until folded at the next rebuild. Search re-ranks
    the union of graph and delta results by exact cosine.
    """

    def __init__(
        self,
        dim: int,
        graph_factory: Callable[[list[str], np.ndarray], Callable],
        rebuild_threshold: int = 1024,
        recall_sample_every: int = 100,
    ):
        self._dim = dim
        self._graph_factory = graph_factory
        self._rebuild_threshold = rebuild_threshold
        # Sampled recall audit: every Nth search also runs an exact pass
        # and records top-k overlap, so an approximation regression is
        # observable in production counters instead of silent. 0 = off.
        self._recall_sample_every = recall_sample_every
        self._search_count = 0
        self.recall_samples = 0
        self.recall_sum = 0.0
        self._delta = BruteForceIndex(dim)
        self._built_ids: list[str] = []
        self._built_matrix = np.empty((0, dim), dtype=np.float32)
        self._built_vec: dict[str, np.ndarray] = {}
        self._graph_search: Callable | None = None
        self._tombstones: set[str] = set()

    def add(self, donor_id: str, vector: np.ndarray) -> None:
        self._tombstones.discard(donor_id)
        self._delta.add(donor_id, vector)
        if self._delta.size() >= self._rebuild_threshold:
            self.rebuild()

    def remove(self, donor_id: str) -> None:
        self._delta.remove(donor_id)
        if donor_id in self._built_vec:
            self._tombstones.add(donor_id)

    def search(
        self,
        query: np.ndarray,
        top_k: int,
        allowed: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        merged: dict[str, float] = {}
        if self._graph_search is not None:
            # overfetch so post-filtering (tombstones, allowed) still
            # leaves top_k live graph candidates in the common case
            fetch = top_k * (4 if allowed is not None else 1) + len(self._tombstones)
            for donor_id, _score in self._graph_search(_unit(query), fetch):
                if donor_id in self._tombstones or donor_id not in self._built_vec:
                    continue
                if allowed is not None and donor_id not in allowed:
                    continue
                # re-rank by exact cosine; graph scores are approximate
                merged[donor_id] = float(self._built_vec[donor_id] @ _unit(query))
        for donor_id, score in self._delta.search(query, top_k, allowed=allowed):
            merged[donor_id] = max(score, merged.get(donor_id, -2.0))
        ranked = sorted(merged.items(), key=lambda kv: -kv[1])[:top_k]
        self._search_count += 1
        if (
            self._recall_sample_every
            and self._search_count % self._recall_sample_every == 0
        ):
            self._audit_recall(query, top_k, allowed, ranked)
        return ranked

    def _audit_recall(
        self,
        query: np.ndarray,
        top_k: int,
        allowed: set[str] | None,
        ranked: list[tuple[str, float]],
    ) -> None:
        exact = BruteForceIndex(self._dim)
        for donor_id, vec in self._built_vec.items():
            if donor_id not in self._tombstones:
                exact.add(donor_id, vec)
        for donor_id in list(self._delta._pos):
            exact.add(donor_id, self._delta._matrix[self._delta._pos[donor_id]])
        truth = {d for d, _ in exact.search(query, top_k, allowed=allowed)}
        if not truth:
            return
        got = {d for d, _ in ranked}
        self.recall_samples += 1
        self.recall_sum += len(truth & got) / len(truth)

    @property
    def observed_recall(self) -> float | None:
        return self.recall_sum / self.recall_samples if self.recall_samples else None

    def size(self) -> int:
        live_built = len(self._built_ids) - len(
            self._tombstones & set(self._built_ids)
        )
        return live_built + self._delta.size()

    def clear(self) -> None:
        self._delta.clear()
        self._built_ids = []
        self._built_matrix = np.empty((0, self._dim), dtype=np.float32)
        self._built_vec = {}
        self._graph_search = None
        self._tombstones = set()

    def rebuild(self) -> None:
        """Fold delta and tombstones into a freshly built graph."""
        keep: dict[str, np.ndarray] = {
            d: v for d, v in self._built_vec.items() if d not in self._tombstones
        }
        for donor_id in list(self._delta._pos):
            keep[donor_id] = self._delta._matrix[self._delta._pos[donor_id]]
        self._built_ids = list(keep)
        self._built_matrix = (
            np.stack([keep[d] for d in self._built_ids])
            if keep
            else np.empty((0, self._dim), dtype=np.float32)
        )
        self._built_vec = keep
        self._graph_search = (
            self._graph_factory(self._built_ids, self._built_matrix)
            if self._built_ids
            else None
        )
        self._delta.clear()
        self._tombstones = set()


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v

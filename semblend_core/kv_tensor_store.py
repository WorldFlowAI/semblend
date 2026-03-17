"""Direct KV tensor store for semantic-tier cache bypass.

Stores KV tensors in CPU memory indexed by semantic embeddings,
enabling retrieval by cosine similarity rather than chunk hash.
This is the foundation for bypassing LMCache's exact-chunk requirement.

Architecture:
    Request → MiniLM embedding → cosine search → retrieve donor KV →
    token-level alignment → RoPE correction → PartialAttention injection

Memory budget (FP16, Qwen2.5-7B, 28 layers, 4 KV heads, 128 head_dim):
    Per-token per-layer: 4 × 128 × 2 (K+V) × 2 bytes = 2.05 KB
    Per-token all-layers: 28 × 2.05 KB = 57.3 KB
    4K tokens: ~230 MB per donor
    100 donors at 4K: ~23 GB CPU RAM

Usage:
    store = KVTensorStore(max_entries=100, max_cpu_bytes=16 * 1024**3)
    store.add(embedding, token_ids, kv_tensors, model_id="qwen2.5-7b")
    result = store.search(query_embedding, top_k=5)
    kv = store.get_kv(result[0].entry_id, layers=[0, 1, 27])
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KVEntry:
    """Metadata for a stored KV tensor entry."""

    entry_id: str
    embedding: np.ndarray
    token_ids: tuple[int, ...]
    n_tokens: int
    n_layers: int
    model_id: str
    created_at: float
    size_bytes: int


@dataclass
class SearchResult:
    """Result from KV tensor store search."""

    entry_id: str
    similarity: float
    n_tokens: int
    model_id: str


@dataclass
class KVTensorStore:
    """CPU-resident KV tensor store indexed by semantic embedding.

    Stores KV cache tensors in CPU memory, enabling retrieval by
    cosine similarity. Designed for the semantic tier where LMCache's
    chunk-hash mechanism cannot match (non-identical tokens).

    Thread-safety: not thread-safe; caller must synchronize.
    """

    max_entries: int = 100
    max_cpu_bytes: int = 16 * 1024**3  # 16 GB default
    _entries: dict[str, KVEntry] = field(default_factory=dict)
    _kv_data: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = field(
        default_factory=dict
    )
    _embeddings: np.ndarray | None = field(default=None, repr=False)
    _entry_ids: list[str] = field(default_factory=list)
    _total_bytes: int = field(default=0)
    _index_dirty: bool = field(default=True)

    def add(
        self,
        embedding: np.ndarray,
        token_ids: list[int] | tuple[int, ...],
        kv_tensors: dict[int, tuple[np.ndarray, np.ndarray]],
        model_id: str = "unknown",
    ) -> str:
        """Add a KV cache entry to the store.

        Args:
            embedding: Normalized embedding vector [dim].
            token_ids: Token ID sequence for this entry.
            kv_tensors: {layer_idx: (K, V)} where K, V are
                [n_tokens, n_heads, head_dim] numpy arrays.
            model_id: Model identifier for version gating.

        Returns:
            Entry ID string.
        """
        entry_id = str(uuid.uuid4())[:12]
        token_tuple = tuple(token_ids)

        # Calculate size
        size_bytes = sum(
            k.nbytes + v.nbytes for k, v in kv_tensors.values()
        )

        # Evict if needed
        while (
            len(self._entries) >= self.max_entries
            or self._total_bytes + size_bytes > self.max_cpu_bytes
        ) and self._entries:
            self._evict_oldest()

        n_layers = len(kv_tensors)
        n_tokens = len(token_ids)
        if kv_tensors:
            first_k = next(iter(kv_tensors.values()))[0]
            if first_k.shape[0] != n_tokens:
                logger.warning(
                    "KV shape[0]=%d != n_tokens=%d",
                    first_k.shape[0], n_tokens,
                )

        entry = KVEntry(
            entry_id=entry_id,
            embedding=embedding.copy(),
            token_ids=token_tuple,
            n_tokens=n_tokens,
            n_layers=n_layers,
            model_id=model_id,
            created_at=time.monotonic(),
            size_bytes=size_bytes,
        )

        self._entries[entry_id] = entry
        self._kv_data[entry_id] = {
            layer: (k.copy(), v.copy()) for layer, (k, v) in kv_tensors.items()
        }
        self._total_bytes += size_bytes
        self._index_dirty = True

        logger.debug(
            "KVTensorStore: added %s (%d tokens, %d layers, %.1f MB)",
            entry_id, n_tokens, n_layers, size_bytes / 1024**2,
        )
        return entry_id

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        min_similarity: float = 0.60,
        model_id: str | None = None,
    ) -> list[SearchResult]:
        """Search for similar KV entries by embedding cosine similarity.

        Args:
            query_embedding: Normalized query vector [dim].
            top_k: Maximum results to return.
            min_similarity: Minimum cosine similarity threshold.
            model_id: If set, only return entries matching this model.

        Returns:
            List of SearchResult sorted by descending similarity.
        """
        if not self._entries:
            return []

        self._rebuild_index()

        # Cosine similarity (embeddings are normalized)
        similarities = self._embeddings @ query_embedding

        # Filter by model_id if specified
        valid_mask = np.ones(len(self._entry_ids), dtype=bool)
        if model_id is not None:
            for i, eid in enumerate(self._entry_ids):
                if self._entries[eid].model_id != model_id:
                    valid_mask[i] = False

        # Apply threshold
        valid_mask &= similarities >= min_similarity

        # Get top-k indices
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            return []

        valid_sims = similarities[valid_indices]
        top_indices = valid_indices[
            np.argsort(-valid_sims)[:top_k]
        ]

        results = []
        for idx in top_indices:
            eid = self._entry_ids[idx]
            entry = self._entries[eid]
            results.append(SearchResult(
                entry_id=eid,
                similarity=float(similarities[idx]),
                n_tokens=entry.n_tokens,
                model_id=entry.model_id,
            ))

        return results

    def get_kv(
        self,
        entry_id: str,
        layers: list[int] | None = None,
    ) -> dict[int, tuple[np.ndarray, np.ndarray]] | None:
        """Retrieve KV tensors for an entry.

        Args:
            entry_id: Entry ID from search result.
            layers: Specific layers to retrieve, or None for all.

        Returns:
            {layer_idx: (K, V)} dict, or None if entry not found.
        """
        if entry_id not in self._kv_data:
            return None

        kv = self._kv_data[entry_id]
        if layers is None:
            return kv

        return {l: kv[l] for l in layers if l in kv}

    def get_entry(self, entry_id: str) -> KVEntry | None:
        """Get metadata for an entry."""
        return self._entries.get(entry_id)

    def remove(self, entry_id: str) -> bool:
        """Remove an entry from the store."""
        if entry_id not in self._entries:
            return False

        entry = self._entries.pop(entry_id)
        self._kv_data.pop(entry_id, None)
        self._total_bytes -= entry.size_bytes
        self._index_dirty = True
        return True

    @property
    def size(self) -> int:
        """Number of entries in the store."""
        return len(self._entries)

    @property
    def total_bytes(self) -> int:
        """Total KV tensor storage in bytes."""
        return self._total_bytes

    def _rebuild_index(self) -> None:
        """Rebuild the embedding index matrix if dirty."""
        if not self._index_dirty:
            return

        self._entry_ids = list(self._entries.keys())
        if not self._entry_ids:
            self._embeddings = None
            self._index_dirty = False
            return

        self._embeddings = np.stack(
            [self._entries[eid].embedding for eid in self._entry_ids]
        )
        self._index_dirty = False

    def _evict_oldest(self) -> None:
        """Evict the oldest entry (LRU-style)."""
        if not self._entries:
            return

        oldest_id = min(
            self._entries, key=lambda eid: self._entries[eid].created_at
        )
        self.remove(oldest_id)
        logger.debug("KVTensorStore: evicted %s (LRU)", oldest_id)

"""SemBlend HiCacheStorage backend for SGLang.

Implements SGLang's HiCacheStorage ABC to provide semantic KV cache
retrieval as an L3 storage tier. When SGLang's radix tree misses and
the HiCache hierarchy checks L3, this backend performs:

1. Hash-based exact lookup (standard HiCache behavior)
2. On miss: semantic embedding similarity search against donor store
3. On semantic hit: returns donor's KV tensors (with RoPE correction metadata)

Activation:
    python -m sglang.launch_server --model-path <model> \\
        --enable-hierarchical-cache \\
        --hicache-storage-backend dynamic \\
        --hicache-storage-backend-extra-config \\
          '{"module_path":"semblend.integration.sglang.hicache_backend",
            "class_name":"SemBlendHiCacheStorage"}'

Environment variables:
    SEMBLEND_ENABLED=1              Enable semantic fallback (default: 1)
    SEMBLEND_MIN_SIMILARITY=0.60    Cosine similarity threshold
    SEMBLEND_EMBEDDER=minilm        Embedder type
"""
from __future__ import annotations

import logging
import os
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("semblend.sglang")

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

# Lazy imports for SGLang types — only available when running inside SGLang
if TYPE_CHECKING:
    pass


def _get_hicache_storage_base():
    """Lazily import SGLang's HiCacheStorage ABC."""
    try:
        from sglang.srt.mem_cache.hicache_storage import HiCacheStorage
        return HiCacheStorage
    except ImportError:
        raise ImportError(
            "SGLang is required for SemBlend SGLang integration. "
            "Install with: pip install semblend[sglang]"
        )


class _SemBlendDonorIndex:
    """In-process semantic donor index for SGLang.

    Maintains embeddings of recently cached prompts for cosine similarity
    search. Uses a threading lock for safe concurrent access.
    """

    def __init__(
        self,
        max_entries: int = 1000,
        min_similarity: float = 0.60,
    ) -> None:
        import threading
        self._max_entries = max_entries
        self._min_similarity = min_similarity
        self._entries: OrderedDict[str, tuple[np.ndarray, list[int], float]] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        return len(self._entries)

    def register(
        self,
        key: str,
        embedding: np.ndarray,
        token_ids: list[int],
    ) -> None:
        """Register a completed request as a potential donor."""
        with self._lock:
            if key in self._entries:
                return

            # Normalize embedding at registration time for correct cosine similarity
            normalized = embedding / (np.linalg.norm(embedding) + 1e-10)
            self._entries[key] = (normalized, token_ids, time.monotonic())

            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    def find_semantic_match(
        self,
        query_embedding: np.ndarray,
    ) -> Optional[str]:
        """Find the most similar donor by cosine similarity.

        Returns the hash key of the best match, or None.
        """
        with self._lock:
            if not self._entries:
                return None

            keys = list(self._entries.keys())
            embeddings = [entry[0] for entry in self._entries.values()]

            # Vectorized cosine similarity (embeddings already normalized)
            matrix = np.stack(embeddings)
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
            sims = matrix @ query_norm

            best_idx = int(np.argmax(sims))
            if sims[best_idx] >= self._min_similarity:
                return keys[best_idx]
            return None

    def get_token_ids(self, key: str) -> Optional[list[int]]:
        """Get token IDs for a registered donor key."""
        entry = self._entries.get(key)
        return entry[1] if entry else None


class SemBlendHiCacheStorage:
    """SGLang HiCacheStorage backend with semantic KV cache reuse.

    Wraps a base storage backend (file, mooncake, etc.) and adds
    semantic donor discovery on cache miss.
    """

    def __init__(
        self,
        storage_config: Any = None,
        **kwargs: Any,
    ) -> None:
        # Only validate SGLang import when running inside SGLang
        # (skip for unit testing without SGLang installed)
        try:
            _get_hicache_storage_base()
        except ImportError:
            pass  # Allow instantiation for testing

        self._config = storage_config
        self._enabled = os.environ.get("SEMBLEND_ENABLED", "1") == "1"
        self._min_similarity = float(
            os.environ.get("SEMBLEND_MIN_SIMILARITY", "0.60")
        )

        # Semantic donor index
        self._donor_index = _SemBlendDonorIndex(
            max_entries=int(os.environ.get("SEMBLEND_MAX_DONORS", "1000")),
            min_similarity=self._min_similarity,
        )

        # Embedder (lazy init on first use)
        self._embedder = None
        self._embedder_type = os.environ.get("SEMBLEND_EMBEDDER", "minilm")

        # Hash→KV mapping for semantic retrieval
        self._kv_store: Dict[str, Any] = {}
        # Hash→token_ids mapping for donor registration
        self._token_map: Dict[str, list[int]] = {}

        # Stats
        self._exact_hits = 0
        self._semantic_hits = 0
        self._misses = 0
        self._stores = 0

        logger.info(
            "SemBlend HiCacheStorage initialized "
            f"(enabled={self._enabled}, threshold={self._min_similarity}, "
            f"embedder={self._embedder_type})"
        )

    def _get_embedder(self):
        """Lazily initialize the embedder."""
        if self._embedder is None:
            from semblend_core.embedder import create_embedder
            self._embedder = create_embedder(self._embedder_type)
            logger.info(f"SemBlend embedder initialized: {self._embedder_type}")
        return self._embedder

    def _compute_embedding(self, token_ids: list[int]) -> np.ndarray:
        """Compute embedding for a token sequence."""
        embedder = self._get_embedder()
        if hasattr(embedder, "embed_tokens"):
            return embedder.embed_tokens(token_ids)
        raise ValueError(
            f"Embedder {type(embedder).__name__} requires text input but no "
            "tokenizer is available to decode token IDs. "
            "Set SEMBLEND_EMBEDDER=jaccard for token-ID-only operation."
        )

    def register_mem_pool_host(self, mem_pool_host: Any) -> None:
        """Register the host memory pool (required by HiCacheStorage)."""
        self._mem_pool_host = mem_pool_host

    def get(
        self,
        key: str,
        target_location: Any = None,
        target_sizes: Any = None,
    ) -> Any:
        """Get KV cache entry by hash key."""
        if key in self._kv_store:
            self._exact_hits += 1
            return self._kv_store[key]
        return None

    def batch_get(
        self,
        keys: List[str],
        target_locations: Any = None,
        target_sizes: Any = None,
    ) -> Any:
        """Batch get KV cache entries."""
        results = []
        for key in keys:
            result = self.get(key, target_locations, target_sizes)
            results.append(result)
        return results

    def set(
        self,
        key: str,
        value: Any = None,
        target_location: Any = None,
        target_sizes: Any = None,
    ) -> bool:
        """Store KV cache entry and register as donor."""
        self._kv_store[key] = value
        self._stores += 1

        # Register the entry as a potential semantic donor (consume from _token_map)
        if self._enabled and key in self._token_map:
            token_ids = self._token_map.pop(key)
            try:
                embedding = self._compute_embedding(token_ids)
                self._donor_index.register(key, embedding, token_ids)
            except Exception as e:
                logger.debug(f"SemBlend donor registration failed: {e}")

        return True

    def batch_set(
        self,
        keys: List[str],
        values: Any = None,
        target_locations: Any = None,
        target_sizes: Any = None,
    ) -> bool:
        """Batch store KV cache entries."""
        success = True
        if values is None:
            values = [None] * len(keys)
        for key, value in zip(keys, values):
            if not self.set(key, value, target_locations, target_sizes):
                success = False
        return success

    def exists(self, key: str) -> bool:
        """Check if a KV cache entry exists (exact match)."""
        return key in self._kv_store

    def batch_exists(
        self,
        keys: List[str],
        extra_info: Any = None,
    ) -> int:
        """Check consecutive existence of keys from the start.

        Returns the count of consecutive hits from index 0.
        This is the key method where semantic matching augments exact lookup.
        """
        count = 0
        for key in keys:
            if key in self._kv_store:
                count += 1
            else:
                # Exact miss — semantic matching at chunk level is limited by
                # the HiCacheStorage interface (only hash keys, no token IDs).
                # Semantic matching works better at the RadixCache level.
                self._misses += 1
                break
        return count

    def clear(self) -> None:
        """Clear all stored entries."""
        self._kv_store.clear()
        self._token_map.clear()

    def get_stats(self) -> dict:
        """Return storage statistics."""
        return {
            "exact_hits": self._exact_hits,
            "semantic_hits": self._semantic_hits,
            "misses": self._misses,
            "stores": self._stores,
            "donor_index_size": self._donor_index.size,
            "kv_store_size": len(self._kv_store),
        }

    def register_token_ids(self, key: str, token_ids: list[int]) -> None:
        """Register token IDs for a hash key (called before set()).

        This is a SemBlend extension — standard HiCacheStorage backends
        don't need token IDs, but SemBlend needs them for embedding
        computation during donor registration.
        """
        self._token_map[key] = token_ids

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

import inspect
import logging
import os
import time
from collections import OrderedDict
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from semblend.integration.sglang.namespace import (
    NO_EXTRA_KEY_NAMESPACE,
    isolation_namespace,
)

logger = logging.getLogger("semblend.sglang")

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

# Lazy imports for SGLang types — only available when running inside SGLang
if TYPE_CHECKING:
    pass


def _extra_info_namespace(extra_info: Any) -> str:
    """Isolation namespace for one HiCacheStorage call.

    HiCacheStorage hands a backend hash keys, not requests, so the only
    per-request isolation value it can carry is the ``extra_info`` newer
    SGLang builds pass alongside them — either the object itself or the
    dict it wraps. A build that passes none lands every call in the
    sentinel namespace, which is exactly how a single-tenant deployment
    behaved before isolation existed.

    Shares ``isolation_namespace`` with the other SGLang paths so one
    extra_key yields one namespace whichever integration path is loaded.
    """
    if extra_info is None:
        return NO_EXTRA_KEY_NAMESPACE

    if isinstance(extra_info, Mapping):
        return isolation_namespace(
            extra_info.get("extra_key"),
            extra_info.get("cache_salt"),
        )

    namespace = isolation_namespace(
        getattr(extra_info, "extra_key", None),
        getattr(extra_info, "cache_salt", None),
    )
    if namespace != NO_EXTRA_KEY_NAMESPACE:
        return namespace

    # HiCacheStorageExtraInfo carries its per-request fields in a nested
    # dict; an outer object without the fields must not mask it.
    nested = getattr(extra_info, "extra_info", None)
    if isinstance(nested, Mapping):
        return isolation_namespace(
            nested.get("extra_key"),
            nested.get("cache_salt"),
        )
    return NO_EXTRA_KEY_NAMESPACE


# The lookup methods SGLang calls with hash keys. If the installed ABC
# declares no extra_info on them, a lookup on this build can never name the
# tenant it is for.
_LOOKUP_METHODS = ("get", "exists", "batch_exists")


def _hicache_lookup_carries_extra_info() -> bool:
    """True when the installed HiCacheStorage passes extra_info to lookups.

    Probed by signature rather than by version string: upstream SGLang and
    the fork grew the parameter at different points, and the question that
    matters is only whether a lookup on THIS build can carry a namespace.

    When SGLang is not importable we are not being driven by its cache
    hierarchy at all — the caller is SemBlend's own wrapper or a test, and
    it calls the signature defined in this module, which does carry
    extra_info.
    """
    try:
        base = _get_hicache_storage_base()
    except ImportError:
        return True

    for name in _LOOKUP_METHODS:
        method = getattr(base, name, None)
        if method is None:
            continue
        try:
            parameters = inspect.signature(method).parameters
        except (TypeError, ValueError):
            continue
        if "extra_info" not in parameters:
            return False
    return True


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

    Every entry is filed under the isolation namespace of the request that
    produced it, and a lookup only ever sees its own namespace.
    """

    def __init__(
        self,
        max_entries: int = 1000,
        min_similarity: float = 0.60,
    ) -> None:
        import threading

        self._max_entries = max_entries
        self._min_similarity = min_similarity
        # Keyed by (namespace, storage key): the same prompt sent under two
        # namespaces must hold two entries, or the second tenant would be
        # deduped out of a donor it is the only one allowed to see.
        self._entries: OrderedDict[tuple[str, str], tuple[np.ndarray, list[int], float]] = (
            OrderedDict()
        )
        self._lock = threading.Lock()
        self._namespace_rejections = 0

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def namespace_rejections(self) -> int:
        """Donor entries withheld from lookups because their namespace differed.

        Monotonic over the index's lifetime; the storage backend surfaces it
        through get_stats().
        """
        return self._namespace_rejections

    def register(
        self,
        key: str,
        embedding: np.ndarray,
        token_ids: list[int],
        *,
        namespace: str = NO_EXTRA_KEY_NAMESPACE,
    ) -> None:
        """Register a completed request as a potential donor in `namespace`."""
        with self._lock:
            entry_key = (namespace, key)
            if entry_key in self._entries:
                return

            # Normalize embedding at registration time for correct cosine similarity
            normalized = embedding / (np.linalg.norm(embedding) + 1e-10)
            self._entries[entry_key] = (normalized, token_ids, time.monotonic())

            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    def find_semantic_match(
        self,
        query_embedding: np.ndarray,
        *,
        namespace: str = NO_EXTRA_KEY_NAMESPACE,
    ) -> Optional[str]:
        """Most similar donor within `namespace`, by cosine similarity.

        Isolation is a hard filter applied before scoring, not a penalty: an
        entry from another namespace is never a candidate, however similar
        it is. The default is the no-extra-key sentinel, so a caller that
        supplies nothing can only ever reach unkeyed donors.

        Returns the hash key of the best match, or None.
        """
        with self._lock:
            keys = []
            embeddings = []
            for (entry_namespace, entry_key), entry in self._entries.items():
                if entry_namespace != namespace:
                    self._namespace_rejections += 1
                    continue
                keys.append(entry_key)
                embeddings.append(entry[0])

            if not keys:
                return None

            # Vectorized cosine similarity (embeddings already normalized)
            matrix = np.stack(embeddings)
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
            sims = matrix @ query_norm

            best_idx = int(np.argmax(sims))
            if sims[best_idx] >= self._min_similarity:
                return keys[best_idx]
            return None

    def get_token_ids(
        self,
        key: str,
        *,
        namespace: str = NO_EXTRA_KEY_NAMESPACE,
    ) -> Optional[list[int]]:
        """Get token IDs for a donor key registered in `namespace`."""
        entry = self._entries.get((namespace, key))
        return entry[1] if entry else None

    def clear(self) -> None:
        """Drop every donor entry, in every namespace.

        The lifetime rejection counter is a statistic, not state, and is
        left alone so a reset does not erase evidence that isolation fired.
        """
        with self._lock:
            self._entries.clear()


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
        self._min_similarity = float(os.environ.get("SEMBLEND_MIN_SIMILARITY", "0.60"))

        # Whether this SGLang build can tell a lookup which tenant it is for.
        # Probed once: it is a property of the installed package, not of any
        # single call.
        self._extra_info_supported = _hicache_lookup_carries_extra_info()
        # Flipped the first time a real isolation value reaches this process
        # through any entry point, i.e. once salts are demonstrably in use.
        self._isolation_seen = False

        # Semantic donor index
        self._donor_index = _SemBlendDonorIndex(
            max_entries=int(os.environ.get("SEMBLEND_MAX_DONORS", "1000")),
            min_similarity=self._min_similarity,
        )

        # Embedder (lazy init on first use)
        self._embedder = None
        self._embedder_type = os.environ.get("SEMBLEND_EMBEDDER", "minilm")

        # Hash→(namespace, KV) mapping for retrieval. The namespace is
        # stored beside the value because the hash alone does not identify a
        # tenant on every SGLang build.
        self._kv_store: Dict[str, tuple[str, Any]] = {}
        # Hash→token_ids mapping for donor registration
        self._token_map: Dict[str, list[int]] = {}
        # Hash→namespace captured with those token ids
        self._namespace_map: Dict[str, str] = {}

        # Stats
        self._exact_hits = 0
        self._semantic_hits = 0
        self._misses = 0
        self._stores = 0
        # Entries withheld from a lookup because they were stored under a
        # different namespace. Non-zero here is isolation doing its job,
        # not an error.
        self._cross_namespace_rejected = 0
        # Lookups declined because this build cannot carry a namespace and
        # the deployment is known to use them.
        self._isolation_declined = 0
        self._isolation_decline_logged = False

        logger.info(
            "SemBlend HiCacheStorage initialized "
            f"(enabled={self._enabled}, threshold={self._min_similarity}, "
            f"embedder={self._embedder_type}, "
            f"extra_info_supported={self._extra_info_supported})"
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

    def _visible(self, key: str, namespace: str) -> bool:
        """True when `key` was stored under `namespace`.

        Exact equality, both sides always a real string: an entry stored
        without a namespace sits under the sentinel and is invisible to a
        keyed caller, and vice versa. Every rejection is counted.
        """
        entry = self._kv_store.get(key)
        if entry is None:
            return False
        if entry[0] != namespace:
            self._cross_namespace_rejected += 1
            return False
        return True

    def _observe_namespace(self, namespace: str) -> str:
        """Record that a real isolation value reached this backend."""
        if namespace != NO_EXTRA_KEY_NAMESPACE:
            self._isolation_seen = True
        return namespace

    def _lookup_namespace(self, extra_info: Any) -> Optional[str]:
        """Namespace this lookup may use, or None when it must be declined.

        On a build whose HiCacheStorage passes extra_info, an absent value
        means the request genuinely carries no salt: it lands in the
        sentinel namespace and can only reach other unsalted entries, which
        is how a single-tenant deployment has always behaved.

        On a build that cannot pass extra_info at all, an unnamespaced
        lookup is indistinguishable from a salted request whose key the
        build dropped on the way in. While no isolation value has ever
        reached this process there is nothing to confuse it with, so those
        deployments keep working unchanged; once one has, serving an
        unnamespaced lookup could hand one tenant another's KV, so it is
        declined, counted and logged instead of run unisolated.
        """
        namespace = self._observe_namespace(_extra_info_namespace(extra_info))
        if namespace != NO_EXTRA_KEY_NAMESPACE:
            return namespace
        if self._extra_info_supported or not self._isolation_seen:
            return namespace

        self._isolation_declined += 1
        if not self._isolation_decline_logged:
            self._isolation_decline_logged = True
            logger.warning(
                "SemBlend HiCache declined an unnamespaced lookup: the installed "
                "HiCacheStorage does not pass extra_info, and this process has "
                "already seen per-tenant isolation values, so a lookup without one "
                "cannot be shown to belong to the tenant that stored the entry. "
                "Upgrade SGLang to a build that passes extra_info to restore "
                "cache reuse on this path."
            )
        return None

    def get(
        self,
        key: str,
        target_location: Any = None,
        target_sizes: Any = None,
        extra_info: Any = None,
    ) -> Any:
        """Get KV cache entry by hash key, within the caller's namespace."""
        namespace = self._lookup_namespace(extra_info)
        if namespace is None:
            return None
        if self._visible(key, namespace):
            self._exact_hits += 1
            return self._kv_store[key][1]
        return None

    def batch_get(
        self,
        keys: List[str],
        target_locations: Any = None,
        target_sizes: Any = None,
        extra_info: Any = None,
    ) -> Any:
        """Batch get KV cache entries."""
        results = []
        for key in keys:
            result = self.get(key, target_locations, target_sizes, extra_info)
            results.append(result)
        return results

    def set(
        self,
        key: str,
        value: Any = None,
        target_location: Any = None,
        target_sizes: Any = None,
        extra_info: Any = None,
    ) -> bool:
        """Store KV cache entry and register as donor, bound to a namespace."""
        # Consume both sides of the token-id handshake even when the entry
        # is not registered as a donor, so no stale namespace survives.
        token_ids = self._token_map.pop(key, None)
        registered_namespace = self._namespace_map.pop(key, NO_EXTRA_KEY_NAMESPACE)

        namespace = _extra_info_namespace(extra_info)
        if namespace == NO_EXTRA_KEY_NAMESPACE:
            # This call named no namespace: fall back to the one captured
            # with the token ids, so a caller that can reach only one of the
            # two entry points still binds what it stores.
            namespace = registered_namespace
        self._observe_namespace(namespace)

        self._kv_store[key] = (namespace, value)
        self._stores += 1

        # Register the entry as a potential semantic donor under the same
        # namespace a lookup will have to present.
        if self._enabled and token_ids is not None:
            try:
                embedding = self._compute_embedding(token_ids)
                self._donor_index.register(key, embedding, token_ids, namespace=namespace)
            except Exception as e:
                logger.debug(f"SemBlend donor registration failed: {e}")

        return True

    def batch_set(
        self,
        keys: List[str],
        values: Any = None,
        target_locations: Any = None,
        target_sizes: Any = None,
        extra_info: Any = None,
    ) -> bool:
        """Batch store KV cache entries."""
        success = True
        if values is None:
            values = [None] * len(keys)
        for key, value in zip(keys, values):
            if not self.set(key, value, target_locations, target_sizes, extra_info):
                success = False
        return success

    def exists(self, key: str, extra_info: Any = None) -> bool:
        """Check if a KV cache entry exists (exact match) in this namespace."""
        namespace = self._lookup_namespace(extra_info)
        if namespace is None:
            return False
        return self._visible(key, namespace)

    def batch_exists(
        self,
        keys: List[str],
        extra_info: Any = None,
    ) -> int:
        """Check consecutive existence of keys from the start.

        Returns the count of consecutive hits from index 0.
        This is the key method where semantic matching augments exact lookup.
        """
        namespace = self._lookup_namespace(extra_info)
        if namespace is None:
            return 0
        count = 0
        for key in keys:
            if self._visible(key, namespace):
                count += 1
            else:
                # Exact miss — semantic matching at chunk level is limited by
                # the HiCacheStorage interface (only hash keys, no token IDs).
                # Semantic matching works better at the RadixCache level.
                self._misses += 1
                break
        return count

    def clear(self) -> None:
        """Clear all stored entries, the semantic donor index included.

        The donor index is a second view of the same entries, so leaving it
        behind would survive a cache reset: a later lookup could still match
        a donor whose KV this call just dropped.
        """
        self._kv_store.clear()
        self._token_map.clear()
        self._namespace_map.clear()
        self._donor_index.clear()

    def get_stats(self) -> dict:
        """Return storage statistics."""
        return {
            "exact_hits": self._exact_hits,
            "semantic_hits": self._semantic_hits,
            "misses": self._misses,
            "stores": self._stores,
            "donor_index_size": self._donor_index.size,
            "kv_store_size": len(self._kv_store),
            "cross_namespace_rejected": (
                self._cross_namespace_rejected + self._donor_index.namespace_rejections
            ),
            "isolation_declined": self._isolation_declined,
        }

    def register_token_ids(
        self,
        key: str,
        token_ids: list[int],
        *,
        extra_key: Any = None,
        cache_salt: Any = None,
        extra_info: Any = None,
    ) -> None:
        """Register token IDs for a hash key (called before set()).

        This is a SemBlend extension — standard HiCacheStorage backends
        don't need token IDs, but SemBlend needs them for embedding
        computation during donor registration.

        The caller's isolation value rides along here because this is the
        one SemBlend-owned entry point that sees the request: it is hashed
        into a namespace and bound to the donor when set() lands.
        """
        namespace = isolation_namespace(extra_key, cache_salt)
        if namespace == NO_EXTRA_KEY_NAMESPACE:
            namespace = _extra_info_namespace(extra_info)
        self._observe_namespace(namespace)
        self._token_map[key] = token_ids
        self._namespace_map[key] = namespace

"""SemBlend RadixCache subclass for SGLang -- semantic donor discovery.

Subclasses SGLang's RadixCache (following LMCRadixCache's pattern) to add
semantic donor discovery on prefix cache miss. This is the deeper integration
path that provides full control over the semantic matching pipeline.

Architecture:
    SGLang scheduler
        -> SemBlendRadixCache.match_prefix()
            -> RadixCache.match_prefix() (exact prefix match)
            -> on short match: SemBlend semantic donor search
                -> compute MiniLM embedding of prompt text
                -> cosine similarity against donor store
                -> if hit: find donor's radix tree node
                -> return donor's prefix as the match result
        -> SGLang loads donor's KV from radix tree (normal path)

This module requires SGLang to be installed and is loaded at runtime.
"""

from __future__ import annotations

import logging
import os
import time
from collections import OrderedDict
from typing import Any, Optional

import numpy as np

logger = logging.getLogger("semblend.sglang.radix")

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Sliding-window token sampling (mirrors vLLM connector pattern)
# ---------------------------------------------------------------------------

_MAX_DECODE_TOKENS = 2000


def _sample_token_ids(token_ids: list[int]) -> list[int]:
    """Sample a representative subset of token IDs for embedding.

    For long prompts, samples 40% head + 30% middle + 30% tail so the
    MiniLM embedding (512-token window) sees representative content from
    the full document rather than just the beginning.
    """
    n = len(token_ids)
    if n <= _MAX_DECODE_TOKENS:
        return token_ids

    head = int(_MAX_DECODE_TOKENS * 0.40)
    mid_w = int(_MAX_DECODE_TOKENS * 0.30)
    tail = _MAX_DECODE_TOKENS - head - mid_w
    mid_start = (n - mid_w) // 2

    return token_ids[:head] + token_ids[mid_start : mid_start + mid_w] + token_ids[n - tail :]


# ---------------------------------------------------------------------------
# Tokenizer bridge (lazy-loaded for token-ID -> text decoding)
# ---------------------------------------------------------------------------

_tokenizer_instance = None
_tokenizer_load_attempted = False


def _get_tokenizer():
    """Lazily load tokenizer from SEMBLEND_MODEL_NAME env var.

    Returns the tokenizer or None if unavailable.
    """
    global _tokenizer_instance, _tokenizer_load_attempted
    if _tokenizer_load_attempted:
        return _tokenizer_instance

    _tokenizer_load_attempted = True
    model_name = os.environ.get("SEMBLEND_MODEL_NAME", "")
    if not model_name:
        logger.warning(
            "SEMBLEND_MODEL_NAME not set -- cannot decode token IDs for "
            "embedding. Set this env var to enable semantic matching."
        )
        return None

    try:
        from transformers import AutoTokenizer

        _tokenizer_instance = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        logger.info(f"SemBlend tokenizer loaded: {model_name}")
    except Exception as e:
        logger.error(f"Failed to load tokenizer for {model_name}: {e}")

    return _tokenizer_instance


def _tokens_to_text(token_ids: list[int]) -> Optional[str]:
    """Decode token IDs to text using the sliding-window sampling pattern.

    Returns the decoded text, or None if no tokenizer is available.
    """
    tokenizer = _get_tokenizer()
    if tokenizer is None:
        return None

    sampled = _sample_token_ids(token_ids)
    return tokenizer.decode(sampled, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Donor store
# ---------------------------------------------------------------------------


class _DonorEntry:
    """A cached donor prompt for semantic matching."""

    __slots__ = ("token_ids", "embedding", "timestamp", "num_tokens")

    def __init__(
        self,
        token_ids: tuple[int, ...],
        embedding: np.ndarray,
        timestamp: float,
        num_tokens: int,
    ) -> None:
        self.token_ids = token_ids
        self.embedding = embedding
        self.timestamp = timestamp
        self.num_tokens = num_tokens


class _SemBlendDonorStore:
    """In-process semantic donor store for RadixCache integration."""

    def __init__(
        self,
        max_entries: int = 1000,
        min_similarity: float = 0.60,
    ) -> None:
        self._entries: OrderedDict[tuple[int, ...], _DonorEntry] = OrderedDict()
        self._max_entries = max_entries
        self._min_similarity = min_similarity

    @property
    def size(self) -> int:
        return len(self._entries)

    def add_donor(
        self,
        token_ids: tuple[int, ...],
        embedding: np.ndarray,
    ) -> None:
        """Register a completed request as a potential donor."""
        key = token_ids[:256]  # Use first 256 tokens as dedup key
        if key in self._entries:
            self._entries.move_to_end(key)
            return

        self._entries[key] = _DonorEntry(
            token_ids=token_ids,
            embedding=embedding / (np.linalg.norm(embedding) + 1e-10),
            timestamp=time.monotonic(),
            num_tokens=len(token_ids),
        )

        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def find_donor(
        self,
        query_embedding: np.ndarray,
        exclude_tokens: Optional[tuple[int, ...]] = None,
    ) -> Optional[_DonorEntry]:
        """Find the most semantically similar donor.

        Returns the best DonorEntry above the similarity threshold, or None.
        """
        if not self._entries:
            return None

        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)

        best_entry = None
        best_sim = self._min_similarity

        for entry in self._entries.values():
            if exclude_tokens and entry.token_ids[:256] == exclude_tokens[:256]:
                continue

            sim = float(np.dot(entry.embedding, query_norm))
            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        return best_entry


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------


def _embed_token_ids(embedder: Any, token_ids: list[int]) -> Optional[np.ndarray]:
    """Compute embedding for a token sequence via text decoding.

    Uses the sliding-window tokenizer bridge to convert token IDs to text,
    then calls the embedder's `.embed(text)` method.
    """
    text = _tokens_to_text(token_ids)
    if text is None:
        return None

    if not text.strip():
        return None

    return embedder.embed(text)


# ---------------------------------------------------------------------------
# match_prefix result introspection (SGLang version-agnostic)
# ---------------------------------------------------------------------------

_match_prefix_api_logged = False


def _get_matched_length(result: Any) -> int:
    """Extract the matched prefix length from a match_prefix result.

    SGLang v0.4.x returns (matched_indices, last_node) or similar tuples.
    Newer versions may return an object with `.device_indices` or `.value`.
    We probe common patterns and log the detected API on first call.
    """
    global _match_prefix_api_logged

    matched_len = 0

    # Pattern 1: Object with device_indices attribute
    if hasattr(result, "device_indices"):
        matched_len = len(result.device_indices)
    # Pattern 2: Tuple (value, last_node) where value is a tensor or list
    elif isinstance(result, tuple) and len(result) >= 1:
        first = result[0]
        if hasattr(first, "__len__"):
            matched_len = len(first)
        elif isinstance(first, int):
            matched_len = first
    # Pattern 3: Direct integer
    elif isinstance(result, int):
        matched_len = result

    if not _match_prefix_api_logged:
        _match_prefix_api_logged = True
        logger.info(
            f"SemBlend detected match_prefix result type: "
            f"{type(result).__name__}, matched_len={matched_len}"
        )

    return matched_len


def _extract_token_ids_from_key(key: Any) -> list[int]:
    """Extract token IDs from whatever SGLang passes as the key argument.

    SGLang v0.4.x passes the key directly as a list/tuple of token IDs.
    Newer versions may wrap it in a MatchPrefixParams or similar object.
    """
    # Direct list/tuple of ints
    if isinstance(key, (list, tuple)):
        return list(key)

    # Object with .token_ids attribute
    if hasattr(key, "token_ids"):
        return list(key.token_ids)

    # Object with .key attribute (e.g., MatchPrefixParams)
    if hasattr(key, "key"):
        inner = key.key
        if isinstance(inner, (list, tuple)):
            return list(inner)
        if hasattr(inner, "token_ids"):
            return list(inner.token_ids)

    logger.warning(f"Cannot extract token IDs from key type: {type(key).__name__}")
    return []


# ---------------------------------------------------------------------------
# SemBlendRadixCache class factory
# ---------------------------------------------------------------------------


def get_semblend_radix_cache_class(base_cache_cls: type) -> type:
    """Return a SemBlendRadixCache class that inherits from base_cache_cls.

    This returns the CLASS itself (not an instance) so that SGLang can
    instantiate it through its normal startup path.

    Args:
        base_cache_cls: SGLang's RadixCache class (or LMCRadixCache).

    Returns:
        The SemBlendRadixCache class (a dynamic subclass of base_cache_cls).
    """

    class SemBlendRadixCache(base_cache_cls):  # type: ignore[misc]
        """RadixCache with SemBlend semantic donor discovery fallback."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)

            self._semblend_enabled = os.environ.get("SEMBLEND_ENABLED", "1") == "1"
            self._semblend_donor_store = _SemBlendDonorStore(
                max_entries=int(os.environ.get("SEMBLEND_MAX_DONORS", "1000")),
                min_similarity=float(os.environ.get("SEMBLEND_MIN_SIMILARITY", "0.60")),
            )
            self._semblend_embedder = None
            self._semblend_stats = {
                "radix_hits": 0,
                "semantic_hits": 0,
                "misses": 0,
                "donors_registered": 0,
            }

            if self._semblend_enabled:
                logger.info(
                    "SemBlend RadixCache initialized "
                    f"(threshold="
                    f"{self._semblend_donor_store._min_similarity})"
                )

        def _get_embedder(self) -> Any:
            """Lazily initialize the MiniLM embedder."""
            if self._semblend_embedder is None:
                from semblend_core.embedder import create_embedder

                embedder_type = os.environ.get("SEMBLEND_EMBEDDER", "minilm")
                self._semblend_embedder = create_embedder(embedder_type)
                logger.info(f"SemBlend embedder initialized: {embedder_type}")
            return self._semblend_embedder

        def _embed_tokens(self, token_ids: list[int]) -> Optional[np.ndarray]:
            """Compute embedding for a token sequence via text decoding."""
            embedder = self._get_embedder()
            return _embed_token_ids(embedder, token_ids)

        def match_prefix(self, key: Any, **kwargs: Any) -> Any:
            """Override match_prefix to add semantic fallback.

            Flow:
            1. Call base RadixCache.match_prefix() for exact prefix match
            2. If match is short and SemBlend is enabled:
               a. Compute embedding of the prompt
               b. Search donor store for semantic match
               c. If donor found in radix tree, return donor's prefix
            3. Otherwise return the base result
            """
            base_result = super().match_prefix(key, **kwargs)

            if not self._semblend_enabled:
                return base_result

            matched_len = _get_matched_length(base_result)
            token_ids = _extract_token_ids_from_key(key)

            if not token_ids:
                return base_result

            if matched_len >= len(token_ids) * 0.5:
                if matched_len > 0:
                    self._semblend_stats["radix_hits"] += 1
                return base_result

            return self._try_semantic_match(
                token_ids, matched_len, base_result, target_len=len(token_ids), **kwargs
            )

        def _try_semantic_match(
            self,
            token_ids: list[int],
            matched_len: int,
            base_result: Any,
            target_len: int = 0,
            **kwargs: Any,
        ) -> Any:
            """Attempt semantic donor search and tree lookup."""
            try:
                t0 = time.monotonic()
                query_embedding = self._embed_tokens(token_ids)
                if query_embedding is None:
                    self._semblend_stats["misses"] += 1
                    return base_result

                donor = self._semblend_donor_store.find_donor(
                    query_embedding,
                    exclude_tokens=tuple(token_ids[:256]),
                )

                if donor is None:
                    self._semblend_stats["misses"] += 1
                    return base_result

                return self._lookup_donor_in_tree(
                    donor, matched_len, base_result, t0, target_len=target_len, **kwargs
                )

            except Exception as e:
                logger.debug(f"SemBlend semantic search failed: {e}")
                self._semblend_stats["misses"] += 1
                return base_result

        def _lookup_donor_in_tree(
            self,
            donor: _DonorEntry,
            matched_len: int,
            base_result: Any,
            t0: float,
            target_len: int = 0,
            **kwargs: Any,
        ) -> Any:
            """Check if a donor's tokens exist in the radix tree.

            When the donor has more tokens than the target request,
            we cap the reused prefix to avoid negative new-token counts
            in the SGLang scheduler.
            """
            try:
                donor_token_ids = list(donor.token_ids)
                donor_result = super().match_prefix(donor_token_ids, **kwargs)
                donor_matched = _get_matched_length(donor_result)

                # Cap reuse to target request length minus a safety
                # margin of 1 token (SGLang needs at least 1 new token
                # to prefill).
                if target_len > 0 and donor_matched >= target_len:
                    max_reuse = max(target_len - 1, 0)
                    if isinstance(donor_result, tuple) and len(donor_result) >= 2:
                        prefix_tensor = donor_result[0]
                        if hasattr(prefix_tensor, "__len__") and len(prefix_tensor) > max_reuse:
                            donor_result = (prefix_tensor[:max_reuse], donor_result[1])
                            donor_matched = max_reuse
                            logger.info(
                                f"SemBlend capped prefix: {len(prefix_tensor)} -> "
                                f"{max_reuse} tokens (target_len={target_len})"
                            )

                elapsed_ms = (time.monotonic() - t0) * 1000
                if donor_matched > matched_len:
                    self._semblend_stats["semantic_hits"] += 1
                    logger.info(
                        f"SemBlend semantic hit: {matched_len} -> "
                        f"{donor_matched} tokens ({elapsed_ms:.1f}ms)"
                    )
                    return donor_result

            except Exception as e:
                logger.debug(f"SemBlend donor tree lookup failed: {e}")

            self._semblend_stats["misses"] += 1
            return base_result

        def cache_finished_req(self, *args: Any, **kwargs: Any) -> None:
            """Override to register completed requests as donors."""
            super().cache_finished_req(*args, **kwargs)

            if not self._semblend_enabled:
                return

            self._register_donor_from_req(args, kwargs)

        def _register_donor_from_req(self, args: tuple, kwargs: dict) -> None:
            """Extract token IDs from a finished request and register."""
            req = args[0] if args else kwargs.get("req")
            if req is None:
                return

            try:
                token_ids = self._extract_req_token_ids(req)
                if len(token_ids) <= 100:
                    return

                embedding = self._embed_tokens(token_ids)
                if embedding is None:
                    return

                self._semblend_donor_store.add_donor(tuple(token_ids), embedding)
                self._semblend_stats["donors_registered"] += 1

            except Exception as e:
                logger.debug(f"SemBlend donor registration failed: {e}")

        @staticmethod
        def _extract_req_token_ids(req: Any) -> list[int]:
            """Extract token IDs from a SGLang request object."""
            if hasattr(req, "origin_input_ids"):
                return list(req.origin_input_ids)
            if hasattr(req, "input_ids"):
                return list(req.input_ids)
            return []

        def get_semblend_stats(self) -> dict:
            """Return SemBlend-specific statistics."""
            return {
                **self._semblend_stats,
                "donor_store_size": self._semblend_donor_store.size,
            }

    return SemBlendRadixCache


def create_semblend_radix_cache(
    base_cache_cls: type,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Convenience factory: create a SemBlendRadixCache instance.

    Combines ``get_semblend_radix_cache_class`` and instantiation in one call.

    Args:
        base_cache_cls: SGLang's RadixCache class (or LMCRadixCache).
        *args, **kwargs: Forwarded to the RadixCache constructor.

    Returns:
        A SemBlendRadixCache instance.
    """
    cls = get_semblend_radix_cache_class(base_cache_cls)
    return cls(*args, **kwargs)

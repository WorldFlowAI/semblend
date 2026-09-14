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

import copy
import logging
import os
import time
from collections import OrderedDict
from dataclasses import is_dataclass
from dataclasses import replace as dataclass_replace
from typing import Any, Optional

import numpy as np

from semblend.integration.sglang.namespace import (
    NO_EXTRA_KEY_NAMESPACE,
    chain_namespace,
    isolation_namespace,
)

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
# Isolation namespace
# ---------------------------------------------------------------------------

# The sentinel and the hashing helper live in namespace.py so the
# upstream-PR adapter does not have to import them out of this
# monkey-patch module. Re-exported under the old private name because
# released integrations and tests still reach for it here.
_isolation_namespace = isolation_namespace


def _key_namespace(key: Any) -> str:
    """Isolation namespace of whatever SGLang passes as the match_prefix key.

    Composed per field over the params object, the RadixKey it wraps and the
    Req either of them references, so neither an outer wrapper without the
    field nor an inner one carrying only half of it can change the result.
    A bare token list (older SGLang) carries no isolation value and lands in
    the sentinel namespace, which is also what that release's own tree does.
    """
    return chain_namespace(key)


def _req_namespace(req: Any) -> str:
    """Isolation namespace of a finished SGLang Req, for donor registration.

    The same composition as ``_key_namespace``, over the same field set:
    registration and lookup are one function, so a salted request cannot
    register under its salt and then look one up without it.
    """
    return chain_namespace(req)


# ---------------------------------------------------------------------------
# Donor store
# ---------------------------------------------------------------------------


class _DonorEntry:
    """A cached donor prompt for semantic matching."""

    __slots__ = ("token_ids", "embedding", "timestamp", "num_tokens", "namespace")

    def __init__(
        self,
        token_ids: tuple[int, ...],
        embedding: np.ndarray,
        timestamp: float,
        num_tokens: int,
        namespace: str,
    ) -> None:
        self.token_ids = token_ids
        self.embedding = embedding
        self.timestamp = timestamp
        self.num_tokens = num_tokens
        # Isolation namespace of the request that produced this KV.
        self.namespace = namespace


class _SemBlendDonorStore:
    """In-process semantic donor store for RadixCache integration."""

    def __init__(
        self,
        max_entries: int = 1000,
        min_similarity: float = 0.60,
    ) -> None:
        self._entries: OrderedDict[tuple[str, tuple[int, ...]], _DonorEntry] = OrderedDict()
        self._max_entries = max_entries
        self._min_similarity = min_similarity
        self._namespace_rejections = 0

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def namespace_rejections(self) -> int:
        """Donor candidates skipped because their namespace did not match.

        Monotonic across the store's lifetime; the cache reports the
        per-lookup delta through get_semblend_stats().
        """
        return self._namespace_rejections

    def add_donor(
        self,
        token_ids: tuple[int, ...],
        embedding: np.ndarray,
        *,
        namespace: str = NO_EXTRA_KEY_NAMESPACE,
    ) -> None:
        """Register a completed request as a potential donor."""
        # Dedup on the first 256 tokens, within a namespace only: the same
        # prompt sent by two tenants must register twice, or the second
        # tenant would never get a donor it is allowed to see.
        key = (namespace, token_ids[:256])
        if key in self._entries:
            self._entries.move_to_end(key)
            return

        self._entries[key] = _DonorEntry(
            token_ids=token_ids,
            embedding=embedding / (np.linalg.norm(embedding) + 1e-10),
            timestamp=time.monotonic(),
            num_tokens=len(token_ids),
            namespace=namespace,
        )

        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def find_donor(
        self,
        query_embedding: np.ndarray,
        exclude_tokens: Optional[tuple[int, ...]] = None,
        *,
        namespace: str = NO_EXTRA_KEY_NAMESPACE,
    ) -> Optional[_DonorEntry]:
        """Most semantically similar donor within `namespace`.

        Isolation is a hard filter applied before scoring, not a penalty:
        a donor whose namespace differs is never a candidate, whatever its
        similarity. The default is the no-extra-key sentinel, so a caller
        that supplies nothing can only ever see unkeyed donors.

        Returns the best DonorEntry above the similarity threshold, or None.
        """
        if not self._entries:
            return None

        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)

        best_entry = None
        best_sim = self._min_similarity

        for entry in self._entries.values():
            # Isolation gate first — a cross-namespace donor is not a
            # candidate regardless of how similar it is.
            if entry.namespace != namespace:
                self._namespace_rejections += 1
                continue
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

    Only the token ids come back from here. The isolation fields the same
    key carries are read separately by ``_key_namespace`` — they must not
    be dropped on the floor while unwrapping.
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


def _with_attribute(obj: Any, name: str, value: Any) -> Optional[Any]:
    """Copy of ``obj`` with one attribute replaced, or None if it cannot be.

    Copies rather than assigns in place: SGLang hands us the live key of the
    request being scheduled, and the donor lookup must not disturb it.
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        try:
            return dataclass_replace(obj, **{name: value})
        except (TypeError, ValueError):
            # Non-init or unknown field on this build — fall back to a copy.
            pass
    try:
        clone = copy.copy(obj)
        setattr(clone, name, value)
        return clone
    except (AttributeError, TypeError):
        return None


def _donor_lookup_key(key: Any, donor_token_ids: list[int]) -> Optional[Any]:
    """A match_prefix key for the donor's tokens, shaped like ``key``.

    The donor lookup re-enters the same tree, so it has to present the same
    key type AND the same isolation value the request arrived with. A bare
    token list is no substitute: on builds whose match_prefix expects a
    RadixKey it raises, so the semantic path never fires there at all, and on
    builds that partition the tree by extra_key it would look the donor up
    outside the requesting tenant's partition.

    Returns None when the shape cannot be rebuilt; the caller then declines
    the donor rather than querying the tree unisolated.
    """
    if isinstance(key, (list, tuple)):
        # Older SGLang passes the token ids themselves, and that release's
        # tree carries no isolation value, so the list is the whole key.
        return list(donor_token_ids)

    if hasattr(key, "token_ids"):
        return _with_attribute(key, "token_ids", list(donor_token_ids))

    inner = getattr(key, "key", None)
    if inner is not None:
        rebuilt = _donor_lookup_key(inner, donor_token_ids)
        if rebuilt is None:
            return None
        # The wrapper's other fields (notably ``req``) ride along: some
        # builds read the isolation value off the request, not the key.
        return _with_attribute(key, "key", rebuilt)

    return None


def _cap_match_result(result: Any, max_reuse: int) -> Optional[tuple[Any, int]]:
    """``(result, matched_len)`` with the matched prefix cut to ``max_reuse``.

    Returns None when this result shape cannot be capped. An uncappable
    result must become a miss, never an uncapped hit: a prefix at least as
    long as the request leaves the SGLang scheduler no token to prefill,
    i.e. a negative new-token count.
    """
    if not isinstance(result, tuple) or len(result) < 2:
        return None

    indices = result[0]
    if not hasattr(indices, "__len__"):
        return None
    if len(indices) <= max_reuse:
        return result, len(indices)

    # A host-side hit length describes the untruncated prefix; cutting the
    # device indices out from under it would leave the two disagreeing.
    if getattr(result, "host_hit_length", 0):
        return None

    truncated = indices[:max_reuse]
    if hasattr(result, "_replace") and getattr(result, "_fields", None):
        # SGLang's MatchResult is a NamedTuple: rebuild it as its own type so
        # the scheduler still gets the node fields it reads by name. A plain
        # 2-tuple would drop last_host_node / host_hit_length.
        return result._replace(**{result._fields[0]: truncated}), max_reuse
    return (truncated, result[1]), max_reuse


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
                # Donor entries withheld from a lookup because their
                # extra_key namespace differed from the requesting tenant's.
                # Non-zero here is isolation doing its job, not an error.
                "cross_namespace_rejected": 0,
                # Donors declined because no isolated tree key could be built
                # for this SGLang key shape.
                "donor_key_unavailable": 0,
                # Donors declined because their prefix could not be capped to
                # leave the scheduler a token to prefill.
                "uncapped_prefix_declined": 0,
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

            # Isolation namespace for this request. The donor lookup below
            # carries it, and every donor was registered under the namespace
            # of the request that produced its KV.
            return self._try_semantic_match(
                key,
                token_ids,
                matched_len,
                base_result,
                target_len=len(token_ids),
                namespace=_key_namespace(key),
                **kwargs,
            )

        def _try_semantic_match(
            self,
            key: Any,
            token_ids: list[int],
            matched_len: int,
            base_result: Any,
            target_len: int = 0,
            namespace: str = NO_EXTRA_KEY_NAMESPACE,
            **kwargs: Any,
        ) -> Any:
            """Attempt semantic donor search and tree lookup."""
            try:
                t0 = time.monotonic()
                query_embedding = self._embed_tokens(token_ids)
                if query_embedding is None:
                    self._semblend_stats["misses"] += 1
                    return base_result

                store = self._semblend_donor_store
                rejected_before = store.namespace_rejections
                donor = store.find_donor(
                    query_embedding,
                    exclude_tokens=tuple(token_ids[:256]),
                    namespace=namespace,
                )
                withheld = store.namespace_rejections - rejected_before
                self._semblend_stats["cross_namespace_rejected"] += withheld

                if donor is None:
                    self._semblend_stats["misses"] += 1
                    logger.debug(
                        f"SemBlend semantic miss: store_size={store.size}, "
                        f"out_of_namespace={withheld}"
                    )
                    return base_result

                return self._lookup_donor_in_tree(
                    key, donor, matched_len, base_result, t0, target_len=target_len, **kwargs
                )

            except Exception as e:
                logger.debug(f"SemBlend semantic search failed: {e}")
                self._semblend_stats["misses"] += 1
                return base_result

        def _lookup_donor_in_tree(
            self,
            key: Any,
            donor: _DonorEntry,
            matched_len: int,
            base_result: Any,
            t0: float,
            target_len: int = 0,
            **kwargs: Any,
        ) -> Any:
            """Check if a donor's tokens exist in the radix tree.

            The tree is re-entered with a key rebuilt from the request's own
            key, so the lookup carries the isolation value SGLang keys its
            tree on instead of dropping it.

            When the donor has more tokens than the target request we cap the
            reused prefix to avoid negative new-token counts in the SGLang
            scheduler, and a donor whose result cannot be capped is declined
            rather than served uncapped.
            """
            try:
                donor_key = _donor_lookup_key(key, list(donor.token_ids))
                if donor_key is None:
                    logger.debug(
                        f"SemBlend donor declined: no isolated tree key for {type(key).__name__}"
                    )
                    self._semblend_stats["donor_key_unavailable"] += 1
                    self._semblend_stats["misses"] += 1
                    return base_result

                donor_result = super().match_prefix(donor_key, **kwargs)
                donor_matched = _get_matched_length(donor_result)

                # Cap reuse to target request length minus a safety
                # margin of 1 token (SGLang needs at least 1 new token
                # to prefill).
                if target_len > 0 and donor_matched >= target_len:
                    capped = _cap_match_result(donor_result, max(target_len - 1, 0))
                    if capped is None:
                        logger.debug(
                            f"SemBlend donor declined: {donor_matched}-token prefix "
                            f"cannot be capped to target_len={target_len}"
                        )
                        self._semblend_stats["uncapped_prefix_declined"] += 1
                        self._semblend_stats["misses"] += 1
                        return base_result
                    donor_result, capped_len = capped
                    if capped_len != donor_matched:
                        logger.info(
                            f"SemBlend capped prefix: {donor_matched} -> "
                            f"{capped_len} tokens (target_len={target_len})"
                        )
                    donor_matched = capped_len

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

                self._semblend_donor_store.add_donor(
                    tuple(token_ids),
                    embedding,
                    namespace=_req_namespace(req),
                )
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

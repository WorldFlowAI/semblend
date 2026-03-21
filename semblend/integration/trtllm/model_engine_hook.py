"""TRT-LLM model engine hooks -- prefix match interception for semantic KV reuse.

Implements three approaches for intercepting TRT-LLM's prefix matching,
applied in order of preference:

Approach B (Token Substitution, preferred):
    Before enqueue_request(): run SemBlend pipeline. On semantic hit,
    substitute donor's token IDs for the prompt prefix. TRT-LLM's radix
    tree then matches the donor's tokens -> loads cached KV. After load,
    apply RoPE correction via post-load hook.

Approach A (Radix Tree Subclass):
    If TRT-LLM's PyTorch backend uses a Python-accessible radix tree,
    subclass/patch the tree's search/match_prefix method. On exact miss,
    run SemBlend pipeline. On semantic hit, return donor's match result.

Approach C (Block-Level Injection, fallback):
    After TRT-LLM allocates blocks for the new request, before prefill:
    inject donor KV into allocated blocks via get_buffers(). Signal
    engine to skip prefill for injected tokens.

The active approach is selected at init based on available TRT-LLM APIs.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

logger = logging.getLogger("semblend.trtllm.hook")


class SemBlendModelEngineHook:
    """Hook into TRT-LLM's PyTorch ModelEngine for semantic KV reuse.

    Wraps the model engine's request submission path to intercept prefix
    matching and inject semantic donor KV when applicable.

    Usage:
        from tensorrt_llm.torch import ModelEngine
        engine = ModelEngine(...)
        hook = SemBlendModelEngineHook(engine, backend)
        # hook.wrap() patches engine's enqueue method
        hook.wrap()

    Args:
        engine: TRT-LLM PyTorch ModelEngine instance.
        backend: TRTLLMPyTorchBackend instance with initialized pipeline.
        approach: Force a specific approach ("token_sub", "radix_patch",
            "block_inject"). If None, auto-detects best available.
    """

    def __init__(
        self,
        engine: Any,
        backend: Any,
        approach: str | None = None,
    ) -> None:
        self._engine = engine
        self._backend = backend
        self._approach = approach or os.environ.get(
            "SEMBLEND_TRTLLM_APPROACH", "auto"
        )
        self._original_enqueue = None
        self._active_approach = None
        self._stats = {
            "requests_intercepted": 0,
            "substitutions_applied": 0,
            "corrections_applied": 0,
        }

    def wrap(self) -> str:
        """Patch the engine's request submission path.

        Auto-detects the best approach based on available TRT-LLM APIs.

        Returns:
            Name of the active approach ("token_sub", "radix_patch",
            "block_inject", or "none" if no approach works).
        """
        if self._approach == "auto":
            approach = self._detect_best_approach()
        else:
            approach = self._approach

        if approach == "token_sub":
            self._wrap_token_substitution()
        elif approach == "radix_patch":
            self._wrap_radix_patch()
        elif approach == "block_inject":
            self._wrap_block_injection()
        else:
            logger.warning(
                "No viable SemBlend approach detected for TRT-LLM. "
                "Semantic KV reuse disabled."
            )
            self._active_approach = "none"
            return "none"

        self._active_approach = approach
        logger.info("SemBlend TRT-LLM hook active: approach=%s", approach)
        return approach

    def unwrap(self) -> None:
        """Restore the original engine behavior."""
        if self._original_enqueue is not None:
            self._engine.enqueue_request = self._original_enqueue
            self._original_enqueue = None
            self._active_approach = None
            logger.info("SemBlend TRT-LLM hook removed")

    @property
    def active_approach(self) -> str | None:
        return self._active_approach

    def get_stats(self) -> dict:
        return {**self._stats}

    # ------------------------------------------------------------------
    # Approach detection
    # ------------------------------------------------------------------

    def _detect_best_approach(self) -> str:
        """Detect the best interception approach for this TRT-LLM version.

        Checks for available APIs in order of preference:
        1. Python-accessible enqueue_request -> token substitution
        2. Python radix tree class -> radix patch
        3. KVCacheManager block allocation API -> block injection
        """
        # Check for enqueue_request (token substitution)
        if hasattr(self._engine, "enqueue_request"):
            logger.info(
                "Detected enqueue_request API -> using token substitution"
            )
            return "token_sub"

        # Check for enqueue (variant API name)
        if hasattr(self._engine, "enqueue"):
            logger.info(
                "Detected enqueue API -> using token substitution"
            )
            return "token_sub"

        # Check for Python-accessible radix tree
        kv_mgr = getattr(self._engine, "kv_cache_manager", None)
        if kv_mgr is not None:
            radix_tree = getattr(kv_mgr, "radix_tree", None)
            if radix_tree is not None and hasattr(radix_tree, "match_prefix"):
                logger.info(
                    "Detected Python radix tree -> using radix patch"
                )
                return "radix_patch"

        # Check for block allocation API
        if kv_mgr is not None and hasattr(kv_mgr, "get_buffers"):
            logger.info(
                "Detected get_buffers API -> using block injection"
            )
            return "block_inject"

        return "none"

    # ------------------------------------------------------------------
    # Approach B: Token Substitution (preferred)
    # ------------------------------------------------------------------

    def _wrap_token_substitution(self) -> None:
        """Patch enqueue_request to substitute donor tokens on semantic hit.

        Before the original enqueue_request:
        1. Run SemBlend pipeline on incoming token IDs
        2. If semantic hit: replace token IDs with donor's tokens
        3. TRT-LLM's radix tree matches donor -> loads cached KV
        4. Schedule RoPE correction after KV load
        """
        enqueue_attr = "enqueue_request"
        if not hasattr(self._engine, enqueue_attr):
            enqueue_attr = "enqueue"

        self._original_enqueue = getattr(self._engine, enqueue_attr)
        hook = self

        def patched_enqueue(request: Any, *args: Any, **kwargs: Any) -> Any:
            return hook._intercept_request(
                request, hook._original_enqueue, *args, **kwargs
            )

        setattr(self._engine, enqueue_attr, patched_enqueue)
        logger.info("Patched %s.%s for token substitution", type(self._engine).__name__, enqueue_attr)

    def _intercept_request(
        self,
        request: Any,
        original_fn: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Intercept a request and attempt semantic donor substitution.

        Graceful degradation: any error falls through to the original
        path without SemBlend enhancement.
        """
        self._stats["requests_intercepted"] += 1

        try:
            token_ids = self._extract_token_ids(request)
            if token_ids is None or len(token_ids) < 100:
                return original_fn(request, *args, **kwargs)

            donor_info = self._backend.find_semantic_donor(token_ids)
            if donor_info is None:
                return original_fn(request, *args, **kwargs)

            # Substitute donor tokens
            donor_tokens = donor_info["donor_tokens"]
            self._substitute_tokens(request, donor_tokens)
            self._stats["substitutions_applied"] += 1

            # Schedule post-load RoPE correction
            position_map = donor_info.get("position_map")
            if position_map is not None and position_map.needs_correction:
                self._schedule_rope_correction(request, position_map)

            return original_fn(request, *args, **kwargs)

        except Exception as e:
            logger.debug("SemBlend interception failed (graceful fallthrough): %s", e)
            return original_fn(request, *args, **kwargs)

    def _extract_token_ids(self, request: Any) -> list[int] | None:
        """Extract token IDs from a TRT-LLM request object."""
        # Try common attribute names
        for attr in ("token_ids", "input_ids", "input_token_ids", "tokens"):
            if hasattr(request, attr):
                val = getattr(request, attr)
                if isinstance(val, (list, tuple)):
                    return list(val)
                if hasattr(val, "tolist"):
                    return val.tolist()
        return None

    def _substitute_tokens(self, request: Any, donor_tokens: list[int]) -> None:
        """Replace the request's token IDs with donor tokens.

        Same technique used in vLLM connector (semblend_connector.py:790):
        req_meta.token_ids = new_tokens
        """
        for attr in ("token_ids", "input_ids", "input_token_ids", "tokens"):
            if hasattr(request, attr):
                setattr(request, attr, donor_tokens)
                return

    def _schedule_rope_correction(
        self, request: Any, position_map: Any
    ) -> None:
        """Tag the request for post-load RoPE correction.

        The correction is applied after TRT-LLM loads the donor's KV
        blocks into the new request's cache.
        """
        # Store correction metadata on the request for the post-load hook
        request._semblend_position_map = position_map
        request._semblend_needs_correction = True

    # ------------------------------------------------------------------
    # Approach A: Radix Tree Patch
    # ------------------------------------------------------------------

    def _wrap_radix_patch(self) -> None:
        """Patch the KVCacheManager's radix tree for semantic fallback.

        Similar to SGLang's radix_backend.py pattern: subclass/patch
        the tree's match_prefix method.
        """
        kv_mgr = self._engine.kv_cache_manager
        radix_tree = kv_mgr.radix_tree
        original_match = radix_tree.match_prefix
        hook = self

        def patched_match_prefix(token_ids: Any, *args: Any, **kwargs: Any) -> Any:
            result = original_match(token_ids, *args, **kwargs)
            matched_len = hook._get_matched_length(result)
            total_len = len(token_ids) if hasattr(token_ids, "__len__") else 0

            if matched_len >= total_len * 0.5:
                return result

            # Short match -- try semantic fallback
            return hook._semantic_radix_fallback(
                token_ids, matched_len, result,
                original_match, *args, **kwargs,
            )

        radix_tree.match_prefix = patched_match_prefix
        logger.info("Patched radix tree match_prefix for semantic fallback")

    def _semantic_radix_fallback(
        self,
        token_ids: Any,
        matched_len: int,
        base_result: Any,
        original_match: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Attempt semantic donor lookup, then check radix tree for donor."""
        try:
            ids = list(token_ids) if hasattr(token_ids, "__iter__") else []
            donor_info = self._backend.find_semantic_donor(ids)
            if donor_info is None:
                return base_result

            # Check if donor's tokens are in the radix tree
            donor_tokens = donor_info["donor_tokens"]
            donor_result = original_match(donor_tokens, *args, **kwargs)
            donor_matched = self._get_matched_length(donor_result)

            if donor_matched > matched_len:
                self._stats["substitutions_applied"] += 1
                return donor_result

        except Exception as e:
            logger.debug("Semantic radix fallback failed: %s", e)

        return base_result

    @staticmethod
    def _get_matched_length(result: Any) -> int:
        """Extract matched prefix length from a match result."""
        if isinstance(result, int):
            return result
        if isinstance(result, tuple) and len(result) >= 1:
            first = result[0]
            if hasattr(first, "__len__"):
                return len(first)
            if isinstance(first, int):
                return first
        if hasattr(result, "matched_length"):
            return result.matched_length
        return 0

    # ------------------------------------------------------------------
    # Approach C: Block-Level Injection
    # ------------------------------------------------------------------

    def _wrap_block_injection(self) -> None:
        """Set up block-level KV injection for pre-prefill injection.

        This is the fallback approach when neither enqueue nor radix
        tree are patchable. Requires the engine to expose a pre-prefill
        hook point.
        """
        logger.warning(
            "Block injection approach selected but not yet implemented. "
            "This requires upstream TRT-LLM API support "
            "(SemanticCacheLookupProvider / PostPrefixLoadHook)."
        )


class PostPrefillRoPEHook:
    """Post-prefill hook that applies RoPE correction on injected donor KV.

    Registered as a callback after TRT-LLM completes prefill for a request
    that had donor KV substituted. Iterates over all layers and applies
    the position delta correction to K cache.

    Args:
        backend: TRTLLMPyTorchBackend for KV access and RoPE computation.
    """

    def __init__(self, backend: Any) -> None:
        self._backend = backend

    def on_prefill_complete(
        self,
        request: Any,
        block_table: Any,
    ) -> None:
        """Called after TRT-LLM loads prefix KV for a request.

        Checks if the request has pending RoPE correction metadata
        and applies it to all layers.
        """
        position_map = getattr(request, "_semblend_position_map", None)
        needs_correction = getattr(request, "_semblend_needs_correction", False)

        if not needs_correction or position_map is None:
            return
        if not position_map.needs_correction:
            return

        t0 = time.monotonic()
        num_layers = self._backend._num_layers

        for layer_idx in range(num_layers):
            kv_buffer = self._backend._kv_mgr.get_buffers(layer_idx)
            self._backend.apply_rope_correction(
                kv_cache=kv_buffer,
                position_deltas=position_map,
                rope_config={
                    "rope_base": self._backend._rope_base,
                    "head_dim": self._backend._head_dim,
                    "block_table": block_table,
                },
            )

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "RoPE correction applied: %d layers, %d pairs (%.1fms)",
            num_layers,
            position_map.num_pairs,
            elapsed_ms,
        )

        # Clear correction flag
        request._semblend_needs_correction = False

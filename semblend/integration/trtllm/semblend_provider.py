"""SemBlend reference implementation of SemanticCacheLookupProvider.

This is the concrete implementation that would be included in the
upstream PR to NVIDIA/TensorRT-LLM as the reference provider.

It bridges SemBlend's pipeline (embedding, donor lookup, alignment)
with the TRT-LLM SemanticCacheLookupProvider ABC, enabling semantic
KV cache reuse in TRT-LLM's prefix matching path.

Usage (after upstream PR is merged):
    from tensorrt_llm.kv_cache import KvCacheConfig
    from semblend.integration.trtllm.semblend_provider import SemBlendProvider

    config = KvCacheConfig(
        semantic_cache_provider=SemBlendProvider(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            min_similarity=0.60,
        ),
    )
    llm = LLM(model=model, kv_cache_config=config)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from semblend.integration.trtllm.upstream_interface import (
    PostPrefixLoadHook,
    SemanticCacheLookupProvider,
    SemanticMatchResult,
)

logger = logging.getLogger("semblend.trtllm.provider")


class SemBlendProvider(SemanticCacheLookupProvider):
    """SemBlend implementation of TRT-LLM's SemanticCacheLookupProvider.

    Provides semantic donor discovery for TRT-LLM's KV cache system.
    When the radix tree exact-prefix match fails, this provider searches
    for semantically similar cached prompts using MiniLM embeddings.

    Args:
        model_name: Model name for tokenizer and bathtub preset lookup.
        min_similarity: Minimum cosine similarity for donor matching.
        min_reuse_ratio: Minimum alignment reuse ratio.
        max_donors: Maximum entries in the donor store.
        embedder_type: Embedder type ("minilm", "onnx-gpu", "jaccard").
        chunk_size: KV block size for alignment (default 128 for TRT-LLM).
    """

    def __init__(
        self,
        model_name: str = "",
        min_similarity: float = 0.60,
        min_reuse_ratio: float = 0.50,
        max_donors: int = 10_000,
        embedder_type: str = "minilm",
        chunk_size: int = 128,
    ) -> None:
        self._model_name = model_name or os.environ.get("SEMBLEND_MODEL_NAME", "")
        self._min_similarity = min_similarity
        self._min_reuse_ratio = min_reuse_ratio
        self._max_donors = max_donors
        self._embedder_type = embedder_type
        self._chunk_size = chunk_size

        self._pipeline = None
        self._tokenizer = None
        self._tokenizer_load_attempted = False

        self._stats = {
            "queries": 0,
            "hits": 0,
            "misses": 0,
            "registrations": 0,
            "evictions": 0,
            "avg_query_ms": 0.0,
        }

        logger.info(
            "SemBlendProvider initialized: model=%s, min_sim=%.2f, chunk_size=%d, max_donors=%d",
            self._model_name,
            self._min_similarity,
            self._chunk_size,
            self._max_donors,
        )

    def find_semantic_match(
        self,
        token_ids: list[int],
        prompt_text: str,
    ) -> SemanticMatchResult | None:
        """Find a semantically similar cached prompt.

        Called when exact prefix match is shorter than threshold.
        Runs the full SemBlend pipeline: embed -> search -> align.

        Args:
            token_ids: Token IDs of the incoming request.
            prompt_text: Decoded prompt text for embedding.

        Returns:
            SemanticMatchResult on hit, None on miss.
        """
        self._stats["queries"] += 1
        t0 = time.monotonic()

        pipeline = self._get_pipeline()
        if pipeline is None:
            self._stats["misses"] += 1
            return None

        # Use provided prompt_text, or decode from tokens
        if not prompt_text:
            prompt_text = self._tokens_to_text(token_ids) or ""
        if not prompt_text or len(token_ids) < 100:
            self._stats["misses"] += 1
            return None

        result = pipeline.find_donor(
            token_ids=token_ids,
            prompt_text=prompt_text,
        )

        elapsed_ms = (time.monotonic() - t0) * 1000
        self._update_avg_ms(elapsed_ms)

        if not result.found:
            self._stats["misses"] += 1
            return None

        self._stats["hits"] += 1

        # Build position mapping from slot actions
        position_mapping = []
        if result.position_map and result.position_map.num_pairs > 0:
            position_mapping = list(
                zip(
                    result.position_map.donor_positions,
                    result.position_map.target_positions,
                )
            )

        logger.info(
            "Semantic hit: donor=%s sim=%.3f reuse=%.2f pairs=%d (%.1fms)",
            result.donor_id,
            result.similarity,
            result.reuse_ratio,
            len(position_mapping),
            elapsed_ms,
        )

        return SemanticMatchResult(
            donor_token_ids=result.donor_tokens,
            similarity=result.similarity,
            reuse_ratio=result.reuse_ratio,
            position_mapping=position_mapping,
            donor_id=result.donor_id or "",
            metadata={
                "timings": {
                    "embed_ms": result.timings.embed_ms,
                    "lookup_ms": result.timings.lookup_ms,
                    "total_ms": result.timings.total_ms,
                },
            },
        )

    def register_completed(
        self,
        request_id: str,
        token_ids: list[int],
        prompt_text: str,
    ) -> None:
        """Register a completed request as a potential donor."""
        pipeline = self._get_pipeline()
        if pipeline is None:
            return

        if not prompt_text:
            prompt_text = self._tokens_to_text(token_ids) or ""
        if not prompt_text or len(token_ids) < 100:
            return

        pipeline.register_donor(
            request_id=request_id,
            token_ids=token_ids,
            prompt_text=prompt_text,
        )
        self._stats["registrations"] += 1

    def on_eviction(self, request_id: str) -> None:
        """Handle donor eviction (no-op for in-memory store)."""
        self._stats["evictions"] += 1

    def get_stats(self) -> dict:
        pipeline = self._pipeline
        donor_count = pipeline.donor_count if pipeline is not None else 0
        return {
            **self._stats,
            "donor_store_size": donor_count,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        enabled = os.environ.get("SEMBLEND_ENABLED", "1") == "1"
        if not enabled:
            return None

        from semblend_core.pipeline import SemBlendPipeline

        self._pipeline = SemBlendPipeline(
            max_donors=self._max_donors,
            min_similarity=self._min_similarity,
            min_reuse_ratio=self._min_reuse_ratio,
            embedder_type=self._embedder_type,
            model_name=self._model_name,
            chunk_size=self._chunk_size,
        )
        return self._pipeline

    def _get_tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer
        if self._tokenizer_load_attempted:
            return None
        self._tokenizer_load_attempted = True

        if not self._model_name:
            return None

        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_name, trust_remote_code=True
            )
        except Exception as e:
            logger.error("Failed to load tokenizer: %s", e)
        return self._tokenizer

    def _tokens_to_text(self, token_ids: list[int]) -> str | None:
        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            return None
        max_decode = 2000
        sampled = token_ids[:max_decode] if len(token_ids) > max_decode else token_ids
        return tokenizer.decode(sampled, skip_special_tokens=True)

    def _update_avg_ms(self, elapsed_ms: float) -> None:
        n = self._stats["hits"] + self._stats["misses"]
        if n > 0:
            self._stats["avg_query_ms"] = (self._stats["avg_query_ms"] * (n - 1) + elapsed_ms) / n


class SemBlendPostLoadHook(PostPrefixLoadHook):
    """SemBlend implementation of TRT-LLM's PostPrefixLoadHook.

    Applies RoPE position correction to K cache tensors after
    semantic donor KV is loaded. Uses the existing SemBlend
    Triton kernels with TRT-LLM stride computation.

    Args:
        rope_base: RoPE base frequency (default 10000.0).
    """

    def __init__(self, rope_base: float = 10000.0) -> None:
        self._rope_base = rope_base
        self._stats = {
            "corrections_applied": 0,
            "total_pairs_corrected": 0,
        }

    def on_prefix_loaded(
        self,
        kv_buffers: list[Any],
        block_table: Any,
        position_mapping: list[tuple[int, int]],
        rope_config: dict,
    ) -> None:
        """Apply RoPE correction after prefix KV load.

        For each layer's KV buffer, applies RoPE(target_pos - donor_pos)
        correction to the K cache. V cache is unchanged.
        """
        if not position_mapping:
            return

        # Check if any position pairs actually need correction
        needs_correction = any(d != t for d, t in position_mapping)
        if not needs_correction:
            return

        rope_base = rope_config.get("rope_base", self._rope_base)

        from semblend_core.rope_correction import apply_rope_delta_inplace

        total_corrected = 0
        for layer_idx, kv_buffer in enumerate(kv_buffers):
            for donor_pos, target_pos in position_mapping:
                delta = target_pos - donor_pos
                if delta != 0:
                    modified = apply_rope_delta_inplace(
                        kv_cache=kv_buffer,
                        block_table=block_table,
                        positions=[target_pos],
                        delta=delta,
                        rope_base=rope_base,
                    )
                    total_corrected += modified

        self._stats["corrections_applied"] += 1
        self._stats["total_pairs_corrected"] += total_corrected

        logger.debug(
            "RoPE correction: %d layers, %d pairs, %d positions modified",
            len(kv_buffers),
            len(position_mapping),
            total_corrected,
        )

    def get_stats(self) -> dict:
        return {**self._stats}

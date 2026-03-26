"""Dynamo KvIndexer wrapper with SemBlend semantic fallback.

Wraps Dynamo's KvIndexer (Python binding of the Rust RadixTree) to add
semantic donor discovery when exact prefix matching returns low overlap.

The wrapper intercepts find_matches_for_request():
1. Call Dynamo's RadixTree for exact block-hash prefix match
2. If overlap is below threshold: run SemBlend semantic search
3. If semantic hit: look up donor's token hashes in the RadixTree
4. Return the higher of (exact overlap, donor overlap)

This is analogous to the SGLang RadixCache pattern:
    radix_backend.py:match_prefix() -> semantic fallback -> donor tree lookup

But operates at the Dynamo routing layer, making it engine-agnostic.

Requires: dynamo Python bindings (pip install nvidia-dynamo)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

logger = logging.getLogger("semblend.dynamo.indexer")


class SemBlendKvIndexerWrapper:
    """Wraps Dynamo's KvIndexer with semantic fallback on prefix miss.

    When the RadixTree returns low overlap for a request, this wrapper
    runs SemBlend's embedding search to find a semantically similar
    cached request. If found, it queries the RadixTree with the donor's
    tokens to get the donor's overlap scores.

    Args:
        inner_indexer: Dynamo's KvIndexer (Python binding).
            Must expose find_matches_for_request(tokens, lora_name).
        kv_block_size: Tokens per KV block (default 32 for Dynamo).
        min_similarity: Minimum cosine similarity for semantic matching.
        min_overlap_ratio: Below this overlap ratio, semantic search triggers.
        max_donors: Maximum entries in the semantic donor store.
        embedder_type: Embedder type ("minilm", "jaccard", etc.).
        model_name: Model name for bathtub curve preset lookup.
    """

    def __init__(
        self,
        inner_indexer: Any,
        kv_block_size: int = 32,
        min_similarity: float = 0.60,
        min_overlap_ratio: float = 0.50,
        max_donors: int = 10_000,
        embedder_type: str = "minilm",
        model_name: str = "",
    ) -> None:
        self._inner = inner_indexer
        self._kv_block_size = kv_block_size
        self._min_overlap_ratio = min_overlap_ratio

        # Lazy init to avoid import overhead at module scope
        self._pipeline = None
        self._min_similarity = min_similarity
        self._max_donors = max_donors
        self._embedder_type = embedder_type
        self._model_name = model_name

        # Tokenizer for token-to-text decoding
        self._tokenizer = None
        self._tokenizer_load_attempted = False

        # Stats
        self._stats = {
            "total_queries": 0,
            "exact_hits": 0,
            "semantic_hits": 0,
            "semantic_misses": 0,
            "donors_registered": 0,
            "avg_semantic_ms": 0.0,
        }

        logger.info(
            "SemBlend KvIndexer wrapper initialized: block_size=%d, min_sim=%.2f, min_overlap=%.2f",
            kv_block_size,
            min_similarity,
            min_overlap_ratio,
        )

    def find_matches_for_request(
        self,
        tokens: list[int],
        lora_name: str | None = None,
    ) -> dict:
        """Find KV cache matches with semantic fallback.

        First queries Dynamo's RadixTree for exact prefix matches.
        If overlap is below threshold, runs SemBlend semantic search
        and returns the donor's overlap scores if better.

        Args:
            tokens: Token IDs of the incoming request.
            lora_name: Optional LoRA adapter name.

        Returns:
            OverlapScores dict: {worker_id: num_matched_blocks}.
        """
        self._stats["total_queries"] += 1

        # Step 1: Exact prefix match via Dynamo RadixTree
        exact_scores = self._inner.find_matches_for_request(tokens, lora_name)

        # Check if exact match is good enough
        total_blocks = max(len(tokens) // self._kv_block_size, 1)
        best_overlap = max(exact_scores.values()) if exact_scores else 0
        overlap_ratio = best_overlap / total_blocks

        if overlap_ratio >= self._min_overlap_ratio:
            self._stats["exact_hits"] += 1
            return exact_scores

        # Step 2: Semantic fallback
        semantic_scores = self._try_semantic_match(tokens, lora_name)
        if semantic_scores is not None:
            # Return the better of exact and semantic scores
            merged = dict(exact_scores)
            for worker, score in semantic_scores.items():
                if score > merged.get(worker, 0):
                    merged[worker] = score
            return merged

        return exact_scores

    def apply_event(self, event: Any) -> None:
        """Forward events to inner indexer and register donors."""
        self._inner.apply_event(event)

        # Register completed requests as donors
        self._register_donor_from_event(event)

    def remove_worker(self, worker_id: Any) -> None:
        """Forward to inner indexer."""
        self._inner.remove_worker(worker_id)

    def remove_worker_dp_rank(self, worker_id: Any, dp_rank: Any) -> None:
        """Forward to inner indexer."""
        self._inner.remove_worker_dp_rank(worker_id, dp_rank)

    def shutdown(self) -> None:
        """Forward to inner indexer."""
        self._inner.shutdown()

    # ------------------------------------------------------------------
    # Semantic search
    # ------------------------------------------------------------------

    def _try_semantic_match(
        self,
        tokens: list[int],
        lora_name: str | None,
    ) -> dict | None:
        """Attempt semantic donor lookup, then check RadixTree for donor."""
        if len(tokens) < 100:
            self._stats["semantic_misses"] += 1
            return None

        t0 = time.monotonic()

        pipeline = self._get_pipeline()
        if pipeline is None:
            self._stats["semantic_misses"] += 1
            return None

        prompt_text = self._tokens_to_text(tokens)
        if prompt_text is None:
            self._stats["semantic_misses"] += 1
            return None

        result = pipeline.find_donor(
            token_ids=tokens,
            prompt_text=prompt_text,
        )

        if not result.found:
            elapsed_ms = (time.monotonic() - t0) * 1000
            self._update_avg_ms(elapsed_ms)
            self._stats["semantic_misses"] += 1
            return None

        # Found a semantic match — look up donor's tokens in RadixTree
        donor_scores = self._inner.find_matches_for_request(
            result.donor_tokens,
            lora_name,
        )

        elapsed_ms = (time.monotonic() - t0) * 1000
        self._update_avg_ms(elapsed_ms)

        if donor_scores and max(donor_scores.values(), default=0) > 0:
            self._stats["semantic_hits"] += 1
            logger.info(
                "SemBlend semantic hit: donor=%s, sim=%.3f, reuse=%.2f, "
                "donor_overlap=%d blocks (%.1fms)",
                result.donor_id,
                result.similarity,
                result.reuse_ratio,
                max(donor_scores.values()),
                elapsed_ms,
            )
            return donor_scores

        self._stats["semantic_misses"] += 1
        return None

    # ------------------------------------------------------------------
    # Donor registration
    # ------------------------------------------------------------------

    def register_completed_request(
        self,
        request_id: str,
        tokens: list[int],
        prompt_text: str = "",
    ) -> None:
        """Register a completed request as a donor for future semantic matching.

        Called by the engine connector after a request finishes generation.
        """
        pipeline = self._get_pipeline()
        if pipeline is None:
            return

        if not prompt_text:
            prompt_text = self._tokens_to_text(tokens) or ""

        if not prompt_text or len(tokens) < 100:
            return

        pipeline.register_donor(
            request_id=request_id,
            token_ids=tokens,
            prompt_text=prompt_text,
        )
        self._stats["donors_registered"] += 1

    def _register_donor_from_event(self, event: Any) -> None:
        """Extract token info from a Dynamo KV event and register donor."""
        # Dynamo events contain block hashes, not raw tokens.
        # Donor registration requires token IDs + text, which must come
        # from the engine connector (not the event stream).
        # This is a no-op; registration happens via register_completed_request().
        pass

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

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
            min_reuse_ratio=float(os.environ.get("SEMBLEND_MIN_REUSE_RATIO", "0.50")),
            embedder_type=self._embedder_type,
            model_name=self._model_name,
            chunk_size=self._kv_block_size,
        )
        return self._pipeline

    def _get_tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer
        if self._tokenizer_load_attempted:
            return None
        self._tokenizer_load_attempted = True

        model_name = self._model_name or os.environ.get("SEMBLEND_MODEL_NAME", "")
        if not model_name:
            return None

        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        except Exception as e:
            logger.error("Failed to load tokenizer: %s", e)
        return self._tokenizer

    def _tokens_to_text(self, token_ids: list[int]) -> str | None:
        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            return None

        max_decode = 2000
        n = len(token_ids)
        if n <= max_decode:
            sampled = token_ids
        else:
            head = int(max_decode * 0.40)
            mid_w = int(max_decode * 0.30)
            tail = max_decode - head - mid_w
            mid_start = (n - mid_w) // 2
            sampled = (
                token_ids[:head] + token_ids[mid_start : mid_start + mid_w] + token_ids[n - tail :]
            )
        return tokenizer.decode(sampled, skip_special_tokens=True)

    def _update_avg_ms(self, elapsed_ms: float) -> None:
        n = self._stats["semantic_hits"] + self._stats["semantic_misses"]
        if n > 0:
            self._stats["avg_semantic_ms"] = (
                self._stats["avg_semantic_ms"] * (n - 1) + elapsed_ms
            ) / n

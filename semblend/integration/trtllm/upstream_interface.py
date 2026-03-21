"""Proposed upstream TRT-LLM interfaces for semantic KV cache reuse.

These ABCs define the standardized interface for semantic cache providers
in TRT-LLM, matching the upstream PR patterns for other engines:
    - LMCache: SemanticLookupProvider (PR #2803) + PostLoadHook (PR #2804)
    - SGLang: SemanticPrefixProvider (PR #20806)

These interfaces are designed for inclusion in an upstream PR to
NVIDIA/TensorRT-LLM. SemBlend serves as the reference implementation.

The interfaces are minimal (2 ABCs, ~4 methods total) to minimize the
surface area of the upstream change while enabling full semantic cache
functionality.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SemanticMatchResult:
    """Result of a semantic cache lookup.

    Returned by SemanticCacheLookupProvider.find_semantic_match() when
    a semantically similar cached prompt is found.

    Attributes:
        donor_token_ids: Token IDs of the matched donor prompt. The engine
            can use these to perform a radix tree lookup for cached KV.
        similarity: Cosine similarity score between query and donor embeddings.
        reuse_ratio: Fraction of donor tokens aligned to the query (0-1).
            Higher means more KV can be reused.
        position_mapping: List of (donor_pos, target_pos) pairs indicating
            where donor KV should be placed in the target sequence. Used
            for RoPE position correction after KV load.
        donor_id: Optional identifier for the matched donor request.
        metadata: Optional engine-specific metadata for the match.
    """
    donor_token_ids: list[int]
    similarity: float
    reuse_ratio: float
    position_mapping: list[tuple[int, int]] = field(default_factory=list)
    donor_id: str = ""
    metadata: dict = field(default_factory=dict)


class SemanticCacheLookupProvider(ABC):
    """Provider for semantic (approximate) KV cache matching in TRT-LLM.

    When TRT-LLM's radix tree exact-prefix match fails or returns a
    short match, this provider is consulted for semantically similar
    cached prompts whose KV can be reused with position correction.

    The provider is registered with the KV cache manager at engine
    startup and queried on each incoming request.

    Integration flow:
        1. New request arrives with token_ids
        2. Radix tree prefix match returns matched_length
        3. If matched_length < threshold:
            a. Call find_semantic_match(token_ids, prompt_text)
            b. If match found: use donor_token_ids for radix tree lookup
            c. Apply RoPE correction via PostPrefixLoadHook
        4. On request completion: call register_completed()

    This interface is designed for upstream inclusion in TRT-LLM,
    following the patterns established by LMCache (PR #2803) and
    SGLang (PR #20806).
    """

    @abstractmethod
    def find_semantic_match(
        self,
        token_ids: list[int],
        prompt_text: str,
    ) -> SemanticMatchResult | None:
        """Find a semantically similar cached prompt.

        Called when the exact prefix match is shorter than the threshold
        (typically < 50% of the prompt length). Should complete in < 10ms
        to avoid adding latency to the request path.

        Args:
            token_ids: Token IDs of the incoming request.
            prompt_text: Decoded prompt text for semantic embedding.

        Returns:
            SemanticMatchResult with donor token IDs, similarity score,
            and position mapping for RoPE correction. None if no match
            above the configured similarity threshold.
        """

    @abstractmethod
    def register_completed(
        self,
        request_id: str,
        token_ids: list[int],
        prompt_text: str,
    ) -> None:
        """Register a completed request as a potential donor.

        Called after a request finishes generation. The provider should
        store the request's embedding for future semantic matching.

        Args:
            request_id: Unique request identifier.
            token_ids: Token IDs of the completed request.
            prompt_text: Decoded prompt text for embedding computation.
        """

    def on_eviction(
        self,
        request_id: str,
    ) -> None:
        """Called when a donor's KV blocks are evicted from the cache.

        Optional: providers may use this to remove stale entries from
        their semantic index. Default implementation is a no-op.

        Args:
            request_id: Identifier of the evicted request.
        """


class PostPrefixLoadHook(ABC):
    """Hook called after prefix KV is loaded from cache, before prefill.

    Used for RoPE position correction on K cache when semantic matching
    produces position misalignment (donor_pos != target_pos).

    The hook is registered alongside the SemanticCacheLookupProvider
    and called by the KV cache manager after loading prefix KV blocks.

    Typical implementation applies RoPE(target_pos - donor_pos) correction
    to each layer's K cache for the misaligned positions, leaving V unchanged.

    This interface matches the PostLoadHook pattern from LMCache (PR #2804).
    """

    @abstractmethod
    def on_prefix_loaded(
        self,
        kv_buffers: list[Any],
        block_table: Any,
        position_mapping: list[tuple[int, int]],
        rope_config: dict,
    ) -> None:
        """Called after prefix KV is loaded from cache, before prefill.

        Args:
            kv_buffers: List of per-layer KV cache tensors. Each tensor
                has shape [num_blocks, 2, tokens_per_block, num_kv_heads, head_dim]
                for TRT-LLM PyTorch backend.
            block_table: Block table mapping logical to physical blocks.
            position_mapping: List of (donor_pos, target_pos) pairs from
                SemanticMatchResult. Used to compute RoPE deltas.
            rope_config: RoPE configuration dict with keys:
                - rope_base: Base frequency (default 10000.0)
                - head_dim: Dimension per attention head
                - rope_scaling: Optional scaling config (for extended context)
        """

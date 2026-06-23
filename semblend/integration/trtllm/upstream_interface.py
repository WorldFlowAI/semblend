"""TensorRT-LLM semantic KV compatibility surface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

try:  # pragma: no cover - exercised when TensorRT-LLM carries the upstream API.
    from tensorrt_llm._torch.pyexecutor.connectors.semantic_kv_cache import (
        CacheNamespace,
        DonorEvicted,
        DonorLocation,
        DonorRegistered,
        DonorRelocated,
        DonorSegment,
        DonorsPurged,
        ProviderGenerationReset,
        SemanticKvEngineAttentionMode,
        SemanticKvEngineExecution,
        SemanticKvEvent,
        SemanticKvLookupRequest,
        SemanticKvLookupResult,
        SemanticKvMaterializationKind,
        SemanticKvPlan,
        SemanticKvProvider,
        SemanticKvPublicationPolicy,
        SemanticKvSegment,
        load_semantic_kv_provider,
    )
except Exception:  # pragma: no cover - default in package unit tests.
    from semblend.integration.trtllm.semantic_types import (
        CacheNamespace,
        DonorEvicted,
        DonorLocation,
        DonorRegistered,
        DonorRelocated,
        DonorSegment,
        DonorsPurged,
        ProviderGenerationReset,
        SemanticKvEngineAttentionMode,
        SemanticKvEngineExecution,
        SemanticKvEvent,
        SemanticKvLookupRequest,
        SemanticKvLookupResult,
        SemanticKvMaterializationKind,
        SemanticKvPlan,
        SemanticKvProvider,
        SemanticKvPublicationPolicy,
        SemanticKvSegment,
        load_semantic_kv_provider,
    )


@dataclass(frozen=True)
class SemanticMatchResult:
    """Legacy result shape kept for older SemBlend TensorRT tests."""

    donor_token_ids: list[int]
    similarity: float
    reuse_ratio: float
    position_mapping: list[tuple[int, int]] = field(default_factory=list)
    donor_id: str = ""
    metadata: dict = field(default_factory=dict)


class SemanticCacheLookupProvider(ABC):
    """Legacy lookup provider facade retained for compatibility."""

    @abstractmethod
    def find_semantic_match(
        self,
        token_ids: list[int],
        prompt_text: str,
    ) -> SemanticMatchResult | None:
        pass

    @abstractmethod
    def register_completed(
        self,
        request_id: str,
        token_ids: list[int],
        prompt_text: str,
    ) -> None:
        pass

    def on_eviction(self, request_id: str) -> None:
        return


class PostPrefixLoadHook(ABC):
    """Legacy RoPE post-load hook facade retained for compatibility."""

    @abstractmethod
    def on_prefix_loaded(
        self,
        kv_buffers: list[Any],
        block_table: Any,
        position_mapping: list[tuple[int, int]],
        rope_config: dict,
    ) -> None:
        pass


__all__ = [
    "CacheNamespace",
    "DonorEvicted",
    "DonorLocation",
    "DonorRegistered",
    "DonorRelocated",
    "DonorSegment",
    "DonorsPurged",
    "PostPrefixLoadHook",
    "ProviderGenerationReset",
    "SemanticCacheLookupProvider",
    "SemanticKvEvent",
    "SemanticKvEngineAttentionMode",
    "SemanticKvEngineExecution",
    "SemanticKvLookupRequest",
    "SemanticKvLookupResult",
    "SemanticKvMaterializationKind",
    "SemanticKvPlan",
    "SemanticKvProvider",
    "SemanticKvPublicationPolicy",
    "SemanticKvSegment",
    "SemanticMatchResult",
    "load_semantic_kv_provider",
]

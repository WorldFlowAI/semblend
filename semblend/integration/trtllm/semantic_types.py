"""Fallback semantic KV cache types for TensorRT-LLM integration.

SemBlend imports TensorRT-LLM's upstream types when available. These local
definitions keep the provider package importable in development and tests.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence


class SemanticKvMaterializationKind(str, Enum):
    DISCOVERY_ONLY = "discovery_only"
    REQUEST_LOCAL_PREFIX = "request_local_prefix"
    REQUEST_LOCAL_SEGMENTED = "request_local_segmented"
    EXACT_PREFIX = "exact_prefix"


class SemanticKvPublicationPolicy(str, Enum):
    EXACT_CACHE = "exact_cache"
    REQUEST_LOCAL = "request_local"


class SemanticKvEngineAttentionMode(str, Enum):
    FULL_PREFILL = "full_prefill"
    SUFFIX_ONLY_AFTER_PREFIX = "suffix_only_after_prefix"


@dataclass(frozen=True)
class CacheNamespace:
    model: str
    tokenizer: str
    kv_layout: str
    block_size: int
    model_revision: str = ""
    tokenizer_revision: str = ""
    kv_dtype: str = ""
    cache_dtype: str = ""
    quantization: str = ""
    adapter: str = ""
    rope_config: Mapping[str, Any] = field(default_factory=dict)
    tensor_parallel: Mapping[str, Any] = field(default_factory=dict)
    backend_cache_layout: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "kv_layout": self.kv_layout,
            "block_size": self.block_size,
            "kv_dtype": self.kv_dtype,
            "cache_dtype": self.cache_dtype,
            "quantization": self.quantization,
            "adapter": self.adapter,
            "rope_config": dict(self.rope_config),
            "tensor_parallel": dict(self.tensor_parallel),
            "backend_cache_layout": self.backend_cache_layout,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CacheNamespace":
        return cls(
            model=str(data.get("model", "")),
            tokenizer=str(data.get("tokenizer", "")),
            model_revision=str(data.get("model_revision", "")),
            tokenizer_revision=str(data.get("tokenizer_revision", "")),
            kv_layout=str(data.get("kv_layout", "")),
            block_size=int(data.get("block_size", 0)),
            kv_dtype=str(data.get("kv_dtype", "")),
            cache_dtype=str(data.get("cache_dtype", "")),
            quantization=str(data.get("quantization", "")),
            adapter=str(data.get("adapter", "")),
            rope_config=dict(data.get("rope_config", {})),
            tensor_parallel=dict(data.get("tensor_parallel", {})),
            backend_cache_layout=str(data.get("backend_cache_layout", "")),
            extra=dict(data.get("extra", {})),
        )


@dataclass(frozen=True)
class DonorLocation:
    kind: str
    worker_id: Optional[int] = None
    dp_rank: Optional[int] = None
    tier: str = ""
    backend: str = ""
    locator: str = ""

    @classmethod
    def worker(cls, worker_id: int, dp_rank: int, tier: str = "device") -> "DonorLocation":
        return cls(kind="worker", worker_id=worker_id, dp_rank=dp_rank, tier=tier)

    @classmethod
    def shared(cls, backend: str, tier: str, locator: str) -> "DonorLocation":
        return cls(kind="shared", backend=backend, tier=tier, locator=locator)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"kind": self.kind, "tier": self.tier}
        if self.kind == "worker":
            data.update({"worker_id": self.worker_id, "dp_rank": self.dp_rank})
        else:
            data.update({"backend": self.backend, "locator": self.locator})
        return data


@dataclass(frozen=True)
class DonorSegment:
    segment_id: int
    token_range: tuple[int, int]
    digest: int
    block_hashes: tuple[int, ...] = ()
    provider_metadata: bytes = b""

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "token_range": list(self.token_range),
            "digest": self.digest,
            "block_hashes": list(self.block_hashes),
            "provider_metadata": list(self.provider_metadata),
        }


@dataclass(frozen=True)
class DonorRegistered:
    donor_id: str
    namespace: CacheNamespace
    location: DonorLocation
    token_count: int
    segments: tuple[DonorSegment, ...]
    provider_generation: int
    block_ids: tuple[int, ...] = ()

    @property
    def kind(self) -> str:
        return "donor_registered"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "donor_id": self.donor_id,
            "namespace": self.namespace.to_dict(),
            "location": self.location.to_dict(),
            "token_count": self.token_count,
            "segments": [segment.to_dict() for segment in self.segments],
            "provider_generation": self.provider_generation,
            "block_ids": list(self.block_ids),
        }


@dataclass(frozen=True)
class DonorEvicted:
    donor_id: str
    provider_generation: int
    segments: Optional[tuple[int, ...]] = None

    @property
    def kind(self) -> str:
        return "donor_evicted"

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "kind": self.kind,
            "donor_id": self.donor_id,
            "provider_generation": self.provider_generation,
        }
        if self.segments is not None:
            data["segments"] = list(self.segments)
        return data


@dataclass(frozen=True)
class DonorsPurged:
    scope: Mapping[str, Any]
    provider_generation: int

    @property
    def kind(self) -> str:
        return "donors_purged"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "scope": dict(self.scope),
            "provider_generation": self.provider_generation,
        }


@dataclass(frozen=True)
class ProviderGenerationReset:
    generation: int

    @property
    def kind(self) -> str:
        return "provider_generation_reset"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "generation": self.generation}


@dataclass(frozen=True)
class DonorRelocated:
    donor_id: str
    new_location: DonorLocation
    provider_generation: int

    @property
    def kind(self) -> str:
        return "donor_relocated"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "donor_id": self.donor_id,
            "new_location": self.new_location.to_dict(),
            "provider_generation": self.provider_generation,
        }


@dataclass(frozen=True)
class SemanticKvEvent:
    schema_version: int
    event_id: int
    worker_id: int
    dp_rank: int
    data: DonorRegistered | DonorEvicted | DonorsPurged | ProviderGenerationReset | DonorRelocated

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "worker_id": self.worker_id,
            "dp_rank": self.dp_rank,
            "data": self.data.to_dict(),
        }


@dataclass(frozen=True)
class SemanticKvSegment:
    donor_id: str
    donor_segment_id: int
    donor_positions: tuple[int, ...]
    target_positions: tuple[int, ...]
    donor_block_ids: tuple[int, ...] = ()
    target_block_ids: tuple[int, ...] = ()
    layer_recompute_mask: Optional[tuple[bool, ...]] = None
    donor_location: Optional[DonorLocation] = None
    provider_metadata: bytes = b""

    @property
    def token_count(self) -> int:
        return len(self.target_positions)


@dataclass(frozen=True)
class SemanticKvEngineExecution:
    attention_mode: SemanticKvEngineAttentionMode
    materialized_prefix_token_count: int
    suffix_start_position: int
    recompute_boundary_layer: Optional[int] = None
    force_recompute_layers: tuple[int, ...] = ()
    require_materialization_barrier: bool = True

    def __post_init__(self) -> None:
        if self.materialized_prefix_token_count < 0:
            raise ValueError("materialized prefix token count must be non-negative")
        if self.suffix_start_position < 0:
            raise ValueError("suffix start position must be non-negative")
        if self.suffix_start_position > self.materialized_prefix_token_count:
            raise ValueError("suffix start must be within the materialized prefix")
        if self.recompute_boundary_layer is not None and self.recompute_boundary_layer < 0:
            raise ValueError("recompute boundary layer must be non-negative")
        if any(layer < 0 for layer in self.force_recompute_layers):
            raise ValueError("force recompute layers must be non-negative")

    @property
    def uses_suffix_only_attention(self) -> bool:
        return self.attention_mode == SemanticKvEngineAttentionMode.SUFFIX_ONLY_AFTER_PREFIX

    def to_dict(self) -> dict[str, Any]:
        return {
            "attention_mode": self.attention_mode.value,
            "materialized_prefix_token_count": self.materialized_prefix_token_count,
            "suffix_start_position": self.suffix_start_position,
            "recompute_boundary_layer": self.recompute_boundary_layer,
            "force_recompute_layers": list(self.force_recompute_layers),
            "require_materialization_barrier": self.require_materialization_barrier,
        }


@dataclass(frozen=True)
class SemanticKvPlan:
    request_id: int
    namespace: CacheNamespace
    kind: SemanticKvMaterializationKind
    publication_policy: SemanticKvPublicationPolicy
    segments: tuple[SemanticKvSegment, ...] = ()
    donor_ids: tuple[str, ...] = ()
    covered_token_count: int = 0
    prefix_token_count: int = 0
    computed_token_count: int = 0
    requires_rope_correction: bool = False
    engine_execution: Optional[SemanticKvEngineExecution] = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.computed_token_count > self.prefix_token_count:
            raise ValueError("computed semantic KV tokens must be a prefix subset")
        if self.engine_execution is not None:
            if (
                self.engine_execution.materialized_prefix_token_count
                > self.prefix_token_count
            ):
                raise ValueError("engine materialized prefix must be within plan prefix")
            if (
                self.engine_execution.uses_suffix_only_attention
                and self.engine_execution.suffix_start_position != self.prefix_token_count
            ):
                raise ValueError("suffix-only execution must start at the plan prefix")
        if (
            self.kind
            in (
                SemanticKvMaterializationKind.REQUEST_LOCAL_PREFIX,
                SemanticKvMaterializationKind.REQUEST_LOCAL_SEGMENTED,
            )
            and self.publication_policy == SemanticKvPublicationPolicy.EXACT_CACHE
        ):
            raise ValueError("non-identical semantic KV must not be published as exact cache")

    @property
    def is_segmented(self) -> bool:
        return self.kind == SemanticKvMaterializationKind.REQUEST_LOCAL_SEGMENTED

    def with_target_block_ids(self, block_ids: Sequence[int]) -> "SemanticKvPlan":
        target_blocks = tuple(int(block_id) for block_id in block_ids)
        return SemanticKvPlan(
            request_id=self.request_id,
            namespace=self.namespace,
            kind=self.kind,
            publication_policy=self.publication_policy,
            segments=tuple(
                SemanticKvSegment(
                    donor_id=segment.donor_id,
                    donor_segment_id=segment.donor_segment_id,
                    donor_positions=segment.donor_positions,
                    target_positions=segment.target_positions,
                    donor_block_ids=segment.donor_block_ids,
                    target_block_ids=target_blocks,
                    layer_recompute_mask=segment.layer_recompute_mask,
                    donor_location=segment.donor_location,
                    provider_metadata=segment.provider_metadata,
                )
                for segment in self.segments
            ),
            donor_ids=self.donor_ids,
            covered_token_count=self.covered_token_count,
            prefix_token_count=self.prefix_token_count,
            computed_token_count=self.computed_token_count,
            requires_rope_correction=self.requires_rope_correction,
            engine_execution=self.engine_execution,
            diagnostics=dict(self.diagnostics),
        )


@dataclass(frozen=True)
class SemanticKvLookupRequest:
    request_id: int
    token_ids: tuple[int, ...]
    namespace: CacheNamespace
    prompt_text: str = ""
    num_computed_tokens: int = 0
    block_hashes: tuple[int, ...] = ()
    cache_salt: Optional[str] = None
    allow_non_identical: bool = True
    allow_segmented: bool = False
    max_segments: int = 1


@dataclass(frozen=True)
class SemanticKvLookupResult:
    found: bool
    plan: Optional[SemanticKvPlan] = None
    similarity: float = 0.0
    reuse_ratio: float = 0.0
    quality_signals: Mapping[str, Any] = field(default_factory=dict)
    rejection_reason: str = ""
    timings_ms: Mapping[str, float] = field(default_factory=dict)


class SemanticKvProvider(ABC):
    @abstractmethod
    def lookup(self, request: SemanticKvLookupRequest) -> SemanticKvLookupResult:
        pass

    @abstractmethod
    def register_donor(self, event: DonorRegistered) -> None:
        pass

    def evict_donor(self, event: DonorEvicted) -> None:
        return

    def clear(self) -> None:
        return

    def close(self) -> None:
        return


def load_semantic_kv_provider(
    module_name: str,
    class_name: str,
    *args: Any,
    **kwargs: Any,
) -> SemanticKvProvider:
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    provider = cls(*args, **kwargs)
    if not isinstance(provider, SemanticKvProvider):
        raise TypeError(f"{module_name}.{class_name} must implement SemanticKvProvider")
    return provider

# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFlow AI. All rights reserved.
# SPDX-License-Identifier: LicenseRef-WorldFlowAI-Proprietary

"""
KVCacheConnector Protocol — abstract KV cache operations for any inference backend.

See synapse-ia/synapse/docs/kv-cache-connector-spec.md for full specification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class KVMetadata:
    """Describes the KV cache format for this backend."""

    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype: str  # "float16", "bfloat16", "float32"
    block_size: int
    device: str  # "cuda:0", "cpu", "tpu:0", etc.
    max_seq_len: int


@dataclass
class KVLoadResult:
    """Result of loading donor KV into the current request's cache."""

    tokens_loaded: int
    layers_loaded: list[int]
    layers_recompute: list[int]
    load_time_ms: float


@runtime_checkable
class KVCacheConnector(Protocol):
    """Abstract KV cache operations for any inference backend.

    Implementations handle the hardware-specific details of loading donor
    KV blocks, extracting KV from completed requests, and applying RoPE
    position correction.
    """

    def load_donor_kv(
        self,
        donor_kv: object,
        donor_positions: Sequence[int],
        target_positions: Sequence[int],
        layers: Optional[Sequence[int]] = None,
    ) -> KVLoadResult:
        """Load donor KV blocks into the current request's KV cache."""
        ...

    def extract_kv(
        self,
        request_id: str,
        token_range: tuple[int, int],
    ) -> object:
        """Extract KV blocks from a completed request for donor storage."""
        ...

    def apply_rope_correction(
        self,
        k_cache: object,
        source_positions: Sequence[int],
        target_positions: Sequence[int],
        rope_theta: float = 10000.0,
    ) -> object:
        """Apply RoPE position correction to K vectors."""
        ...

    def get_metadata(self) -> KVMetadata:
        """Return backend capabilities and KV cache format info."""
        ...

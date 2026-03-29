# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFlow AI. All rights reserved.
# SPDX-License-Identifier: LicenseRef-WorldFlowAI-Proprietary

"""
CPU KVCacheConnector — KV cache operations using numpy/PyTorch on CPU.

No CUDA, no Triton, no GPU. Works on any CPU: x86_64, ARM (Positron),
Apple Silicon. Uses the same math as the Triton kernels but implemented
with numpy vectorized operations.

Performance: ~10x slower than Triton on GPU per operation, but KV reuse
still saves the same fraction of prefill time. On CPU inference (which is
itself 10-100x slower than GPU), the load overhead is negligible.
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Sequence

import numpy as np

from ..connector_protocol import KVCacheConnector, KVLoadResult, KVMetadata

logger = logging.getLogger(__name__)


class CPUKVCacheConnector:
    """KVCacheConnector for CPU inference backends.

    Stores and loads KV cache as numpy arrays in system RAM.
    RoPE correction uses numpy vectorized operations.

    Args:
        num_layers: Number of transformer layers.
        num_kv_heads: Number of KV attention heads.
        head_dim: Dimension per attention head.
        block_size: KV cache block size in tokens.
        max_seq_len: Maximum sequence length.
        rope_theta: RoPE base frequency (default: 10000.0).
        dtype: numpy dtype for KV tensors (default: float16).
    """

    def __init__(
        self,
        num_layers: int = 28,
        num_kv_heads: int = 4,
        head_dim: int = 128,
        block_size: int = 32,
        max_seq_len: int = 16384,
        rope_theta: float = 10000.0,
        dtype: str = "float16",
    ) -> None:
        self._num_layers = num_layers
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._block_size = block_size
        self._max_seq_len = max_seq_len
        self._rope_theta = rope_theta
        self._dtype = getattr(np, dtype)

        # Pre-compute RoPE frequency table
        self._freqs = 1.0 / (
            rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim)
        )

        # Internal KV cache: [layers, 2(K+V), heads, max_seq, dim]
        self._kv_cache: Optional[np.ndarray] = None

    def _ensure_cache(self) -> np.ndarray:
        """Lazily allocate KV cache."""
        if self._kv_cache is None:
            self._kv_cache = np.zeros(
                (
                    self._num_layers,
                    2,
                    self._num_kv_heads,
                    self._max_seq_len,
                    self._head_dim,
                ),
                dtype=self._dtype,
            )
        return self._kv_cache

    def load_donor_kv(
        self,
        donor_kv: object,
        donor_positions: Sequence[int],
        target_positions: Sequence[int],
        layers: Optional[Sequence[int]] = None,
    ) -> KVLoadResult:
        """Load donor KV into the CPU cache with RoPE correction."""
        t0 = time.monotonic()
        cache = self._ensure_cache()

        donor = np.asarray(donor_kv, dtype=self._dtype)
        d_pos = np.array(donor_positions, dtype=np.int64)
        t_pos = np.array(target_positions, dtype=np.int64)

        layer_indices = list(layers) if layers is not None else list(range(self._num_layers))
        layers_recompute: list[int] = []

        for layer_idx in layer_indices:
            # Copy V directly (no position encoding)
            # Use np.take to avoid numpy's advanced indexing dimension reorder
            v_donor = np.take(donor[layer_idx, 1], d_pos, axis=-2)  # [heads, len(d_pos), dim]
            np.put_along_axis(
                cache[layer_idx, 1],
                t_pos[np.newaxis, :, np.newaxis].repeat(self._num_kv_heads, 0).repeat(self._head_dim, 2),
                v_donor,
                axis=-2,
            )

            # Copy K with RoPE correction
            k_donor = np.take(donor[layer_idx, 0], d_pos, axis=-2).astype(np.float32)
            k_corrected = self._rope_correct(k_donor, d_pos, t_pos)
            np.put_along_axis(
                cache[layer_idx, 0],
                t_pos[np.newaxis, :, np.newaxis].repeat(self._num_kv_heads, 0).repeat(self._head_dim, 2),
                k_corrected.astype(self._dtype),
                axis=-2,
            )

        load_ms = (time.monotonic() - t0) * 1000
        logger.debug(
            "CPU load_donor_kv: %d tokens, %d layers, %.1fms",
            len(d_pos),
            len(layer_indices),
            load_ms,
        )

        return KVLoadResult(
            tokens_loaded=len(d_pos),
            layers_loaded=layer_indices,
            layers_recompute=layers_recompute,
            load_time_ms=load_ms,
        )

    def extract_kv(
        self,
        request_id: str,
        token_range: tuple[int, int],
    ) -> object:
        """Extract KV from the cache as a numpy array."""
        cache = self._ensure_cache()
        start, end = token_range
        # Return a copy so the caller owns the data
        return cache[:, :, :, start:end, :].copy()

    def apply_rope_correction(
        self,
        k_cache: object,
        source_positions: Sequence[int],
        target_positions: Sequence[int],
        rope_theta: float = 10000.0,
    ) -> object:
        """Apply RoPE position correction to K vectors using numpy."""
        k = np.asarray(k_cache, dtype=np.float32)
        s_pos = np.array(source_positions, dtype=np.int64)
        t_pos = np.array(target_positions, dtype=np.int64)
        return self._rope_correct(k, s_pos, t_pos)

    def get_metadata(self) -> KVMetadata:
        """Return CPU backend metadata."""
        return KVMetadata(
            num_layers=self._num_layers,
            num_kv_heads=self._num_kv_heads,
            head_dim=self._head_dim,
            dtype=str(self._dtype.__name__) if hasattr(self._dtype, '__name__') else str(self._dtype),
            block_size=self._block_size,
            device="cpu",
            max_seq_len=self._max_seq_len,
        )

    def _rope_correct(
        self,
        k: np.ndarray,
        source_positions: np.ndarray,
        target_positions: np.ndarray,
    ) -> np.ndarray:
        """Apply RoPE delta correction: rotate K from source to target positions.

        Formula: K_target = RoPE(target) · RoPE_inv(source) · K_source
               = RoPE(target - source) · K_source

        Args:
            k: K tensor with last two dims [seq_len, head_dim] in float32.
               Can be [heads, seq, dim] or [seq, dim] or any prefix dims.
            source_positions: [seq_len] original positions.
            target_positions: [seq_len] desired positions.

        Returns:
            Corrected K tensor (same shape).
        """
        if len(source_positions) == 0:
            return k

        delta = (target_positions - source_positions).astype(np.float32)

        # angles[i, j] = delta[i] * freqs[j]  shape: [seq_len, head_dim/2]
        angles = np.outer(delta, self._freqs)

        cos_delta = np.cos(angles)
        sin_delta = np.sin(angles)

        # Reshape for broadcasting: need cos/sin to align with last two dims
        # k shape: [..., seq_len, head_dim] — cos/sin need shape [..., seq_len, head_dim/2]
        # Add leading dims to match k's prefix dimensions
        n_prefix = k.ndim - 2  # number of dims before (seq_len, head_dim)
        for _ in range(n_prefix):
            cos_delta = np.expand_dims(cos_delta, 0)
            sin_delta = np.expand_dims(sin_delta, 0)

        # Apply rotation to interleaved pairs
        k_even = k[..., 0::2].copy()
        k_odd = k[..., 1::2].copy()

        k[..., 0::2] = k_even * cos_delta - k_odd * sin_delta
        k[..., 1::2] = k_even * sin_delta + k_odd * cos_delta

        return k

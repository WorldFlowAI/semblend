"""Per-layer selective recomputation schedule with exponential decay.

The key SemShareKV innovation: instead of binary per-layer decisions (SemBlend's
bathtub), compute a *fraction* of tokens to recompute per layer. HD tokens are
weighted higher (omega_h=0.5) vs cold tokens (omega_c=0.1).

Layer 0: use donor KV directly (no recompute)
Layer 1: full recompute (alpha=1.0) — used for HD detection
Layers 2..L: alpha decays exponentially, HD tokens prioritized
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from semblend_core.semshare.config import SemShareConfig
from semblend_core.semshare.hd_detector import HDDetectionResult


@dataclass(frozen=True)
class LayerSchedule:
    """Per-layer recomputation schedule."""

    # Alpha (recompute fraction) per layer, shape [num_layers]
    per_layer_alpha: tuple[float, ...]

    # Boolean recompute masks per layer, each shape [seq_len]
    # True = recompute this token at this layer
    per_layer_masks: tuple[tuple[bool, ...], ...]

    # Number of layers
    num_layers: int

    # Total tokens recomputed across all layers
    total_recomputed: int

    # Total possible (num_layers * seq_len)
    total_possible: int

    @property
    def recompute_ratio(self) -> float:
        return self.total_recomputed / self.total_possible if self.total_possible > 0 else 0.0


def compute_layer_schedule(
    hd_result: HDDetectionResult,
    num_layers: int,
    config: SemShareConfig,
    unmatched_positions: tuple[int, ...] = (),
) -> LayerSchedule:
    """Compute per-layer recompute masks from HD detection result.

    Args:
        hd_result: HD detection from layer 1
        num_layers: total transformer layers
        config: SemShareKV configuration
        unmatched_positions: target positions with no donor match (always recompute)

    Returns:
        LayerSchedule with per-layer alpha values and boolean masks
    """
    seq_len = hd_result.seq_len
    hd_mask = np.array(hd_result.hd_mask, dtype=bool)
    unmatched_set = set(unmatched_positions)

    # Compute per-token recompute priority scores
    # HD tokens get higher priority, cold tokens lower
    priority = np.where(hd_mask, config.omega_h, config.omega_c)

    # Unmatched tokens always get max priority (must recompute)
    for pos in unmatched_positions:
        if pos < seq_len:
            priority[pos] = 1.0

    # Sort token indices by priority (descending) for mask generation
    sorted_indices = np.argsort(-priority)

    # Compute per-layer alpha schedule
    alphas: list[float] = []
    masks: list[tuple[bool, ...]] = []
    total_recomputed = 0

    for layer_idx in range(num_layers):
        if layer_idx == 0:
            # Layer 0: no recompute (use donor KV)
            alpha = 0.0
            mask = tuple(pos in unmatched_set for pos in range(seq_len))
        elif layer_idx == 1:
            # Layer 1: full recompute (for HD detection)
            alpha = 1.0
            mask = tuple(True for _ in range(seq_len))
        else:
            # Layers 2+: exponential decay
            alpha = config.alpha_base * (config.alpha_decay ** (layer_idx - 2))
            alpha = max(alpha, 0.01)  # floor to ensure some recomputation

            # Select top-alpha fraction of tokens by priority
            num_recompute = max(1, int(seq_len * alpha))
            # Always include unmatched positions
            num_recompute = max(num_recompute, len(unmatched_positions))

            layer_mask = np.zeros(seq_len, dtype=bool)
            for idx in sorted_indices[:num_recompute]:
                layer_mask[idx] = True
            # Force unmatched positions
            for pos in unmatched_positions:
                if pos < seq_len:
                    layer_mask[pos] = True

            mask = tuple(bool(x) for x in layer_mask)

        alphas.append(alpha)
        masks.append(mask)
        total_recomputed += sum(mask)

    return LayerSchedule(
        per_layer_alpha=tuple(alphas),
        per_layer_masks=tuple(masks),
        num_layers=num_layers,
        total_recomputed=total_recomputed,
        total_possible=num_layers * seq_len,
    )

"""PartialAttention scaffold for selective KV recomputation.

Implements the interface for KVShare-style PartialAttention: compute
Q,K,V only for placeholder tokens (new/replaced), while reusing
cached K,V from donor tokens. The cross-attention step correctly
contextualizes new tokens against cached context.

This module provides the data structures and mask computation logic.
The actual CUDA kernel integration requires vLLM fork modifications
and is marked as Phase 3 work.

References:
    - KVShare (arXiv:2503.16525) §4.3 — PartialAttention mechanism
    - CacheBlend (EuroSys 2025) — layer-level selective recomputation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class AttentionMode(Enum):
    """Attention computation mode for a position or layer."""

    FULL = "full"
    """Compute full Q,K,V — standard prefill."""

    REUSE = "reuse"
    """Reuse cached K,V from donor — no computation needed."""

    PARTIAL = "partial"
    """Compute Q,K,V for this position but cross-attend to all cached K,V."""

    SKIP_LAYER = "skip_layer"
    """Skip this layer entirely — CacheBlend fallback when layer
    deviation is below threshold."""


@dataclass(frozen=True)
class PositionMask:
    """Attention mask for a single token position.

    Describes whether this position reuses donor KV, needs fresh
    computation with cross-attention, or uses full attention.
    """

    target_pos: int
    mode: AttentionMode
    donor_pos: int | None = None


@dataclass(frozen=True)
class LayerMask:
    """Attention strategy for a single transformer layer.

    Either the entire layer uses full recomputation (CacheBlend style)
    or per-position partial attention (KVShare style).
    """

    layer_idx: int
    recompute_all: bool
    deviation_score: float
    position_masks: list[PositionMask] = field(default_factory=list)


@dataclass(frozen=True)
class PartialAttentionPlan:
    """Complete per-layer, per-position attention plan.

    This is the output consumed by the vLLM attention kernel integration.
    It tells the engine exactly which positions need fresh Q,K,V
    computation and which can reuse donor K,V vectors.

    Attributes:
        target_len: Number of tokens in the target sequence.
        donor_len: Number of tokens in the donor sequence.
        donor_id: Identifier for the donor cache entry.
        layer_masks: Per-layer attention strategy.
        num_reuse_positions: Count of positions that reuse donor KV.
        num_partial_positions: Count of positions needing partial computation.
        num_full_layers: Count of layers needing full recomputation.
        computation_ratio: Fraction of total work compared to full prefill.
    """

    target_len: int
    donor_len: int
    donor_id: str
    layer_masks: list[LayerMask]
    num_reuse_positions: int
    num_partial_positions: int
    num_full_layers: int
    computation_ratio: float


def build_attention_plan(
    donor_id: str,
    target_len: int,
    donor_len: int,
    copy_positions: list[int],
    placeholder_positions: list[int],
    slot_actions: list[dict],
    layer_hints: list[dict] | None = None,
    num_layers: int = 32,
) -> PartialAttentionPlan:
    """Build a PartialAttentionPlan from a KV transfer plan.

    Converts the transfer plan's slot actions and layer hints into
    per-layer, per-position attention masks suitable for the vLLM
    attention kernel.

    Args:
        donor_id: Donor cache entry identifier.
        target_len: Number of tokens in the target sequence.
        donor_len: Number of tokens in the donor sequence.
        copy_positions: Target positions that copy from donor.
        placeholder_positions: Target positions needing computation.
        slot_actions: Per-position actions from the transfer plan.
        layer_hints: Optional per-layer recomputation hints.
        num_layers: Number of transformer layers.

    Returns:
        PartialAttentionPlan with per-layer, per-position masks.
    """
    # Build per-position masks from slot actions
    position_masks = []
    for sa in slot_actions:
        action = sa.get("action", "")
        target_pos = sa.get("targetPos", sa.get("target_pos", 0))

        if action == "copy_from_donor":
            donor_pos = sa.get("donorPos", sa.get("donor_pos"))
            position_masks.append(
                PositionMask(
                    target_pos=target_pos,
                    mode=AttentionMode.REUSE,
                    donor_pos=donor_pos,
                )
            )
        elif action == "placeholder":
            position_masks.append(
                PositionMask(
                    target_pos=target_pos,
                    mode=AttentionMode.PARTIAL,
                    donor_pos=None,
                )
            )

    # Build per-layer masks
    layer_masks = []
    num_full_layers = 0

    for layer_idx in range(num_layers):
        recompute_all = False
        deviation_score = 0.0

        if layer_hints and layer_idx < len(layer_hints):
            hint = layer_hints[layer_idx]
            recompute_all = hint.get(
                "recomputeAll", hint.get("recompute_all", False)
            )
            deviation_score = hint.get(
                "deviationScore", hint.get("deviation_score", 0.0)
            )

        if recompute_all:
            num_full_layers += 1
            # Full layer recompute — all positions use FULL mode
            full_masks = [
                PositionMask(
                    target_pos=i,
                    mode=AttentionMode.FULL,
                )
                for i in range(target_len)
            ]
            layer_masks.append(
                LayerMask(
                    layer_idx=layer_idx,
                    recompute_all=True,
                    deviation_score=deviation_score,
                    position_masks=full_masks,
                )
            )
        else:
            # Per-position partial attention
            layer_masks.append(
                LayerMask(
                    layer_idx=layer_idx,
                    recompute_all=False,
                    deviation_score=deviation_score,
                    position_masks=list(position_masks),
                )
            )

    num_reuse = len(copy_positions)
    num_partial = len(placeholder_positions)

    # Computation ratio: what fraction of total FLOPs is needed
    # compared to full prefill
    #
    # For non-recompute layers: only placeholder positions need Q,K,V
    # For recompute layers: all positions need Q,K,V
    partial_layers = num_layers - num_full_layers
    if num_layers > 0 and target_len > 0:
        # Partial layers: only placeholder tokens computed
        partial_work = partial_layers * num_partial
        # Full layers: all tokens computed
        full_work = num_full_layers * target_len
        total_possible = num_layers * target_len
        computation_ratio = (
            (partial_work + full_work) / total_possible
            if total_possible > 0
            else 1.0
        )
    else:
        computation_ratio = 1.0

    return PartialAttentionPlan(
        target_len=target_len,
        donor_len=donor_len,
        donor_id=donor_id,
        layer_masks=layer_masks,
        num_reuse_positions=num_reuse,
        num_partial_positions=num_partial,
        num_full_layers=num_full_layers,
        computation_ratio=computation_ratio,
    )


def compute_attention_mask(
    plan: PartialAttentionPlan,
    layer_idx: int,
) -> np.ndarray:
    """Compute a boolean attention mask for a specific layer.

    Returns a 1D boolean array of shape [target_len] where:
    - True = this position needs Q,K,V computation
    - False = this position reuses donor K,V (no computation)

    For layers flagged for full recomputation, all positions are True.

    Args:
        plan: The PartialAttentionPlan.
        layer_idx: The transformer layer index (0-based).

    Returns:
        Boolean numpy array of shape [target_len].
    """
    mask = np.zeros(plan.target_len, dtype=bool)

    if layer_idx >= len(plan.layer_masks):
        # No mask info — default to full computation
        mask[:] = True
        return mask

    layer_mask = plan.layer_masks[layer_idx]

    if layer_mask.recompute_all:
        mask[:] = True
        return mask

    for pm in layer_mask.position_masks:
        if pm.mode in (AttentionMode.PARTIAL, AttentionMode.FULL):
            if pm.target_pos < plan.target_len:
                mask[pm.target_pos] = True

    return mask


def compute_donor_kv_indices(
    plan: PartialAttentionPlan,
    layer_idx: int,
) -> list[tuple[int, int]]:
    """Compute (donor_pos, target_pos) pairs for KV copy operations.

    Returns the list of position pairs where donor K,V should be
    copied to target K,V cache. Only includes positions in REUSE mode
    for the given layer.

    Args:
        plan: The PartialAttentionPlan.
        layer_idx: The transformer layer index.

    Returns:
        List of (donor_pos, target_pos) tuples for KV copy.
    """
    if layer_idx >= len(plan.layer_masks):
        return []

    layer_mask = plan.layer_masks[layer_idx]

    if layer_mask.recompute_all:
        return []

    pairs = []
    for pm in layer_mask.position_masks:
        if pm.mode == AttentionMode.REUSE and pm.donor_pos is not None:
            pairs.append((pm.donor_pos, pm.target_pos))

    return pairs

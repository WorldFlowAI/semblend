"""Bridge between SemShareKV layer schedule and PartialAttentionPlan.

Converts the per-layer graduated recompute schedule into the existing
PartialAttentionPlan format used by the vLLM connector/hook. This allows
SemShareKV's selective recomputation to work with the existing infrastructure
without any connector changes.

Feature flag: SEMBLEND_SELECTIVE_RECOMPUTE=1 (enhances existing pipeline)
"""
from __future__ import annotations

import logging

from semblend_core.partial_attention import (
    AttentionMode,
    LayerMask,
    PartialAttentionPlan,
    PositionMask,
)
from semblend_core.semshare.hd_detector import HDDetectionResult
from semblend_core.semshare.layer_schedule import LayerSchedule

logger = logging.getLogger(__name__)


def build_selective_attention_plan(
    donor_id: str,
    target_len: int,
    donor_len: int,
    schedule: LayerSchedule,
    copy_positions: list[int],
    placeholder_positions: list[int],
    slot_actions: list[dict],
    hd_result: HDDetectionResult | None = None,
) -> PartialAttentionPlan:
    """Build a PartialAttentionPlan from SemShareKV's graduated layer schedule.

    Instead of bathtub's binary per-layer decision, each layer gets a
    per-position mask derived from the exponential decay schedule + HD priority.

    This plugs directly into the existing vLLM connector — the connector
    reads PartialAttentionPlan the same way regardless of whether it came
    from bathtub or selective recompute.

    Args:
        donor_id: Donor cache identifier.
        target_len: Target sequence length.
        donor_len: Donor sequence length.
        schedule: Per-layer recompute schedule from compute_layer_schedule().
        copy_positions: Target positions that copy from donor.
        placeholder_positions: Target positions always needing computation.
        slot_actions: Per-position actions from chunk alignment.
        hd_result: Optional HD detection result for logging/metrics.

    Returns:
        PartialAttentionPlan with per-layer per-position masks.
    """
    # Build donor_pos map from slot actions
    donor_pos_map: dict[int, int] = {}
    for sa in slot_actions:
        action = sa.get("action", "")
        target_pos = sa.get("targetPos", sa.get("target_pos", 0))
        if action == "copy_from_donor":
            donor_pos = sa.get("donorPos", sa.get("donor_pos"))
            donor_pos_map[target_pos] = donor_pos

    placeholder_set = set(placeholder_positions)
    copy_set = set(copy_positions)

    layer_masks: list[LayerMask] = []
    num_full_layers = 0
    total_recomputed = 0

    for layer_idx in range(schedule.num_layers):
        layer_schedule_mask = schedule.per_layer_masks[layer_idx]
        alpha = schedule.per_layer_alpha[layer_idx]

        if alpha >= 0.99:
            # Full recompute layer (layer 1 in SemShareKV, or forced layers)
            num_full_layers += 1
            position_masks = [
                PositionMask(
                    target_pos=i,
                    mode=AttentionMode.FULL,
                )
                for i in range(target_len)
            ]
            total_recomputed += target_len
            layer_masks.append(
                LayerMask(
                    layer_idx=layer_idx,
                    recompute_all=True,
                    deviation_score=alpha,
                    position_masks=position_masks,
                    verification_source="selective_recompute",
                )
            )
        elif alpha <= 0.01 and not placeholder_set:
            # Near-zero recompute — reuse everything (except placeholders)
            position_masks = [
                PositionMask(
                    target_pos=i,
                    mode=AttentionMode.REUSE,
                    donor_pos=donor_pos_map.get(i),
                )
                if i in copy_set
                else PositionMask(
                    target_pos=i,
                    mode=AttentionMode.PARTIAL,
                )
                for i in range(target_len)
            ]
            layer_masks.append(
                LayerMask(
                    layer_idx=layer_idx,
                    recompute_all=False,
                    deviation_score=alpha,
                    position_masks=position_masks,
                    verification_source="selective_recompute",
                )
            )
        else:
            # Graduated: per-token decision from schedule mask
            position_masks = []
            layer_recomputed = 0

            for i in range(target_len):
                should_recompute = (
                    layer_schedule_mask[i] if i < len(layer_schedule_mask) else True
                )

                if i in placeholder_set or should_recompute:
                    # Recompute this token
                    position_masks.append(
                        PositionMask(
                            target_pos=i,
                            mode=AttentionMode.PARTIAL,
                        )
                    )
                    layer_recomputed += 1
                else:
                    # Reuse donor KV
                    position_masks.append(
                        PositionMask(
                            target_pos=i,
                            mode=AttentionMode.REUSE,
                            donor_pos=donor_pos_map.get(i),
                        )
                    )

            total_recomputed += layer_recomputed
            layer_masks.append(
                LayerMask(
                    layer_idx=layer_idx,
                    recompute_all=False,
                    deviation_score=alpha,
                    position_masks=position_masks,
                    verification_source="selective_recompute",
                )
            )

    # Computation ratio
    total_possible = schedule.num_layers * target_len
    computation_ratio = (
        total_recomputed / total_possible if total_possible > 0 else 1.0
    )

    num_reuse = len(copy_positions)
    num_partial = len(placeholder_positions)

    hd_count = hd_result.num_hd_tokens if hd_result else 0
    logger.info(
        "Selective recompute plan: %d layers, alpha range [%.2f, %.2f], "
        "computation_ratio=%.2f, hd_tokens=%d/%d, full_layers=%d",
        schedule.num_layers,
        min(schedule.per_layer_alpha),
        max(schedule.per_layer_alpha),
        computation_ratio,
        hd_count,
        target_len,
        num_full_layers,
    )

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


def enhance_plan_with_selective_recompute(
    existing_plan: PartialAttentionPlan,
    schedule: LayerSchedule,
    hd_result: HDDetectionResult | None = None,
) -> PartialAttentionPlan:
    """Enhance an existing bathtub-generated plan with selective recompute.

    For layers that bathtub marked as REUSE (no recompute), apply the
    graduated schedule to selectively recompute HD tokens. For layers
    bathtub already marked as full recompute, keep them as-is.

    This is the primary integration point — it takes the existing plan
    from the chunk pipeline and upgrades it with per-token granularity.

    Args:
        existing_plan: Plan from build_attention_plan() (bathtub-based).
        schedule: Per-layer schedule from compute_layer_schedule().
        hd_result: HD detection result for HD-aware token selection.

    Returns:
        Enhanced PartialAttentionPlan with per-token graduated recompute.
    """
    enhanced_masks: list[LayerMask] = []
    num_full_layers = 0
    total_recomputed = 0

    for layer_mask in existing_plan.layer_masks:
        layer_idx = layer_mask.layer_idx

        if layer_mask.recompute_all:
            # Bathtub says full recompute — keep it
            enhanced_masks.append(layer_mask)
            num_full_layers += 1
            total_recomputed += existing_plan.target_len
            continue

        if layer_idx >= schedule.num_layers:
            enhanced_masks.append(layer_mask)
            continue

        # Bathtub says REUSE this layer. Apply graduated schedule
        # to selectively recompute HD tokens.
        layer_schedule_mask = schedule.per_layer_masks[layer_idx]
        alpha = schedule.per_layer_alpha[layer_idx]

        if alpha < 0.01:
            # Schedule says skip — keep bathtub's decision
            enhanced_masks.append(layer_mask)
            continue

        # Upgrade: recompute tokens the schedule flags
        new_position_masks: list[PositionMask] = []
        layer_recomputed = 0

        for pm in layer_mask.position_masks:
            pos = pm.target_pos
            should_recompute = (
                layer_schedule_mask[pos] if pos < len(layer_schedule_mask) else False
            )

            if pm.mode == AttentionMode.PARTIAL or pm.mode == AttentionMode.FULL:
                # Already being recomputed — keep it
                new_position_masks.append(pm)
                layer_recomputed += 1
            elif should_recompute:
                # Schedule says recompute this token
                new_position_masks.append(
                    PositionMask(
                        target_pos=pos,
                        mode=AttentionMode.PARTIAL,
                    )
                )
                layer_recomputed += 1
            else:
                # Keep as REUSE
                new_position_masks.append(pm)

        total_recomputed += layer_recomputed
        enhanced_masks.append(
            LayerMask(
                layer_idx=layer_idx,
                recompute_all=False,
                deviation_score=alpha,
                position_masks=new_position_masks,
                verification_source="selective_recompute",
            )
        )

    total_possible = len(enhanced_masks) * existing_plan.target_len
    computation_ratio = (
        total_recomputed / total_possible if total_possible > 0 else 1.0
    )

    return PartialAttentionPlan(
        target_len=existing_plan.target_len,
        donor_len=existing_plan.donor_len,
        donor_id=existing_plan.donor_id,
        layer_masks=enhanced_masks,
        num_reuse_positions=existing_plan.num_reuse_positions,
        num_partial_positions=existing_plan.num_partial_positions,
        num_full_layers=num_full_layers,
        computation_ratio=computation_ratio,
        donor_map=existing_plan.donor_map,
    )

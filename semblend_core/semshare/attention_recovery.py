"""Attention Recovery (AR) metric from SemShareKV.

AR = min{k | sum(top-k attention) / total_attention >= threshold}

Measures how many tokens are needed to cover threshold% of attention mass.
Lower AR means attention is more concentrated → safer to share/evict tokens.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AttentionRecoveryResult:
    """Result of Attention Recovery computation."""

    # Minimum k tokens to reach threshold
    ar_value: int

    # Actual attention coverage at ar_value
    ar_ratio: float

    # Whether threshold was met
    threshold_met: bool

    # Total tokens in sequence
    seq_len: int

    # AR as fraction of sequence length
    ar_fraction: float


def compute_attention_recovery(
    attention_weights: np.ndarray,
    threshold: float = 0.90,
) -> AttentionRecoveryResult:
    """Compute Attention Recovery metric.

    Args:
        attention_weights: shape [seq_len] — averaged attention scores per token
            (typically averaged across heads and query positions)
        threshold: coverage threshold (default 0.90 = 90%)

    Returns:
        AttentionRecoveryResult
    """
    seq_len = len(attention_weights)
    if seq_len == 0:
        return AttentionRecoveryResult(
            ar_value=0,
            ar_ratio=0.0,
            threshold_met=False,
            seq_len=0,
            ar_fraction=0.0,
        )

    total = float(np.sum(attention_weights))
    if total <= 0:
        return AttentionRecoveryResult(
            ar_value=seq_len,
            ar_ratio=0.0,
            threshold_met=False,
            seq_len=seq_len,
            ar_fraction=1.0,
        )

    # Sort descending
    sorted_weights = np.sort(attention_weights)[::-1]
    cumulative = np.cumsum(sorted_weights) / total

    # Find minimum k where cumulative >= threshold
    indices = np.where(cumulative >= threshold)[0]
    if len(indices) == 0:
        ar_value = seq_len
        ar_ratio = float(cumulative[-1]) if seq_len > 0 else 0.0
        threshold_met = False
    else:
        ar_value = int(indices[0]) + 1  # 1-indexed
        ar_ratio = float(cumulative[indices[0]])
        threshold_met = True

    return AttentionRecoveryResult(
        ar_value=ar_value,
        ar_ratio=ar_ratio,
        threshold_met=threshold_met,
        seq_len=seq_len,
        ar_fraction=ar_value / seq_len if seq_len > 0 else 0.0,
    )


def compute_per_layer_ar(
    attention_weights_per_layer: list[np.ndarray],
    threshold: float = 0.90,
) -> list[AttentionRecoveryResult]:
    """Compute AR for each transformer layer.

    The paper observes AR decreases in deeper layers (more sparse attention),
    validating that deeper layers tolerate more cache sharing.

    Args:
        attention_weights_per_layer: list of [seq_len] arrays, one per layer
        threshold: coverage threshold

    Returns:
        List of AttentionRecoveryResult, one per layer
    """
    return [
        compute_attention_recovery(weights, threshold)
        for weights in attention_weights_per_layer
    ]

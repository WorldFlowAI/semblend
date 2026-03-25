"""High Deviation (HD) token detection via L2 norm of KV difference.

After layer 1 full recomputation, compare recomputed KV against donor KV.
Tokens in the top hd_fraction (40%) by L2 deviation are marked as HD tokens
and prioritized for recomputation in subsequent layers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from semblend_core.semshare.config import SemShareConfig


@dataclass(frozen=True)
class HDDetectionResult:
    """Result of High Deviation token detection."""

    # Boolean mask: True = HD token, shape [seq_len]
    hd_mask: tuple[bool, ...]

    # Per-token deviation scores (L2 norm), shape [seq_len]
    deviation_scores: tuple[float, ...]

    # L2 threshold used to separate HD from non-HD
    hd_threshold: float

    # Fraction of tokens flagged as HD
    hd_fraction: float

    @property
    def num_hd_tokens(self) -> int:
        return sum(self.hd_mask)

    @property
    def seq_len(self) -> int:
        return len(self.hd_mask)


def detect_hd_tokens(
    recomputed_k: np.ndarray,
    recomputed_v: np.ndarray,
    donor_k: np.ndarray,
    donor_v: np.ndarray,
    config: SemShareConfig,
) -> HDDetectionResult:
    """Detect High Deviation tokens from layer 1 KV comparison.

    Args:
        recomputed_k: shape [seq_len, head_dim] — K from full recompute of layer 1
        recomputed_v: shape [seq_len, head_dim] — V from full recompute of layer 1
        donor_k: shape [seq_len, head_dim] — K from donor (rearranged to target order)
        donor_v: shape [seq_len, head_dim] — V from donor (rearranged to target order)
        config: SemShareKV configuration

    Returns:
        HDDetectionResult with mask and deviation scores
    """
    seq_len = recomputed_k.shape[0]

    # Compute per-token L2 deviation: sigma_K + sigma_V
    k_diff = np.linalg.norm(recomputed_k - donor_k, axis=-1)  # [seq_len]
    v_diff = np.linalg.norm(recomputed_v - donor_v, axis=-1)  # [seq_len]
    deviation = k_diff + v_diff  # [seq_len]

    # Find threshold: top hd_fraction are HD tokens
    num_hd = max(1, int(seq_len * config.hd_fraction))
    sorted_deviations = np.sort(deviation)[::-1]
    threshold = float(sorted_deviations[min(num_hd - 1, seq_len - 1)])

    # Mark tokens above threshold as HD
    hd_mask = deviation >= threshold

    # Ensure exactly the right fraction (handle ties at boundary)
    actual_hd = int(np.sum(hd_mask))
    if actual_hd > num_hd:
        # Too many at threshold — keep only the first num_hd
        indices_at_threshold = np.where(deviation == threshold)[0]
        excess = actual_hd - num_hd
        for idx in indices_at_threshold[-excess:]:
            hd_mask[idx] = False

    return HDDetectionResult(
        hd_mask=tuple(bool(x) for x in hd_mask),
        deviation_scores=tuple(float(x) for x in deviation),
        hd_threshold=threshold,
        hd_fraction=float(np.sum(hd_mask)) / seq_len if seq_len > 0 else 0.0,
    )

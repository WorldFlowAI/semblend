"""Calibrated bathtub curve for per-layer KV deviation prediction.

Implements SemBlend paper Eq. 4:
    sigma(l, L) = sigma_base + sigma_e * exp(-l/tau_e) + sigma_l * exp(-(L-l)/tau_l)

Early and late transformer layers deviate most between semantically similar prompts,
while middle layers are stable and shareable.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class LayerDeviation:
    """Predicted deviation for a single transformer layer."""

    layer_idx: int
    deviation_score: float
    should_recompute: bool


@dataclass(frozen=True)
class BathtubPreset:
    """Per-model bathtub curve parameters."""

    num_layers: int
    sigma_base: float
    sigma_e: float  # early-layer amplitude
    tau_e: float  # early-layer decay
    sigma_l: float  # late-layer amplitude
    tau_l: float  # late-layer decay
    position_tau: float = 128.0
    fuzzy_alpha: float = 0.15


@dataclass(frozen=True)
class RecomputeConfig:
    """Configurable layer recomputation settings for any model/engine/deployment."""

    threshold: float | None = None
    adaptive_base: float = 0.3
    adaptive_scale: float = 0.4
    force_recompute_layers: tuple[int, ...] | None = None
    skip_recompute_layers: tuple[int, ...] | None = None
    max_recompute_fraction: float = 0.25
    fuzzy_alpha: float = 0.15
    confidence_penalty_scale: float = 0.15
    force_verify_on_fuzzy: bool = True

    @staticmethod
    def from_env() -> "RecomputeConfig":
        """Build config from environment variables."""
        threshold_str = os.environ.get("SEMBLEND_RECOMPUTE_THRESHOLD", "")
        force_layers_str = os.environ.get("SEMBLEND_FORCE_RECOMPUTE_LAYERS", "")
        skip_layers_str = os.environ.get("SEMBLEND_SKIP_RECOMPUTE_LAYERS", "")

        threshold = float(threshold_str) if threshold_str else None
        force_layers = (
            tuple(int(x) for x in force_layers_str.split(",") if x.strip())
            if force_layers_str
            else None
        )
        skip_layers = (
            tuple(int(x) for x in skip_layers_str.split(",") if x.strip())
            if skip_layers_str
            else None
        )

        return RecomputeConfig(
            threshold=threshold,
            adaptive_base=float(os.environ.get("SEMBLEND_RECOMPUTE_ADAPTIVE_BASE", "0.3")),
            adaptive_scale=float(os.environ.get("SEMBLEND_RECOMPUTE_ADAPTIVE_SCALE", "0.4")),
            force_recompute_layers=force_layers,
            skip_recompute_layers=skip_layers,
            max_recompute_fraction=float(os.environ.get("SEMBLEND_MAX_RECOMPUTE_FRACTION", "0.25")),
            fuzzy_alpha=float(os.environ.get("SEMBLEND_FUZZY_ALPHA", "0.15")),
            confidence_penalty_scale=float(
                os.environ.get("SEMBLEND_CONFIDENCE_PENALTY_SCALE", "0.15")
            ),
            force_verify_on_fuzzy=os.environ.get("SEMBLEND_FORCE_VERIFY_FUZZY", "1") == "1",
        )


# Per-model presets calibrated from empirical data and kv-cache-physics findings.
#
# Architecture-specific tuning per kv-cache-physics (arXiv:2603.01426):
#   - LLaMA family: "inverted funnel" — early layers consolidate information;
#     middle/late layers more redundant → prioritize early-layer recomputation.
#   - Qwen family: "funnel" — late layers consolidate; early layers more redundant
#     → prioritize late-layer recomputation.
#
# Current calibration produces:
#   LLaMA (32L): recompute [0,1,2] = 9.4%  (sigma_e=0.45 > sigma_l=0.15)
#   Qwen (28L):  recompute [0,26,27] = 10.7% (sigma_l=0.35 > sigma_e=0.15)
PRESETS: dict[str, BathtubPreset] = {
    # Qwen family: funnel pattern — late layers critical
    "qwen2.5-1.5b": BathtubPreset(
        num_layers=28,
        sigma_base=0.12,
        sigma_e=0.15,
        tau_e=3.0,
        sigma_l=0.35,
        tau_l=3.0,
        position_tau=192.0,
        fuzzy_alpha=0.15,
    ),
    "qwen2.5-7b": BathtubPreset(
        num_layers=28,
        sigma_base=0.12,
        sigma_e=0.15,
        tau_e=3.0,
        sigma_l=0.35,
        tau_l=3.0,
        position_tau=192.0,
        fuzzy_alpha=0.15,
    ),
    # LLaMA family: inverted funnel — early layers critical
    "llama": BathtubPreset(
        num_layers=32,
        sigma_base=0.12,
        sigma_e=0.45,
        tau_e=2.5,
        sigma_l=0.15,
        tau_l=4.0,
        position_tau=128.0,
        fuzzy_alpha=0.12,
    ),
    "default": BathtubPreset(
        num_layers=32,
        sigma_base=0.12,
        sigma_e=0.35,
        tau_e=3.0,
        sigma_l=0.20,
        tau_l=4.0,
        position_tau=128.0,
        fuzzy_alpha=0.15,
    ),
}


def sigma(
    layer_idx: int,
    num_layers: int,
    sigma_base: float = 0.12,
    sigma_e: float = 0.35,
    tau_e: float = 3.0,
    sigma_l: float = 0.20,
    tau_l: float = 4.0,
) -> float:
    """Compute bathtub deviation score for a single layer.

    SemBlend paper Eq. 4:
        sigma(l, L) = sigma_base + sigma_e * exp(-l/tau_e)
                     + sigma_l * exp(-(L-l)/tau_l)

    Args:
        layer_idx: Layer index (0-based).
        num_layers: Total number of layers.
        sigma_base: Baseline deviation (middle layers).
        sigma_e: Early-layer deviation amplitude.
        tau_e: Early-layer decay constant.
        sigma_l: Late-layer deviation amplitude.
        tau_l: Late-layer decay constant.

    Returns:
        Deviation score in [0, 1].
    """
    layer = layer_idx
    big_l = num_layers - 1

    early = sigma_e * math.exp(-layer / tau_e) if tau_e > 0 else 0.0
    late = sigma_l * math.exp(-(big_l - layer) / tau_l) if tau_l > 0 else 0.0

    return min(sigma_base + early + late, 1.0)


def adaptive_threshold(
    similarity: float,
    base: float = 0.3,
    scale: float = 0.4,
) -> float:
    """Compute similarity-adaptive recomputation threshold.

    Higher similarity → higher threshold → fewer layers exceed it → less
    recomputation → more speedup (quality insurance relaxed for similar donors).
    Lower similarity → lower threshold → more layers exceed it → more
    recomputation → better quality (stricter quality insurance for dissimilar donors).

    Formula: threshold = base + similarity * scale

    At similarity=0.0: threshold = base (0.3 — most recomputation)
    At similarity=0.6: threshold = base + 0.6*0.4 = 0.54 (moderate)
    At similarity=1.0: threshold = base + scale (0.7 — minimal recomputation)

    Args:
        similarity: Cosine similarity between donor and target (0.0-1.0).
        base: Minimum threshold at zero similarity.
        scale: Additional threshold range for high similarity.

    Returns:
        Recomputation threshold in [base, base+scale].
    """
    clamped = max(0.0, min(1.0, similarity))
    return base + clamped * scale


def position_factor(
    layer_idx: int,
    num_layers: int,
    model_name: str | None = None,
) -> float:
    """Position-shift sensitivity per layer.

    Models have different sensitivity patterns to position shifts:
      - Qwen (funnel): late layers more position-sensitive
      - LLaMA (inverted): early layers more position-sensitive
    """
    normalized = layer_idx / max(num_layers - 1, 1)
    name_lower = (model_name or "").lower()

    if "qwen" in name_lower:
        return 0.5 + 0.5 * normalized
    elif "llama" in name_lower:
        return 1.0 - 0.3 * normalized
    else:
        return 0.75


def compute_layer_deviations(
    num_layers: int,
    mismatch_fraction: float = 0.0,
    threshold: float = 0.3,
    model_name: str | None = None,
    similarity: float | None = None,
    fuzzy_fraction: float = 0.0,
    mean_fuzzy_confidence: float = 1.0,
    recompute_config: RecomputeConfig | None = None,
) -> list[LayerDeviation]:
    """Compute per-layer deviation predictions using the bathtub curve.

    Args:
        num_layers: Number of transformer layers.
        mismatch_fraction: Fraction of tokens that differ (scales deviations).
        threshold: Deviation threshold above which to recompute a layer.
            Ignored when similarity is provided (adaptive threshold used instead).
        model_name: Optional model name for preset lookup.
        similarity: Optional cosine similarity for adaptive thresholding.
            When provided, overrides the fixed threshold with
            adaptive_threshold(similarity).
        fuzzy_fraction: Fraction of matched tokens that were fuzzy-matched.
        mean_fuzzy_confidence: Average confidence of fuzzy-matched tokens.
        recompute_config: Optional RecomputeConfig for layer recomputation tuning.

    Returns:
        Per-layer deviation predictions.
    """
    preset = get_preset(model_name) if model_name else PRESETS["default"]
    config = recompute_config or RecomputeConfig()

    # Determine effective threshold
    if config.threshold is not None:
        effective_threshold = config.threshold
    elif similarity is not None:
        effective_threshold = adaptive_threshold(
            similarity,
            base=config.adaptive_base,
            scale=config.adaptive_scale,
        )
    else:
        effective_threshold = threshold

    # Confidence penalty for fuzzy matches
    if fuzzy_fraction > 0 and mean_fuzzy_confidence < 1.0:
        confidence_penalty = (1.0 - mean_fuzzy_confidence) * config.confidence_penalty_scale
        effective_threshold = effective_threshold - confidence_penalty

    deviations = []
    for i in range(num_layers):
        score = sigma(
            layer_idx=i,
            num_layers=num_layers,
            sigma_base=preset.sigma_base,
            sigma_e=preset.sigma_e,
            tau_e=preset.tau_e,
            sigma_l=preset.sigma_l,
            tau_l=preset.tau_l,
        )

        # Scale by mismatch fraction
        if mismatch_fraction > 0:
            score = score * (1.0 + mismatch_fraction)

        # Fuzzy deviation boost
        if fuzzy_fraction > 0:
            pf = position_factor(i, num_layers, model_name)
            fuzzy_boost = fuzzy_fraction * config.fuzzy_alpha * pf
            score = score * (1.0 + fuzzy_boost)

        score = min(score, 1.0)

        should_recompute = score >= effective_threshold

        # Apply explicit layer overrides
        if config.force_recompute_layers and i in config.force_recompute_layers:
            should_recompute = True
        if config.skip_recompute_layers and i in config.skip_recompute_layers:
            should_recompute = False

        deviations.append(
            LayerDeviation(
                layer_idx=i,
                deviation_score=score,
                should_recompute=should_recompute,
            )
        )

    # Force-verify edge layers for fuzzy matches with low confidence
    if fuzzy_fraction > 0 and config.force_verify_on_fuzzy and mean_fuzzy_confidence < 0.92:
        for i in range(num_layers):
            if i < preset.tau_e * 1.5 or i > num_layers - 1 - preset.tau_l * 1.5:
                if not deviations[i].should_recompute:
                    deviations[i] = LayerDeviation(
                        layer_idx=i,
                        deviation_score=deviations[i].deviation_score,
                        should_recompute=True,
                    )

    # Cap recomputation fraction
    recompute_count = sum(1 for d in deviations if d.should_recompute)
    max_recompute = int(num_layers * config.max_recompute_fraction)
    if recompute_count > max_recompute:
        # Keep highest-deviation layers, skip the rest
        scored = sorted(
            [(i, d) for i, d in enumerate(deviations) if d.should_recompute],
            key=lambda x: x[1].deviation_score,
            reverse=True,
        )
        keep_set = {idx for idx, _ in scored[:max_recompute]}
        for i, d in enumerate(deviations):
            if d.should_recompute and i not in keep_set:
                deviations[i] = LayerDeviation(
                    layer_idx=i,
                    deviation_score=d.deviation_score,
                    should_recompute=False,
                )

    return deviations


def get_preset(model_name: str) -> BathtubPreset:
    """Get bathtub curve preset for a model.

    Matches by substring (e.g., "Qwen/Qwen2.5-7B-Instruct-AWQ" matches "qwen2.5-7b").
    """
    name_lower = model_name.lower()
    for key, preset in PRESETS.items():
        if key != "default" and key in name_lower:
            return preset
    return PRESETS["default"]

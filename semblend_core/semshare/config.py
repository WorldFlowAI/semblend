"""SemShareKV configuration — all tunable hyperparameters."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SemShareConfig:
    """Immutable configuration for SemShareKV token-level matching.

    Default values follow the paper (arXiv:2509.24832) unless noted.
    """

    # LSH parameters
    lsh_num_tables: int = 8
    lsh_num_bits: int = 16
    lsh_w: float = 4.0  # hash bucket width
    lsh_seed: int = 42

    # High Deviation detection
    hd_fraction: float = 0.40  # top 40% by L2 deviation

    # Per-layer recomputation weights
    omega_h: float = 0.5  # HD token recompute weight
    omega_c: float = 0.1  # cold (non-HD) token recompute weight

    # Layer schedule
    alpha_base: float = 0.8  # base recompute fraction for layer 2
    alpha_decay: float = 0.85  # multiplicative decay per layer

    # Attention Recovery
    ar_threshold: float = 0.90  # 90% attention coverage

    # Model dimensions (overridden at runtime)
    embed_dim: int = 4096

    # Matching thresholds
    min_similarity: float = 0.3  # minimum per-token similarity to accept match
    min_match_ratio: float = 0.50  # minimum fraction of tokens that must match

    # Memory management
    max_donors: int = 1000  # cap for semshare mode (token data is large)
    ttl_seconds: float = 300.0  # evict donors older than this

    # LSH distance normalization (from paper)
    lsh_dist_min: float = 0.0
    lsh_dist_max: float = 30.0

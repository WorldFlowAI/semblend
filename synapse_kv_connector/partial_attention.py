"""Re-exports from semblend_core.partial_attention.

All partial attention logic lives in the shared semblend_core package.
This module re-exports everything for existing import paths.
"""
from semblend_core.partial_attention import (  # noqa: F401
    AttentionMode,
    LayerMask,
    PartialAttentionPlan,
    PositionMask,
    build_attention_plan,
    compute_attention_mask,
    compute_donor_kv_indices,
)

"""Re-exports from semblend_core.rope_correction.

All RoPE correction logic (delta + NoPE) lives in the shared
semblend_core package. This module re-exports for existing import paths.
"""
from semblend_core.rope_correction import (  # noqa: F401
    HAS_TRITON,
    apply_rope_delta_inplace,
    nope_permute_paged_kv,
    permute_paged_kv_with_rope,
    rope_correct_k,
    rope_correct_scatter_paged,
)

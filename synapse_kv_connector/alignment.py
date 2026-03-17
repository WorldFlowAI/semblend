"""Re-exports from semblend_core.alignment.

All alignment logic lives in the shared semblend_core package.
This module re-exports everything for existing import paths.
"""
from semblend_core.alignment import (  # noqa: F401
    DEFAULT_CHUNK_SIZE,
    AlignmentResult,
    SlotAction,
    SlotActionType,
    _chunk_hash,
    _fallback_prefix_alignment,
    _fuzzy_match_chunk,
    _levenshtein_alignment,
    _token_set_alignment,
    compute_alignment,
    compute_batch_alignment,
    compute_chunk_alignment,
    compute_fuzzy_chunk_alignment,
    estimate_reuse_ratio,
)

# Legacy alias for backward compatibility
LMCACHE_CHUNK_SIZE = DEFAULT_CHUNK_SIZE

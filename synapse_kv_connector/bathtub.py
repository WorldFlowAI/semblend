"""Re-exports from semblend_core.bathtub.

All bathtub curve logic lives in the shared semblend_core package.
This module re-exports everything for existing import paths.
"""
from semblend_core.bathtub import (  # noqa: F401
    PRESETS,
    BathtubPreset,
    LayerDeviation,
    adaptive_threshold,
    compute_layer_deviations,
    get_preset,
    sigma,
)

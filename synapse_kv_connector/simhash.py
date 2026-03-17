"""Re-exports from semblend_core.simhash.

All SimHash logic lives in the shared semblend_core package.
This module re-exports everything for existing import paths.
"""
from semblend_core.simhash import (  # noqa: F401
    bulk_hamming_distance,
    compute_simhash,
    hamming_distance,
    is_plausible_donor,
)

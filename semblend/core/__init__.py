"""SemBlend Core — backend-agnostic semantic KV donor discovery.

This module re-exports from semblend_core for the semblend.core namespace.
All actual implementation lives in semblend_core/ for backward compatibility.
"""
from semblend_core import *  # noqa: F401,F403
from semblend_core import (
    HAS_ROPE_CORRECTION as HAS_ROPE_CORRECTION,
)
from semblend_core import (
    HAS_TRITON_KERNELS as HAS_TRITON_KERNELS,
)

# Let semblend_core.__all__ flow through — don't restrict with a local __all__

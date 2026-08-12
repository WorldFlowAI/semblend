"""Legacy compatibility namespace for SemBlend vLLM integration.

New code should import SemBlend through ``semblend.integration.*``. This
namespace remains only for older vLLM connector module paths.
"""

from semblend_kv_connector.partial_attention import (
    AttentionMode,
    PartialAttentionPlan,
    build_attention_plan,
    compute_attention_mask,
    compute_donor_kv_indices,
)

# Triton kernels — optional, require torch + triton
try:
    from semblend_kv_connector.triton_kernels import (
        PartialPrefillResult,
        masked_qkv_projection,
        partial_prefill,
        partial_prefill_attention,
        scatter_donor_kv,
        scatter_donor_kv_paged,
    )

    HAS_TRITON_KERNELS = True
except ImportError:
    HAS_TRITON_KERNELS = False

# Model runner hook — optional, requires torch
try:
    from semblend_kv_connector.model_runner_hook import (
        PartialAttentionHook,
        patch_model_runner,
    )

    HAS_MODEL_HOOK = True
except ImportError:
    HAS_MODEL_HOOK = False

# SemBlend vLLM connector — optional, requires vLLM + LMCache
try:
    from semblend_kv_connector.semblend_connector import SemBlendConnectorV1

    HAS_SEMBLEND_CONNECTOR = True
except ImportError:
    HAS_SEMBLEND_CONNECTOR = False

# SemBlend pipeline — in-process semantic donor discovery
try:
    from semblend_kv_connector.pipeline import SemBlendPipeline

    HAS_SEMBLEND_PIPELINE = True
except ImportError:
    HAS_SEMBLEND_PIPELINE = False

__all__ = [
    "AttentionMode",
    "HAS_MODEL_HOOK",
    "HAS_SEMBLEND_CONNECTOR",
    "HAS_SEMBLEND_PIPELINE",
    "HAS_TRITON_KERNELS",
    "PartialAttentionHook",
    "PartialAttentionPlan",
    "PartialPrefillResult",
    "SemBlendConnectorV1",
    "SemBlendPipeline",
    "build_attention_plan",
    "compute_attention_mask",
    "compute_donor_kv_indices",
    "masked_qkv_projection",
    "partial_prefill",
    "partial_prefill_attention",
    "patch_model_runner",
    "scatter_donor_kv",
    "scatter_donor_kv_paged",
]

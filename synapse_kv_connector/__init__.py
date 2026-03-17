"""Synapse KV Cache Connector for vLLM — SemBlend semantic KV cache reuse.

Built on LMCache, provides semantic KV cache sharing across vLLM instances
through Synapse's REST API. Enables cross-request KV reuse for semantically
similar prompts, not just exact prefix matches.

Includes Triton CUDA kernels for GPU-accelerated PartialAttention:
    - scatter_donor_kv: Scatter donor K,V into target KV cache
    - masked_qkv_projection: Selective QKV GEMM for compute positions
    - partial_prefill_attention: Causal attention for compute positions only
    - partial_prefill: Full orchestrator chaining all three kernels
"""

from synapse_kv_connector.client import SynapseKVClient
from synapse_kv_connector.connector import SynapseKVConnector
from synapse_kv_connector.partial_attention import (
    AttentionMode,
    PartialAttentionPlan,
    build_attention_plan,
    compute_attention_mask,
    compute_donor_kv_indices,
)
from synapse_kv_connector.segment_client import (
    FindDonorResult,
    KvSlotAction,
    KvTransferPlan,
    SynapseSegmentClient,
)

# Triton kernels — optional, require torch + triton
try:
    from synapse_kv_connector.triton_kernels import (
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
    from synapse_kv_connector.model_runner_hook import (
        PartialAttentionHook,
        patch_model_runner,
    )

    HAS_MODEL_HOOK = True
except ImportError:
    HAS_MODEL_HOOK = False

# SemBlend vLLM connector — optional, requires vLLM + LMCache
try:
    from synapse_kv_connector.semblend_connector import SemBlendConnectorV1

    HAS_SEMBLEND_CONNECTOR = True
except ImportError:
    HAS_SEMBLEND_CONNECTOR = False

# SemBlend pipeline — in-process semantic donor discovery
try:
    from synapse_kv_connector.pipeline import SemBlendPipeline

    HAS_SEMBLEND_PIPELINE = True
except ImportError:
    HAS_SEMBLEND_PIPELINE = False

__all__ = [
    "AttentionMode",
    "FindDonorResult",
    "HAS_MODEL_HOOK",
    "HAS_TRITON_KERNELS",
    "KvSlotAction",
    "KvTransferPlan",
    "PartialAttentionPlan",
    "SynapseKVClient",
    "SynapseKVConnector",
    "SynapseSegmentClient",
    "build_attention_plan",
    "compute_attention_mask",
    "compute_donor_kv_indices",
]

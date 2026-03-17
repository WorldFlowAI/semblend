"""vLLM attention kernel patch for PartialAttention KV reuse.

Hooks into vLLM's model runner to consume PartialAttention plans
produced by the SynapseKVConnector. When a Tier 3 semantic match
returns a non-contiguous transfer plan, this module:

1. Scatters donor K,V into the target KV cache using a Triton GPU
   kernel (``scatter_donor_kv``), running in O(num_pairs) across
   all layers, heads, and positions simultaneously.

2. Computes Q,K,V projections only for placeholder positions via a
   masked GEMM (``masked_qkv_projection``), skipping reuse positions
   entirely and reducing prefill FLOPs proportionally.

3. Runs causal attention only for compute positions via
   ``partial_prefill_attention``, using online softmax over the full
   KV context (including donor values at reuse positions).

4. Reports the effective ``computation_ratio`` and per-kernel timing
   for paper benchmarks.

Integration points:
    - Called by vLLM's ``ModelRunner.execute_model()`` before each
      attention layer, if a PartialAttention plan is active.
    - The plan is attached to the request metadata by the connector's
      ``start_load_kv()`` method.

References:
    - KVShare (arXiv:2503.16525) §4.3 — PartialAttention mechanism
    - CacheBlend (EuroSys 2025) — layer-level selective recomputation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from synapse_kv_connector.partial_attention import (
    PartialAttentionPlan,
    compute_attention_mask,
    compute_donor_kv_indices,
)

logger = logging.getLogger(__name__)

# Conditional torch import for GPU-accelerated path
try:
    import torch

    from synapse_kv_connector.triton_kernels import (
        PartialPrefillResult,
        partial_prefill,
        partial_prefill_attention,
        scatter_donor_kv,
    )

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.info("torch not available — GPU-accelerated patch disabled")


@dataclass(frozen=True)
class PrefillPatchResult:
    """Result of applying a PartialAttention patch to a prefill step.

    Attributes:
        positions_computed: Number of positions that ran Q,K,V compute.
        positions_reused: Number of positions that reused donor K,V.
        computation_ratio: Fraction of full-prefill FLOPs actually used.
        layers_fully_recomputed: Layers that fell back to full compute.
        scatter_time_ms: Time for Triton KV scatter (GPU only).
        projection_time_ms: Time for masked QKV projection (GPU only).
        attention_time_ms: Time for partial attention (GPU only).
    """

    positions_computed: int
    positions_reused: int
    computation_ratio: float
    layers_fully_recomputed: int
    scatter_time_ms: float = 0.0
    projection_time_ms: float = 0.0
    attention_time_ms: float = 0.0


def apply_kv_patch(
    plan: PartialAttentionPlan,
    layer_idx: int,
    kv_cache: np.ndarray,
    donor_kv: np.ndarray,
) -> np.ndarray:
    """Apply donor KV values at reuse positions for a single layer.

    If torch + Triton are available and inputs are on GPU, uses the
    ``scatter_donor_kv`` Triton kernel. Otherwise falls back to
    element-wise numpy copy.

    Args:
        plan: The PartialAttention plan from the connector.
        layer_idx: Current transformer layer index (0-based).
        kv_cache: Target KV cache tensor for this layer.
            Shape: [2, num_heads, seq_len, head_dim] (K and V).
        donor_kv: Donor KV tensor for this layer.
            Shape: [2, num_heads, donor_len, head_dim].

    Returns:
        Updated kv_cache with donor values copied at reuse positions.
    """
    pairs = compute_donor_kv_indices(plan, layer_idx)

    if not pairs:
        return kv_cache

    # GPU-accelerated path via Triton
    if HAS_TORCH and isinstance(kv_cache, torch.Tensor):
        donor_pos = torch.tensor(
            [p[0] for p in pairs], dtype=torch.int32, device=kv_cache.device
        )
        target_pos = torch.tensor(
            [p[1] for p in pairs], dtype=torch.int32, device=kv_cache.device
        )
        # Reshape for scatter kernel: add layer dim
        target_5d = kv_cache.unsqueeze(0)  # [1, 2, num_heads, seq_len, head_dim]
        donor_5d = donor_kv.unsqueeze(0)  # [1, 2, num_heads, donor_len, head_dim]
        scatter_donor_kv(target_5d, donor_5d, donor_pos, target_pos)
        return kv_cache

    # Numpy fallback
    for donor_pos, target_pos in pairs:
        if donor_pos < donor_kv.shape[2] and target_pos < kv_cache.shape[2]:
            kv_cache[0, :, target_pos, :] = donor_kv[0, :, donor_pos, :]
            kv_cache[1, :, target_pos, :] = donor_kv[1, :, donor_pos, :]

    return kv_cache


def apply_kv_patch_gpu(
    plan: PartialAttentionPlan,
    target_kv: "torch.Tensor",
    donor_kv: "torch.Tensor",
) -> "torch.Tensor":
    """GPU-accelerated KV scatter across all layers simultaneously.

    Unlike ``apply_kv_patch`` which works layer-by-layer, this function
    scatters donor KV across ALL layers in a single Triton kernel launch.

    Args:
        plan: The PartialAttention plan.
        target_kv: Target KV cache [num_layers, 2, num_heads, seq_len, head_dim].
        donor_kv: Donor KV cache [num_layers, 2, num_heads, donor_len, head_dim].

    Returns:
        Updated target_kv (modified in-place, also returned).

    Raises:
        RuntimeError: If torch/Triton not available.
    """
    if not HAS_TORCH:
        raise RuntimeError("torch required for GPU KV scatter")

    # Collect all (donor_pos, target_pos) pairs from layer 0
    # (same positions apply to all layers)
    pairs = compute_donor_kv_indices(plan, 0)
    if not pairs:
        return target_kv

    donor_pos = torch.tensor(
        [p[0] for p in pairs], dtype=torch.int32, device=target_kv.device
    )
    target_pos = torch.tensor(
        [p[1] for p in pairs], dtype=torch.int32, device=target_kv.device
    )

    scatter_donor_kv(target_kv, donor_kv, donor_pos, target_pos)
    return target_kv


def get_compute_mask(
    plan: PartialAttentionPlan,
    layer_idx: int,
) -> np.ndarray:
    """Get the boolean compute mask for a layer.

    Returns a 1D boolean array where True means "compute Q,K,V for
    this position" and False means "reuse donor K,V".

    Args:
        plan: The PartialAttention plan.
        layer_idx: Current transformer layer index.

    Returns:
        Boolean numpy array of shape [target_len].
    """
    return compute_attention_mask(plan, layer_idx)


def get_compute_mask_gpu(
    plan: PartialAttentionPlan,
    layer_idx: int,
    device: str = "cuda",
) -> "torch.Tensor":
    """Get the compute mask as a GPU tensor.

    Args:
        plan: The PartialAttention plan.
        layer_idx: Transformer layer index.
        device: Target device (default: "cuda").

    Returns:
        Boolean torch tensor on the specified device.
    """
    if not HAS_TORCH:
        raise RuntimeError("torch required for GPU compute mask")

    np_mask = compute_attention_mask(plan, layer_idx)
    return torch.from_numpy(np_mask).to(device)


def execute_partial_prefill(
    plan: PartialAttentionPlan,
    hidden_states: "torch.Tensor",
    donor_kv: "torch.Tensor",
    qkv_weight: "torch.Tensor",
    output_projection: "torch.Tensor",
    num_heads: int,
    head_dim: int,
    qkv_bias: "torch.Tensor | None" = None,
    output_bias: "torch.Tensor | None" = None,
) -> PartialPrefillResult:
    """Execute a complete partial prefill for a single layer.

    This is the main entry point for vLLM integration. Orchestrates:
    1. Scatter donor KV into target cache (Triton kernel)
    2. Masked QKV projection for compute positions only (Triton kernel)
    3. Causal attention for compute positions (Triton kernel)

    Args:
        plan: PartialAttention plan from the connector.
        hidden_states: Transformer input [seq_len, hidden_dim].
        donor_kv: Donor KV [num_layers, 2, num_heads, donor_len, head_dim].
        qkv_weight: Combined QKV weight [3*num_heads*head_dim, hidden_dim].
        output_projection: Output weight [hidden_dim, num_heads*head_dim].
        num_heads: Number of attention heads.
        head_dim: Dimension per head.
        qkv_bias: Optional QKV bias.
        output_bias: Optional output projection bias.

    Returns:
        PartialPrefillResult with output, KV cache, and timing info.
    """
    if not HAS_TORCH:
        raise RuntimeError("torch required for partial prefill")

    # Build position tensors from plan
    pairs = compute_donor_kv_indices(plan, 0)
    device = hidden_states.device

    donor_positions = torch.tensor(
        [p[0] for p in pairs], dtype=torch.int32, device=device
    )
    target_positions = torch.tensor(
        [p[1] for p in pairs], dtype=torch.int32, device=device
    )
    compute_mask = get_compute_mask_gpu(plan, 0, device=str(device))

    return partial_prefill(
        hidden_states=hidden_states,
        donor_kv=donor_kv,
        qkv_weight=qkv_weight,
        output_projection=output_projection,
        donor_positions=donor_positions,
        target_positions=target_positions,
        compute_mask=compute_mask,
        num_heads=num_heads,
        head_dim=head_dim,
        qkv_bias=qkv_bias,
        output_bias=output_bias,
    )


def summarize_patch(plan: PartialAttentionPlan) -> PrefillPatchResult:
    """Summarize the PartialAttention patch for logging/benchmarking.

    Args:
        plan: The completed PartialAttention plan.

    Returns:
        PrefillPatchResult with aggregate statistics.
    """
    return PrefillPatchResult(
        positions_computed=plan.num_partial_positions,
        positions_reused=plan.num_reuse_positions,
        computation_ratio=plan.computation_ratio,
        layers_fully_recomputed=plan.num_full_layers,
    )

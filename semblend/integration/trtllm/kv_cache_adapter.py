"""TRT-LLM KV cache layout adapter -- stride computation for zero-copy reuse.

TRT-LLM's PyTorch backend uses a different KV cache layout than vLLM:

    TRT-LLM:  [num_blocks, 2, tokens_per_block, num_kv_heads, head_dim]
    vLLM:     [num_blocks, 2, num_heads,         block_size,   head_dim]

The dimensions are: tokens_per_block ↔ num_kv_heads are swapped.

Since SemBlend's Triton kernels (rope_correct_scatter_paged_kernel,
scatter_donor_kv_paged_kernel) accept explicit stride parameters, we
handle the layout difference by computing the correct strides -- NO
data transposition needed.

This is the same approach used for vLLM's flash_attn layout detection
in apply_rope_delta_inplace (rope_correction.py:749).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class KVCacheStrides:
    """Stride parameters for a paged KV cache tensor.

    These strides are passed directly to Triton kernels, allowing
    them to work with any KV cache memory layout without transposition.
    """
    kv_stride_block: int
    kv_stride_kv: int
    kv_stride_head: int
    kv_stride_pos: int
    kv_stride_dim: int


@dataclass(frozen=True)
class KVCacheLayout:
    """Describes the KV cache layout of a specific engine backend.

    Attributes:
        num_blocks: Total number of physical KV blocks.
        num_kv_heads: Number of key-value attention heads.
        tokens_per_block: Tokens stored per block (e.g. 128 for TRT-LLM).
        head_dim: Dimension per attention head.
        layout_name: Human-readable layout identifier.
    """
    num_blocks: int
    num_kv_heads: int
    tokens_per_block: int
    head_dim: int
    layout_name: str


def detect_trtllm_layout(kv_tensor: Any) -> KVCacheLayout:
    """Detect TRT-LLM PyTorch backend KV cache layout from tensor shape.

    TRT-LLM PyTorch backend layout:
        [num_blocks, 2, tokens_per_block, num_kv_heads, head_dim]

    The key distinction from vLLM: tokens_per_block comes BEFORE num_kv_heads.
    TRT-LLM typically uses tokens_per_block=128 (configurable via KvCacheConfig).

    Args:
        kv_tensor: KV cache tensor from KVCacheManager.get_buffers(layer_idx).

    Returns:
        KVCacheLayout describing the tensor's dimensions.

    Raises:
        ValueError: If tensor shape doesn't match expected TRT-LLM layout.
    """
    if kv_tensor.ndim != 5:
        raise ValueError(
            f"Expected 5D KV cache tensor, got {kv_tensor.ndim}D: "
            f"shape={tuple(kv_tensor.shape)}"
        )

    shape = tuple(kv_tensor.shape)

    if shape[1] != 2:
        raise ValueError(
            f"Expected dim[1]=2 (K,V), got {shape[1]}: shape={shape}"
        )

    # TRT-LLM: [num_blocks, 2, tokens_per_block, num_kv_heads, head_dim]
    num_blocks = shape[0]
    tokens_per_block = shape[2]
    num_kv_heads = shape[3]
    head_dim = shape[4]

    return KVCacheLayout(
        num_blocks=num_blocks,
        num_kv_heads=num_kv_heads,
        tokens_per_block=tokens_per_block,
        head_dim=head_dim,
        layout_name="trtllm_pytorch",
    )


def trtllm_kv_strides(kv_tensor: Any) -> KVCacheStrides:
    """Compute strides for TRT-LLM paged KV cache layout.

    TRT-LLM layout: [num_blocks, 2, tokens_per_block, num_kv_heads, head_dim]

    The stride names map to SemBlend's Triton kernel parameters:
        kv_stride_block -> stride along num_blocks dimension
        kv_stride_kv    -> stride along K/V dimension (always dim 1)
        kv_stride_pos   -> stride along token position dimension
        kv_stride_head  -> stride along attention head dimension
        kv_stride_dim   -> stride along head_dim dimension

    Note: For TRT-LLM, pos (dim 2) comes before head (dim 3), which is
    the reverse of vLLM's standard layout. The stride computation handles
    this automatically.

    Args:
        kv_tensor: KV cache tensor from KVCacheManager.get_buffers(layer_idx).

    Returns:
        KVCacheStrides with correct strides for Triton kernel dispatch.
    """
    return KVCacheStrides(
        kv_stride_block=kv_tensor.stride(0),
        kv_stride_kv=kv_tensor.stride(1),
        kv_stride_pos=kv_tensor.stride(2),   # tokens_per_block dim
        kv_stride_head=kv_tensor.stride(3),   # num_kv_heads dim
        kv_stride_dim=kv_tensor.stride(4),
    )


def vllm_kv_strides(kv_tensor: Any) -> KVCacheStrides:
    """Compute strides for vLLM standard paged KV cache layout.

    vLLM layout: [num_blocks, 2, num_heads, block_size, head_dim]

    Provided for reference and cross-engine testing.

    Args:
        kv_tensor: KV cache tensor in vLLM layout.

    Returns:
        KVCacheStrides with correct strides for Triton kernel dispatch.
    """
    return KVCacheStrides(
        kv_stride_block=kv_tensor.stride(0),
        kv_stride_kv=kv_tensor.stride(1),
        kv_stride_head=kv_tensor.stride(2),   # num_heads dim
        kv_stride_pos=kv_tensor.stride(3),    # block_size dim
        kv_stride_dim=kv_tensor.stride(4),
    )


def is_trtllm_layout(kv_tensor: Any) -> bool:
    """Heuristic detection: is this a TRT-LLM KV cache layout?

    TRT-LLM uses tokens_per_block (typically 64 or 128) at dim[2],
    while vLLM places num_heads (typically 8-128) at dim[2].

    Heuristic: if dim[2] is a power of 2 and dim[2] >= 32 and
    dim[2] > dim[3], it's likely TRT-LLM (tokens_per_block > num_kv_heads
    for models with GQA/MQA like Qwen2.5-7B where num_kv_heads=4).

    For unambiguous detection, use detect_trtllm_layout() with explicit
    engine context.

    Args:
        kv_tensor: 5D KV cache tensor.

    Returns:
        True if tensor appears to be TRT-LLM layout.
    """
    if kv_tensor.ndim != 5 or kv_tensor.shape[1] != 2:
        return False

    dim2 = kv_tensor.shape[2]  # tokens_per_block or num_heads
    dim3 = kv_tensor.shape[3]  # num_kv_heads or block_size

    # TRT-LLM: dim2=tokens_per_block (64/128), dim3=num_kv_heads (4-32)
    # vLLM: dim2=num_heads (4-128), dim3=block_size (16-32)
    #
    # Most reliable signal: tokens_per_block is typically 64 or 128
    # and is always a power of 2
    is_pow2 = dim2 > 0 and (dim2 & (dim2 - 1)) == 0
    return is_pow2 and dim2 >= 64 and dim2 > dim3

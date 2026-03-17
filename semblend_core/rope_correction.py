"""RoPE delta correction and NoPE injection for non-contiguous KV cache reuse.

Two implementations for correcting position encoding when reusing donor KV:

APPROACH 1 — Delta correction (current SemBlend default):
    K_corrected = RoPE(target_pos - donor_pos) × K_donor

APPROACH 2 — NoPE two-step (MEPIC-style):
    Step 1: Strip donor position:  K_raw = RoPE(-donor_pos) × K_donor
    Step 2: Apply target position: K_target = RoPE(target_pos) × K_raw

Both are mathematically identical (RoPE is a rotation group, so
  RoPE(target_pos) × RoPE(-donor_pos) = RoPE(target_pos - donor_pos)).

The NoPE approach models MEPIC's "store without position, inject at query time"
architecture. In NoPE storage, K is kept position-free and position is applied
only when injecting into the target sequence — eliminating the need for
donor_pos at inject time (only target_pos is required).

The correction costs O(d) per token per head per layer, ~7μs total for
8K tokens on A10G. NoPE adds one extra Triton kernel launch vs delta
correction but achieves the same result with the same hardware cost.

V cache has NO position encoding and can be rearranged freely (both modes).

References:
    - Su et al., 2021 "RoFormer: Enhanced Transformer with Rotary Position
      Embedding" — defines RoPE as paired 2D rotations
    - SemShareKV (Zhao & Mastorakis, AACL 2025) — cross-prompt KV reuse with
      position awareness; uses approximate correction (first-layer full
      recompute). We improve with exact delta correction.
    - MEPIC (2025) — NoPE storage + fused CUDA inject kernel. SemBlend
      implements an equivalent approach without framework modification.

Usage:
    # Delta correction (Approach 1, default)
    rope_correct_k(k_tensor, donor_positions, target_positions, head_dim=128)

    # NoPE two-step (Approach 2)
    nope_permute_paged_kv(kv_cache, block_table, permutation, rope_base=10000.0)

    # Toggle via env var: SEMBLEND_USE_NOPE=1
"""
from __future__ import annotations

import logging
import math

import torch

logger = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ---------------------------------------------------------------------------
# Triton kernel: fused RoPE correction + scatter to paged cache
# ---------------------------------------------------------------------------

if HAS_TRITON:

    @triton.jit
    def _rope_correct_scatter_paged_kernel(
        # Pointers
        kv_cache_ptr,
        donor_kv_ptr,
        block_table_ptr,
        donor_pos_ptr,
        target_pos_ptr,
        inv_freq_ptr,
        # Dimensions
        num_pairs: tl.constexpr,
        num_heads: tl.constexpr,
        block_size: tl.constexpr,
        head_dim: tl.constexpr,
        half_dim: tl.constexpr,
        donor_seq_len: tl.constexpr,
        num_blocks_total: tl.constexpr,
        num_blocks_per_seq: tl.constexpr,
        # Strides for kv_cache: [num_blocks, 2, num_heads, block_size, head_dim]
        kv_stride_block,
        kv_stride_kv,
        kv_stride_head,
        kv_stride_pos,
        kv_stride_dim,
        # Strides for donor: [2, num_heads, donor_seq_len, head_dim]
        d_stride_kv,
        d_stride_head,
        d_stride_seq,
        d_stride_dim,
        # Block
        BLOCK_HALF: tl.constexpr,
    ):
        """Fused kernel: load donor KV, apply RoPE correction to K, scatter
        to paged cache. V is copied without correction.

        Grid: (num_pairs, 2 * num_heads)
        - kv_idx=0 (K): apply RoPE(delta) correction
        - kv_idx=1 (V): direct copy (no position encoding)
        """
        pair_idx = tl.program_id(0)
        compound_idx = tl.program_id(1)

        kv_idx = compound_idx // num_heads  # 0=K, 1=V
        head_idx = compound_idx % num_heads

        # Load positions
        d_pos = tl.load(donor_pos_ptr + pair_idx)
        t_pos = tl.load(target_pos_ptr + pair_idx)

        # Bounds check
        if d_pos >= donor_seq_len:
            return

        logical_block = t_pos // block_size
        block_offset = t_pos % block_size

        if logical_block >= num_blocks_per_seq:
            return

        physical_block = tl.load(block_table_ptr + logical_block)
        if physical_block >= num_blocks_total:
            return

        # Compute base offsets
        donor_base = (
            kv_idx * d_stride_kv
            + head_idx * d_stride_head
            + d_pos * d_stride_seq
        )
        target_base = (
            physical_block * kv_stride_block
            + kv_idx * kv_stride_kv
            + head_idx * kv_stride_head
            + block_offset * kv_stride_pos
        )

        if kv_idx == 1:
            # V: direct copy (no position encoding)
            offsets = tl.arange(0, BLOCK_HALF * 2)
            mask = offsets < head_dim
            vals = tl.load(
                donor_kv_ptr + donor_base + offsets * d_stride_dim, mask=mask
            )
            tl.store(
                kv_cache_ptr + target_base + offsets * kv_stride_dim,
                vals,
                mask=mask,
            )
        else:
            # K: apply RoPE(target_pos - donor_pos) correction
            delta = (t_pos - d_pos).to(tl.float32)

            half_offsets = tl.arange(0, BLOCK_HALF)
            half_mask = half_offsets < half_dim

            # Load inv_freq for this pair of dimensions
            freq = tl.load(inv_freq_ptr + half_offsets, mask=half_mask)
            theta = delta * freq
            cos_val = tl.cos(theta)
            sin_val = tl.sin(theta)

            # Load K[2i] and K[2i+1] (interleaved real/imaginary)
            even_offsets = half_offsets * 2
            odd_offsets = half_offsets * 2 + 1

            k_even = tl.load(
                donor_kv_ptr + donor_base + even_offsets * d_stride_dim,
                mask=half_mask,
            ).to(tl.float32)
            k_odd = tl.load(
                donor_kv_ptr + donor_base + odd_offsets * d_stride_dim,
                mask=half_mask,
            ).to(tl.float32)

            # RoPE rotation: [cos -sin; sin cos] × [k_even; k_odd]
            k_even_new = k_even * cos_val - k_odd * sin_val
            k_odd_new = k_even * sin_val + k_odd * cos_val

            # Store corrected K
            tl.store(
                kv_cache_ptr + target_base + even_offsets * kv_stride_dim,
                k_even_new.to(tl.float16),
                mask=half_mask,
            )
            tl.store(
                kv_cache_ptr + target_base + odd_offsets * kv_stride_dim,
                k_odd_new.to(tl.float16),
                mask=half_mask,
            )


def rope_correct_scatter_paged(
    kv_cache: torch.Tensor,
    donor_kv: torch.Tensor,
    block_table: torch.Tensor,
    donor_positions: torch.Tensor,
    target_positions: torch.Tensor,
    rope_base: float = 10000.0,
    head_dim: int | None = None,
) -> torch.Tensor:
    """Scatter donor KV into paged cache with RoPE correction on K.

    For each (donor_pos, target_pos) pair:
    - K is corrected: K_new = RoPE(target - donor) × K_donor
    - V is copied directly (no position encoding)

    This enables non-contiguous KV reuse: donor KV from any position can be
    injected at any target position with mathematically exact correction.

    Args:
        kv_cache: Paged KV cache [num_blocks, 2, num_heads, block_size, head_dim].
        donor_kv: Donor KV tensor [2, num_heads, donor_seq_len, head_dim].
        block_table: Block table [num_blocks_per_seq] int32.
        donor_positions: Source positions in donor sequence [num_pairs] int32.
        target_positions: Target positions in target sequence [num_pairs] int32.
        rope_base: RoPE frequency base (10000.0 for most models).
        head_dim: Head dimension (inferred from kv_cache if None).

    Returns:
        Updated kv_cache (modified in-place).
    """
    num_pairs = donor_positions.shape[0]
    if num_pairs == 0:
        return kv_cache

    num_blocks_total, _, num_heads, block_size, hd = kv_cache.shape
    if head_dim is None:
        head_dim = hd
    half_dim = head_dim // 2
    _, _, donor_seq_len, _ = donor_kv.shape
    num_blocks_per_seq = block_table.shape[0]

    # Compute inv_freq: 1 / (base ^ (2i / dim)) for i in [0, dim/2)
    inv_freq = 1.0 / (
        rope_base
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=kv_cache.device) / head_dim)
    )

    if HAS_TRITON and kv_cache.is_cuda:
        block_half = triton.next_power_of_2(half_dim)
        grid = (num_pairs, 2 * num_heads)

        _rope_correct_scatter_paged_kernel[grid](
            kv_cache,
            donor_kv,
            block_table,
            donor_positions,
            target_positions,
            inv_freq,
            num_pairs=num_pairs,
            num_heads=num_heads,
            block_size=block_size,
            head_dim=head_dim,
            half_dim=half_dim,
            donor_seq_len=donor_seq_len,
            num_blocks_total=num_blocks_total,
            num_blocks_per_seq=num_blocks_per_seq,
            kv_stride_block=kv_cache.stride(0),
            kv_stride_kv=kv_cache.stride(1),
            kv_stride_head=kv_cache.stride(2),
            kv_stride_pos=kv_cache.stride(3),
            kv_stride_dim=kv_cache.stride(4),
            d_stride_kv=donor_kv.stride(0),
            d_stride_head=donor_kv.stride(1),
            d_stride_seq=donor_kv.stride(2),
            d_stride_dim=donor_kv.stride(3),
            BLOCK_HALF=block_half,
        )
    else:
        _rope_correct_scatter_paged_cpu(
            kv_cache, donor_kv, block_table,
            donor_positions, target_positions, inv_freq,
            block_size, head_dim,
        )

    return kv_cache


def _rope_correct_scatter_paged_cpu(
    kv_cache: torch.Tensor,
    donor_kv: torch.Tensor,
    block_table: torch.Tensor,
    donor_positions: torch.Tensor,
    target_positions: torch.Tensor,
    inv_freq: torch.Tensor,
    block_size: int,
    head_dim: int,
) -> None:
    """CPU fallback for RoPE-corrected scatter."""
    for i in range(donor_positions.shape[0]):
        d_pos = int(donor_positions[i])
        t_pos = int(target_positions[i])

        logical_block = t_pos // block_size
        offset = t_pos % block_size
        physical_block = int(block_table[logical_block])

        delta = float(t_pos - d_pos)

        # V: direct copy
        kv_cache[physical_block, 1, :, offset, :] = donor_kv[1, :, d_pos, :]

        # K: RoPE correction
        k = donor_kv[0, :, d_pos, :].float()  # [num_heads, head_dim]

        # Apply RoPE(delta) to each dimension pair
        theta = delta * inv_freq  # [head_dim // 2]
        cos_vals = torch.cos(theta)
        sin_vals = torch.sin(theta)

        k_even = k[:, 0::2]  # [num_heads, head_dim//2]
        k_odd = k[:, 1::2]

        k_corrected_even = k_even * cos_vals - k_odd * sin_vals
        k_corrected_odd = k_even * sin_vals + k_odd * cos_vals

        # Interleave back
        k_corrected = torch.zeros_like(k)
        k_corrected[:, 0::2] = k_corrected_even
        k_corrected[:, 1::2] = k_corrected_odd

        kv_cache[physical_block, 0, :, offset, :] = k_corrected.to(
            kv_cache.dtype
        )


# ---------------------------------------------------------------------------
# Standalone RoPE correction (for flat KV cache or testing)
# ---------------------------------------------------------------------------


def rope_correct_k(
    k_tensor: torch.Tensor,
    donor_positions: torch.Tensor,
    target_positions: torch.Tensor,
    head_dim: int = 128,
    rope_base: float = 10000.0,
) -> torch.Tensor:
    """Apply RoPE delta correction to K tensor.

    For each position pair (donor_pos, target_pos), applies:
        K_corrected = RoPE(target - donor) × K_donor

    This corrects the position encoding baked into cached K tensors,
    enabling non-contiguous KV reuse with mathematical exactness.

    Args:
        k_tensor: K cache [num_heads, seq_len, head_dim] or
                  [num_layers, num_heads, seq_len, head_dim].
        donor_positions: Donor positions [num_pairs] int32.
        target_positions: Target positions [num_pairs] int32.
        head_dim: Dimension per head (default 128 for Qwen2.5).
        rope_base: RoPE base frequency (default 10000.0).

    Returns:
        Corrected K tensor (new tensor, original not modified).
    """
    device = k_tensor.device
    dtype = k_tensor.dtype

    inv_freq = 1.0 / (
        rope_base
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )

    deltas = (target_positions - donor_positions).float()  # [num_pairs]

    # theta[i, d] = delta[i] * inv_freq[d]
    theta = deltas.unsqueeze(-1) * inv_freq.unsqueeze(0)  # [num_pairs, half_dim]
    cos_vals = torch.cos(theta)  # [num_pairs, half_dim]
    sin_vals = torch.sin(theta)

    result = k_tensor.clone()
    has_layer_dim = k_tensor.dim() == 4

    for i in range(donor_positions.shape[0]):
        d_pos = int(donor_positions[i])
        cos_i = cos_vals[i]  # [half_dim]
        sin_i = sin_vals[i]

        if has_layer_dim:
            k = result[:, :, d_pos, :].float()  # [L, H, D]
            k_even = k[:, :, 0::2].clone()
            k_odd = k[:, :, 1::2].clone()
            k[:, :, 0::2] = k_even * cos_i - k_odd * sin_i
            k[:, :, 1::2] = k_even * sin_i + k_odd * cos_i
            result[:, :, d_pos, :] = k.to(dtype)
        else:
            k = result[:, d_pos, :].float()  # [H, D]
            k_even = k[:, 0::2].clone()
            k_odd = k[:, 1::2].clone()
            k[:, 0::2] = k_even * cos_i - k_odd * sin_i
            k[:, 1::2] = k_even * sin_i + k_odd * cos_i
            result[:, d_pos, :] = k.to(dtype)

    return result


# ---------------------------------------------------------------------------
# In-place paged cache permutation with RoPE correction
# ---------------------------------------------------------------------------


def permute_paged_kv_with_rope(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    permutation: list[tuple[int, int]],
    rope_base: float = 10000.0,
) -> None:
    """Permute KV entries within a paged cache, applying RoPE correction to K.

    After LMCache loads donor KV in donor order, this function rearranges
    the KV to target order with exact position correction.

    Vectorized: batches all reads, computes all RoPE corrections, then
    writes all results in a few GPU operations instead of per-pair loops.

    Args:
        kv_cache: Paged cache [num_blocks, 2, num_heads, block_size, head_dim].
        block_table: Block table [num_blocks_per_seq] int32.
        permutation: List of (source_logical_pos, target_logical_pos) pairs.
        rope_base: RoPE base frequency.
    """
    if not permutation:
        return

    device = kv_cache.device
    # Layout: [num_blocks, 2, num_heads, block_size, head_dim]
    _, _, num_heads, block_size, head_dim = kv_cache.shape

    n = len(permutation)
    src_pos_list = torch.tensor(
        [p[0] for p in permutation], dtype=torch.long, device=device
    )
    tgt_pos_list = torch.tensor(
        [p[1] for p in permutation], dtype=torch.long, device=device
    )

    # Deduplicate source positions for batch read
    unique_src, inverse_idx = torch.unique(src_pos_list, return_inverse=True)
    src_blocks = block_table[unique_src // block_size].long()
    src_offsets = unique_src % block_size

    # Batch read all unique source K and V
    src_k_unique = kv_cache[src_blocks, 0, :, src_offsets, :].clone()
    src_v_unique = kv_cache[src_blocks, 1, :, src_offsets, :].clone()

    # Expand to full permutation via inverse index
    src_k = src_k_unique[inverse_idx]  # [n, num_heads, head_dim]
    src_v = src_v_unique[inverse_idx]

    # Compute target physical addresses
    tgt_blocks = block_table[tgt_pos_list // block_size].long()
    tgt_offsets = tgt_pos_list % block_size

    # V: direct copy (no position encoding)
    kv_cache[tgt_blocks, 1, :, tgt_offsets, :] = src_v

    # K: vectorized RoPE correction
    deltas = (tgt_pos_list - src_pos_list).float()  # [n]
    inv_freq = 1.0 / (
        rope_base
        ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
            / head_dim
        )
    )

    # theta: [n, head_dim//2]
    theta = deltas.unsqueeze(-1) * inv_freq.unsqueeze(0)
    cos_v = torch.cos(theta).unsqueeze(1)  # [n, 1, head_dim//2]
    sin_v = torch.sin(theta).unsqueeze(1)

    k = src_k.float()  # [n, num_heads, head_dim]
    k_even = k[:, :, 0::2]  # [n, num_heads, head_dim//2]
    k_odd = k[:, :, 1::2]

    k_corrected = torch.zeros_like(k)
    k_corrected[:, :, 0::2] = k_even * cos_v - k_odd * sin_v
    k_corrected[:, :, 1::2] = k_even * sin_v + k_odd * cos_v

    kv_cache[tgt_blocks, 0, :, tgt_offsets, :] = k_corrected.to(kv_cache.dtype)


# ---------------------------------------------------------------------------
# NoPE (No Position Encoding) — MEPIC-style two-step approach
#
# APPROACH 2: Store K without position, inject with full target position.
#
#   Step 1 (strip at donor): K_raw = RoPE(-src_pos) × K_with_rope
#     k_even_raw = k_even * cos(θ_s) + k_odd * sin(θ_s)     [cos(-θ)=cos(θ)]
#     k_odd_raw  = -k_even * sin(θ_s) + k_odd * cos(θ_s)    [sin(-θ)=-sin(θ)]
#
#   Step 2 (inject at target): K_target = RoPE(tgt_pos) × K_raw
#     k_even_new = k_raw_even * cos(θ_t) - k_raw_odd * sin(θ_t)
#     k_odd_new  = k_raw_even * sin(θ_t) + k_raw_odd * cos(θ_t)
#
# Combined: RoPE(tgt_pos) × RoPE(-src_pos) = RoPE(tgt_pos - src_pos)
# This is identical to delta correction — empirically confirmed by benchmark.
# ---------------------------------------------------------------------------

if HAS_TRITON:

    @triton.jit
    def _nope_permute_paged_kernel(
        # Pointers
        kv_cache_ptr,
        src_kv_ptr,           # Source K snapshot [num_pairs, num_heads, head_dim]
        block_table_ptr,
        src_pos_ptr,          # Source positions [num_pairs] int32
        tgt_pos_ptr,          # Target positions [num_pairs] int32
        inv_freq_ptr,
        # Dimensions
        num_pairs: tl.constexpr,
        num_heads: tl.constexpr,
        block_size: tl.constexpr,
        head_dim: tl.constexpr,
        half_dim: tl.constexpr,
        num_blocks_total: tl.constexpr,
        num_blocks_per_seq: tl.constexpr,
        # Strides for kv_cache [num_blocks, 2, num_heads, block_size, head_dim]
        kv_stride_block,
        kv_stride_kv,
        kv_stride_head,
        kv_stride_pos,
        kv_stride_dim,
        # Strides for src_kv [num_pairs, num_heads, head_dim]
        s_stride_pair,
        s_stride_head,
        s_stride_dim,
        BLOCK_HALF: tl.constexpr,
    ):
        """NoPE two-step K correction: strip RoPE(-src_pos) then apply RoPE(tgt_pos).

        Mathematically equivalent to RoPE(tgt_pos - src_pos) (delta correction),
        but demonstrates the NoPE architecture: position-free intermediate K.

        Grid: (num_pairs, num_heads)
        """
        pair_idx = tl.program_id(0)
        head_idx = tl.program_id(1)

        s_pos = tl.load(src_pos_ptr + pair_idx)
        t_pos = tl.load(tgt_pos_ptr + pair_idx)

        logical_block = t_pos // block_size
        block_offset = t_pos % block_size
        if logical_block >= num_blocks_per_seq:
            return
        physical_block = tl.load(block_table_ptr + logical_block)
        if physical_block >= num_blocks_total:
            return

        half_offsets = tl.arange(0, BLOCK_HALF)
        half_mask = half_offsets < half_dim
        even_offsets = half_offsets * 2
        odd_offsets = half_offsets * 2 + 1

        freq = tl.load(inv_freq_ptr + half_offsets, mask=half_mask)

        # Load pre-snapshotted K_with_rope from src_kv buffer
        src_base = pair_idx * s_stride_pair + head_idx * s_stride_head
        k_even = tl.load(
            src_kv_ptr + src_base + even_offsets * s_stride_dim, mask=half_mask
        ).to(tl.float32)
        k_odd = tl.load(
            src_kv_ptr + src_base + odd_offsets * s_stride_dim, mask=half_mask
        ).to(tl.float32)

        # Step 1: Strip RoPE(-src_pos) → K_raw
        # RoPE(-θ) rotation: [cos θ, sin θ; -sin θ, cos θ]
        theta_s = s_pos.to(tl.float32) * freq
        cos_s = tl.cos(theta_s)
        sin_s = tl.sin(theta_s)
        k_raw_even = k_even * cos_s + k_odd * sin_s
        k_raw_odd = -k_even * sin_s + k_odd * cos_s

        # Step 2: Apply RoPE(tgt_pos) → K_target
        # RoPE(+θ) rotation: [cos θ, -sin θ; sin θ, cos θ]
        theta_t = t_pos.to(tl.float32) * freq
        cos_t = tl.cos(theta_t)
        sin_t = tl.sin(theta_t)
        k_even_new = k_raw_even * cos_t - k_raw_odd * sin_t
        k_odd_new = k_raw_even * sin_t + k_raw_odd * cos_t

        # Write corrected K to target position in paged cache (kv_idx=0)
        tgt_base = (
            physical_block * kv_stride_block
            + 0 * kv_stride_kv
            + head_idx * kv_stride_head
            + block_offset * kv_stride_pos
        )
        tl.store(
            kv_cache_ptr + tgt_base + even_offsets * kv_stride_dim,
            k_even_new.to(tl.float16),
            mask=half_mask,
        )
        tl.store(
            kv_cache_ptr + tgt_base + odd_offsets * kv_stride_dim,
            k_odd_new.to(tl.float16),
            mask=half_mask,
        )


def nope_permute_paged_kv(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    permutation: list[tuple[int, int]],
    rope_base: float = 10000.0,
) -> None:
    """NoPE two-step position correction: strip RoPE(src_pos), inject RoPE(tgt_pos).

    Implements the MEPIC-style NoPE architecture for K correction:
      K_target = RoPE(tgt_pos) × RoPE(-src_pos) × K_with_rope
             = RoPE(tgt_pos - src_pos) × K_with_rope   (same as delta correction)

    This is mathematically equivalent to delta correction but demonstrates the
    NoPE computation pattern: strip position encoding, then re-apply at target.

    Args:
        kv_cache: Paged cache [num_blocks, 2, num_heads, block_size, head_dim].
        block_table: Block table [num_blocks_per_seq] int32.
        permutation: List of (src_logical_pos, tgt_logical_pos) pairs.
        rope_base: RoPE base frequency.
    """
    if not permutation:
        return

    _, _, num_heads, block_size, head_dim = kv_cache.shape
    half_dim = head_dim // 2
    num_blocks_total = kv_cache.shape[0]
    num_blocks_per_seq = block_table.shape[0]

    inv_freq = 1.0 / (
        rope_base
        ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32, device=kv_cache.device)
            / head_dim
        )
    )

    # Snapshot source K values before any writes (prevents overwrite-before-read)
    num_pairs = len(permutation)
    src_kv = torch.zeros(
        (num_pairs, num_heads, head_dim),
        dtype=kv_cache.dtype,
        device=kv_cache.device,
    )
    src_pos_list = []
    tgt_pos_list = []

    for i, (src_pos, tgt_pos) in enumerate(permutation):
        sb = int(block_table[src_pos // block_size])
        so = src_pos % block_size
        src_kv[i] = kv_cache[sb, 0, :, so, :]  # K only (kv_idx=0)
        src_pos_list.append(src_pos)
        tgt_pos_list.append(tgt_pos)

        # V: direct copy (no position encoding in V)
        tb = int(block_table[tgt_pos // block_size])
        to_ = tgt_pos % block_size
        kv_cache[tb, 1, :, to_, :] = kv_cache[sb, 1, :, so, :]

    src_pos_t = torch.tensor(src_pos_list, dtype=torch.int32, device=kv_cache.device)
    tgt_pos_t = torch.tensor(tgt_pos_list, dtype=torch.int32, device=kv_cache.device)

    if HAS_TRITON and kv_cache.is_cuda:
        block_half = triton.next_power_of_2(half_dim)
        grid = (num_pairs, num_heads)

        _nope_permute_paged_kernel[grid](
            kv_cache,
            src_kv,
            block_table,
            src_pos_t,
            tgt_pos_t,
            inv_freq,
            num_pairs=num_pairs,
            num_heads=num_heads,
            block_size=block_size,
            head_dim=head_dim,
            half_dim=half_dim,
            num_blocks_total=num_blocks_total,
            num_blocks_per_seq=num_blocks_per_seq,
            kv_stride_block=kv_cache.stride(0),
            kv_stride_kv=kv_cache.stride(1),
            kv_stride_head=kv_cache.stride(2),
            kv_stride_pos=kv_cache.stride(3),
            kv_stride_dim=kv_cache.stride(4),
            s_stride_pair=src_kv.stride(0),
            s_stride_head=src_kv.stride(1),
            s_stride_dim=src_kv.stride(2),
            BLOCK_HALF=block_half,
        )
    else:
        _nope_permute_paged_kv_cpu(
            kv_cache, src_kv, block_table,
            src_pos_t, tgt_pos_t, inv_freq,
            block_size, head_dim,
        )


def apply_rope_delta_inplace(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    positions: list[int],
    delta: int,
    rope_base: float = 10000.0,
) -> int:
    """Apply RoPE(delta) rotation to K values at specified positions in-place.

    For each position p, transforms K:
        K[p] = RoPE(delta) × K[p]

    After this call, K at position p has RoPE encoding as if it were
    originally computed at position (p + delta). V is NOT modified.

    Used for SEMBLEND_FORCE_DELTA E2E validation: artificially corrupt K
    positions to demonstrate RoPE correction necessity. Applying delta=Δ
    then delta=-Δ is an identity (RoPE group property).

    Auto-detects KV cache layout:
      - vLLM flash_attn: [2, num_blocks, block_size, num_kv_heads, head_dim]
      - Standard:        [num_blocks, 2, num_heads, block_size, head_dim]

    Args:
        kv_cache: Per-layer KV cache tensor (either layout).
        block_table: Block table [num_blocks_per_seq] int32/int64.
        positions: Logical positions to modify.
        delta: RoPE offset to apply (positive or negative).
        rope_base: RoPE base frequency.

    Returns:
        Number of positions modified.
    """
    if delta == 0 or not positions:
        return 0

    device = kv_cache.device

    # Auto-detect layout
    # vLLM: [2, num_blocks, block_size, num_kv_heads, head_dim]
    # Standard: [num_blocks, 2, num_heads, block_size, head_dim]
    is_vllm_layout = kv_cache.shape[0] == 2 and (
        kv_cache.ndim == 5 and kv_cache.shape[1] != 2
    )

    if is_vllm_layout:
        head_dim = kv_cache.shape[4]
        block_size = kv_cache.shape[2]
    else:
        head_dim = kv_cache.shape[4]
        block_size = kv_cache.shape[3]

    inv_freq = 1.0 / (
        rope_base
        ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
            / head_dim
        )
    )

    theta = float(delta) * inv_freq
    cos_v = torch.cos(theta)
    sin_v = torch.sin(theta)

    modified = 0
    for pos in positions:
        blk_idx = pos // block_size
        if blk_idx >= block_table.shape[0]:
            continue
        blk = int(block_table[blk_idx])
        off = pos % block_size

        if is_vllm_layout:
            k = kv_cache[0, blk, off, :, :].float()
        else:
            k = kv_cache[blk, 0, :, off, :].float()

        k_even = k[:, 0::2].clone()
        k_odd = k[:, 1::2].clone()

        k[:, 0::2] = k_even * cos_v - k_odd * sin_v
        k[:, 1::2] = k_even * sin_v + k_odd * cos_v

        if is_vllm_layout:
            kv_cache[0, blk, off, :, :] = k.to(kv_cache.dtype)
        else:
            kv_cache[blk, 0, :, off, :] = k.to(kv_cache.dtype)

        modified += 1

    return modified


def _nope_permute_paged_kv_cpu(
    kv_cache: torch.Tensor,
    src_kv: torch.Tensor,
    block_table: torch.Tensor,
    src_positions: torch.Tensor,
    tgt_positions: torch.Tensor,
    inv_freq: torch.Tensor,
    block_size: int,
    head_dim: int,
) -> None:
    """CPU fallback for NoPE two-step K correction."""
    for i in range(src_positions.shape[0]):
        s_pos = float(src_positions[i])
        t_pos = float(tgt_positions[i])

        tb = int(block_table[int(t_pos) // block_size])
        to_ = int(t_pos) % block_size

        k = src_kv[i].float()  # [num_heads, head_dim]

        # Step 1: strip RoPE(-s_pos) → K_raw
        theta_s = s_pos * inv_freq
        cos_s = torch.cos(theta_s)
        sin_s = torch.sin(theta_s)
        k_even = k[:, 0::2].clone()
        k_odd = k[:, 1::2].clone()
        k_raw_even = k_even * cos_s + k_odd * sin_s
        k_raw_odd = -k_even * sin_s + k_odd * cos_s

        # Step 2: apply RoPE(t_pos) → K_target
        theta_t = t_pos * inv_freq
        cos_t = torch.cos(theta_t)
        sin_t = torch.sin(theta_t)
        k_out = torch.zeros_like(k)
        k_out[:, 0::2] = k_raw_even * cos_t - k_raw_odd * sin_t
        k_out[:, 1::2] = k_raw_even * sin_t + k_raw_odd * cos_t

        kv_cache[tb, 0, :, to_, :] = k_out.to(kv_cache.dtype)

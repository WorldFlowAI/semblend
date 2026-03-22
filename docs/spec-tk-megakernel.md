# TK Fused Partial-Prefill Megakernel — Technical Specification

## Status: DRAFT
## Target Hardware: NVIDIA H100 (sm_90), B200 (sm_100)

---

## 1. Motivation

SemBlend's partial prefill currently executes as **4 separate GPU operations** with
intermediate global memory round-trips between each:

```
┌──────────────────────┐   global mem   ┌──────────────────────┐
│ 1. scatter_donor_kv  │ ─────────────► │ KV cache (written)   │
└──────────────────────┘                └──────────────────────┘
                                                 │ read back
┌──────────────────────┐   global mem   ┌────────▼─────────────┐
│ 2. rope_correct_k    │ ─────────────► │ K cache (corrected)  │
└──────────────────────┘                └──────────────────────┘
                                                 │ read back
┌──────────────────────┐   global mem   ┌────────▼─────────────┐
│ 3. masked_qkv_proj   │ ─────────────► │ Q,K,V (compute only) │
└──────────────────────┘                └──────────────────────┘
                                                 │ read back
┌──────────────────────┐   global mem   ┌────────▼─────────────┐
│ 4. partial_attention  │ ─────────────► │ output (compute only)│
└──────────────────────┘                └──────────────────────┘
```

**Problems with current approach:**
- 4 kernel launches (each ~5-10us overhead on H100)
- 3 unnecessary global memory round-trips for intermediate K,V data
- Triton can't access H100-specific features (TMA, WGMMA, tensor memory)
- Partial attention kernel scans compute_mask sequentially per output position
- No shared memory reuse across stages

**Goal:** Fuse all 4 stages into a single ThunderKittens kernel that:
1. Loads donor KV + hidden states once from global memory
2. Applies RoPE correction in registers
3. Computes masked QKV projection using WGMMA
4. Runs partial causal attention with online softmax
5. Writes final output once to global memory

---

## 2. Current Kernel Analysis

### 2.1 Stage 1: Scatter Donor KV

```
Grid: (num_pairs, num_layers * 2 * num_heads)
Per program: copy head_dim elements for one (layer, K/V, head, position pair)
Memory: read donor[d_pos], write target[t_pos] — pure copy, no compute
```

**Bottleneck:** Memory bandwidth bound. Each program reads/writes 128-256 bytes.
For 4K reuse pairs, 32 layers, 32 heads: 8M programs. Massively parallel but
each does trivial work.

### 2.2 Stage 2: RoPE Correction

```
Grid: (num_pairs, 2 * num_heads) — for paged variant
Per program: compute cos/sin for delta, rotate K[even/odd] pairs
Compute: 2 trig evals + 4 FMAs per dimension pair
Memory: read K (already in cache from scatter), write corrected K back
```

**Bottleneck:** Compute bound per-element (trig functions), but very little total
work. ~7us for 8K tokens. Already fused with scatter in `rope_correct_scatter_paged`.

### 2.3 Stage 3: Masked QKV Projection

```
Grid: (cdiv(seq_len, BLOCK_SEQ), cdiv(out_features, BLOCK_OUT))
Per program: tiled GEMM for [BLOCK_SEQ, in_features] @ [in_features, BLOCK_OUT]
Compute: in_features * BLOCK_SEQ * BLOCK_OUT FMAs per output tile
Memory: read hidden_states + qkv_weight, write QKV output
Masking: zero out reuse positions after GEMM (wasteful — computes then discards)
```

**Bottleneck:** This is the most expensive stage. For Qwen2.5-7B:
- hidden_dim = 3584, num_heads = 28, head_dim = 128
- out_features = 3 * 28 * 128 = 10752
- Per compute-position: 3584 * 10752 = 38.5M FMAs
- At 60% reuse: only 40% of seq_len needs this, but the current kernel
  still loads weights for all positions (weight tiles are shared)

**Key optimization opportunity:** Skip entire BLOCK_SEQ tiles where all positions
are reuse. Currently zeros them out post-compute; TK can skip the WGMMA entirely.

### 2.4 Stage 4: Partial Prefill Attention

```
Grid: (num_compute_positions, num_heads)
Per program: causal attention for one (head, compute_position)
  - Scan compute_mask to find actual position (O(seq_len) scan!)
  - Load Q vector
  - Iterate over K blocks: QK^T with online softmax
  - Iterate over V blocks: weighted accumulate
Memory: read Q[1, head_dim], K[1..pos, head_dim], V[1..pos, head_dim]
```

**Bottleneck:** The sequential mask scan is O(seq_len) per compute position.
For 8K sequence with 40% compute (3200 positions), that's 3200 * 8192 = 26M
comparisons just for position lookup. This is pure overhead.

The attention itself is memory-bound (loading K,V tiles from global memory for
each compute position independently — no K,V sharing across compute positions
within a head).

---

## 3. Fused Megakernel Design

### 3.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                 TK Partial Prefill Megakernel                    │
│                                                                  │
│  Grid: (num_heads, num_layers)                                  │
│  Block: 4 warps (128 threads) = 1 warpgroup                    │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Phase A: Load + Scatter + RoPE (shared memory staging)     │ │
│  │                                                             │ │
│  │  FOR each tile of positions [TILE_POS = 64]:               │ │
│  │    1. Load compute_mask tile → registers                    │ │
│  │    2. IF reuse positions in tile:                           │ │
│  │       a. TMA load donor K,V from global → shared           │ │
│  │       b. RoPE correct K in registers (cos/sin from delta)  │ │
│  │       c. Write corrected K + V to KV cache (global)        │ │
│  │    3. Prefetch next tile donor/hidden via TMA               │ │
│  └────────────────────────────────────────────────────────────┘ │
│                          │                                       │
│  ┌────────────────────────▼───────────────────────────────────┐ │
│  │ Phase B: Masked QKV Projection (WGMMA)                     │ │
│  │                                                             │ │
│  │  FOR each compute-position in tile:                        │ │
│  │    1. Load hidden_states[pos] → shared tile [1, hidden_dim]│ │
│  │    2. WGMMA: Q,K,V = hidden @ W_qkv                       │ │
│  │       (weight tiles stay in shared mem across positions)    │ │
│  │    3. Store new K,V to KV cache (overwrites at compute pos)│ │
│  │    4. Keep Q in registers for Phase C                      │ │
│  └────────────────────────────────────────────────────────────┘ │
│                          │                                       │
│  ┌────────────────────────▼───────────────────────────────────┐ │
│  │ Phase C: Partial Causal Attention (WGMMA + online softmax) │ │
│  │                                                             │ │
│  │  FOR each compute-position (Q in registers):               │ │
│  │    FOR k_tile in [0, actual_pos] step TILE_K:              │ │
│  │      1. TMA load K_tile[TILE_K, head_dim] from KV cache    │ │
│  │      2. WGMMA: scores = Q @ K_tile^T (1×TILE_K)           │ │
│  │      3. Causal mask + online softmax update                 │ │
│  │      4. TMA load V_tile[TILE_K, head_dim]                  │ │
│  │      5. WGMMA: acc += softmax_weights @ V_tile             │ │
│  │    6. Normalize acc, write output[pos]                      │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Output: final_output[compute_positions, head_dim]              │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 TK Type Mappings

```cuda
#include "kittens.cuh"
using namespace kittens;

// Core tile types (16x16 native, composed for larger)
using q_tile    = rt_bf<1, 8>;    // [16, 128] — one Q vector per warp (head_dim=128)
using k_tile    = rt_bf<4, 8>;    // [64, 128] — TILE_K=64 K vectors
using v_tile    = rt_bf<4, 8>;    // [64, 128] — TILE_K=64 V vectors
using score_tile = rt_fl<1, 4>;   // [16, 64]  — QK^T scores (float32 accum)
using acc_tile  = rt_fl<1, 8>;    // [16, 128] — attention output accumulator

// Shared memory tiles for staging
using k_smem    = st_bf<4, 8>;    // [64, 128] in shared
using v_smem    = st_bf<4, 8>;    // [64, 128] in shared
using h_smem    = st_bf<1, 224>;  // [16, 3584] hidden states (Qwen2.5-7B hidden_dim)
using w_smem    = st_bf<224, 8>;  // [3584, 128] weight tile (one head's QKV slice)

// Position metadata (in registers)
using mask_reg = rt_fl<1, 4>;     // [16, 64] — compute mask bits

// RoPE precomputed tables (in shared memory)
using rope_smem = st_fl<1, 4>;    // [16, 64] — inv_freq table (half_dim=64)
```

### 3.3 Memory Traffic Comparison

**Model config:** Qwen2.5-7B (hidden=3584, heads=28, kv_heads=4, head_dim=128, layers=28)
**Scenario:** seq_len=4096, 60% reuse (2458 reuse, 1638 compute), paged cache

#### Current (4 separate kernels):

| Stage | Read (MB) | Write (MB) | Notes |
|-------|-----------|------------|-------|
| Scatter | 2458 * 128 * 2B = 0.6 | 0.6 | Per layer×head; ×28 layers ×4 kv_heads |
| RoPE | 0.6 (re-read K) | 0.6 | K only, same positions |
| QKV proj | 4096 * 3584 * 2B = 28.0 | 4096 * 10752 * 2B = 84.0 | Computes all, zeros reuse |
| Attention | 1638 * avg(2048) * 128 * 2B = ~500 | 1638 * 128 * 2B = 0.4 | K,V re-read per compute pos |
| **Total** | **~530 MB** | **~86 MB** | |

**Global mem round-trip overhead:** ~170 MB (intermediate K,V,QKV written then re-read)

#### Fused TK megakernel:

| Phase | Read (MB) | Write (MB) | Notes |
|-------|-----------|------------|-------|
| A: Scatter+RoPE | 0.6 | 0.6 | Donor → cache, one pass |
| B: QKV proj | 1638 * 3584 * 2B = 11.2 | 1638 * 128 * 2B * 3 = 1.2 | Skip reuse positions entirely |
|  | + weights: 3584 * 10752 * 2B = 73.0 | | Weight read (amortized via shared mem) |
| C: Attention | same ~500 | 0.4 | K,V from cache (same as current) |
| **Total** | **~586 MB** | **~2.2 MB** | |

**Savings:**
- **Write reduction: ~84 MB → ~2.2 MB** (38x less writes) — no intermediate QKV buffer
- **Read reduction: ~530 → ~586 MB** (slight increase from weight amortization, but
  eliminates 170 MB of intermediate re-reads)
- **Net memory traffic saved: ~170 MB** (intermediate buffers eliminated)
- **Kernel launch overhead: 4 × 5-10us → 1 × 5-10us** = 15-30us saved

### 3.4 Compute Savings

**Masked QKV projection (biggest win):**
- Current: computes QKV for ALL 4096 positions, zeros out 2458 → wastes 60% of GEMM FLOPs
- Fused: skips WGMMA tiles entirely when all positions in tile are reuse
- At 60% reuse with TILE_POS=64: ~60% of tiles skipped → **~2.5x fewer GEMM FLOPs**

**Position mask scan (eliminated):**
- Current: O(seq_len) sequential scan per compute position in attention kernel
- Fused: precompute compact position list in Phase A → O(1) lookup in Phase C
- Saves: 1638 * 4096 mask comparisons = 6.7M ops (minor but eliminates serial bottleneck)

---

## 4. Kernel Implementation

### 4.1 Entry Point

```cuda
// tk_partial_prefill.cu

#include "kittens.cuh"
using namespace kittens;

// Compile-time config
constexpr int HEAD_DIM = 128;     // Fixed for Qwen/LLaMA
constexpr int TILE_K = 64;        // K,V tile height (attention)
constexpr int TILE_POS = 64;      // Position tile for scatter/projection
constexpr int NUM_WARPS = 4;      // = 1 warpgroup for WGMMA

// Per-head, per-layer kernel
template<int HIDDEN_DIM, bool HAS_ROPE>
__global__ void tk_partial_prefill_kernel(
    // Inputs
    const bf16*  __restrict__ hidden_states,   // [seq_len, HIDDEN_DIM]
    const bf16*  __restrict__ donor_kv,         // [2, num_kv_heads, donor_len, HEAD_DIM]
    const bf16*  __restrict__ qkv_weight,       // [3*HEAD_DIM, HIDDEN_DIM] (per-head slice)
    const float* __restrict__ inv_freq,         // [HEAD_DIM/2]

    // Position metadata
    const int32_t* __restrict__ compute_positions,  // [num_compute] — compact list
    const int32_t* __restrict__ donor_positions,     // [num_reuse]
    const int32_t* __restrict__ target_positions,    // [num_reuse]

    // Outputs
    bf16*  __restrict__ kv_cache,               // [num_blocks, 2, num_kv_heads, block_size, HEAD_DIM]
    bf16*  __restrict__ output,                 // [seq_len, HEAD_DIM] — only compute positions written
    const int32_t* __restrict__ block_table,    // [num_blocks_per_seq]

    // Dimensions
    int seq_len,
    int num_compute,
    int num_reuse,
    int block_size,
    int num_blocks_per_seq,

    // Attention
    float scale
) {
    int head_idx  = blockIdx.x;   // which attention head
    int layer_idx = blockIdx.y;   // which transformer layer

    // ── Shared memory allocation ──
    extern __shared__ char smem_raw[];
    // Layout:
    //   [0, k_smem_bytes)          — k_tile staging
    //   [k_smem_bytes, +v_bytes)   — v_tile staging
    //   [+v_bytes, +h_bytes)       — hidden states tile
    //   [+h_bytes, +w_bytes)       — weight tile (persistent across positions)

    auto& k_shared = *(k_smem*)(smem_raw);
    auto& v_shared = *(v_smem*)(smem_raw + sizeof(k_smem));

    // ── Phase A: Scatter + RoPE ──
    // Each warp handles a subset of reuse pairs
    for (int pair = threadIdx.x / 32; pair < num_reuse; pair += NUM_WARPS) {
        int d_pos = donor_positions[pair];
        int t_pos = target_positions[pair];

        // Load donor K,V for this head
        // ... TMA async load into shared ...

        if constexpr (HAS_ROPE) {
            // RoPE correction on K in registers
            int delta = t_pos - d_pos;
            // Apply rotation: [cos(delta*freq), -sin(delta*freq)] x [k_even, k_odd]
            // ... register-level RoPE ...
        }

        // Write to paged KV cache
        int logical_block = t_pos / block_size;
        int block_offset  = t_pos % block_size;
        int physical_block = block_table[logical_block];
        // ... store to kv_cache[physical_block, K/V, head, offset, :] ...
    }
    __syncthreads();

    // ── Phase B: Masked QKV Projection (WGMMA) ──
    // Load weight tile into shared memory (persistent — amortized across positions)
    // W_qkv shape for this head: [3*HEAD_DIM, HIDDEN_DIM] = [384, 3584]
    // Process in HIDDEN_DIM tiles of 128

    // For each compute position:
    for (int ci = 0; ci < num_compute; ci++) {
        int pos = compute_positions[ci];

        // Load hidden_states[pos, :] into shared
        // ... TMA load [1, HIDDEN_DIM] ...

        // WGMMA: qkv = hidden @ W_qkv^T
        // Result: q[HEAD_DIM], k[HEAD_DIM], v[HEAD_DIM] in registers
        rt_bf<1, 8> q_reg, k_reg, v_reg;  // [16, 128] each

        // ... tiled WGMMA accumulation over HIDDEN_DIM ...

        // Write new K,V to KV cache at this compute position
        // ... store k_reg, v_reg to kv_cache[pos] ...

        // ── Phase C: Partial Attention (inline per compute position) ──
        // Q is already in q_reg from Phase B — no global memory round-trip!

        rt_fl<1, 8> acc;          // [16, 128] float32 accumulator
        float m_prev = -INFINITY;
        float l_prev = 0.0f;

        // Causal attention: attend to positions [0, pos]
        for (int k_start = 0; k_start <= pos; k_start += TILE_K) {
            int tile_len = min(TILE_K, pos + 1 - k_start);

            // TMA load K tile from KV cache
            // ... async load K[k_start:k_start+TILE_K, HEAD_DIM] → k_shared ...

            // WGMMA: scores = q_reg @ k_shared^T → [1, TILE_K]
            rt_fl<1, 4> scores;  // [16, 64]
            // ... WGMMA matmul ...

            // Scale + causal mask
            mul(scores, scores, scale);
            // ... mask positions > pos to -inf ...

            // Online softmax
            float m_new = max(m_prev, row_max(scores));
            // ... exp, correction, accumulate ...

            // TMA load V tile from KV cache
            // ... async load V[k_start:k_start+TILE_K, HEAD_DIM] → v_shared ...

            // WGMMA: acc += softmax_weights @ V_tile
            // ... WGMMA matmul accumulate ...
        }

        // Normalize and write output
        div(acc, acc, l_prev);
        // ... store acc → output[pos, :] as bf16 ...
    }
}
```

### 4.2 Shared Memory Budget (H100)

H100 provides **228 KB** shared memory per SM (configurable up to 228KB with
`cudaFuncSetAttribute`).

| Buffer | Size | Notes |
|--------|------|-------|
| K tile (TILE_K=64, HEAD_DIM=128, bf16) | 64 * 128 * 2 = 16 KB | Double-buffered: 32 KB |
| V tile (same) | 16 KB | Double-buffered: 32 KB |
| Hidden states tile (1 pos) | 1 * 3584 * 2 = 7 KB | Single-buffered |
| Weight tile (HEAD_DIM slice) | 128 * 128 * 2 = 32 KB | Tiled over HIDDEN_DIM |
| RoPE inv_freq | 64 * 4 = 256 B | Persistent |
| Position metadata | ~2 KB | Compute list, reuse list |
| **Total** | **~105 KB** | Well within H100 228KB limit |

Double-buffering K,V tiles enables overlapping TMA loads with WGMMA compute.

### 4.3 Warpgroup Decomposition

```
Warpgroup (4 warps):
  Warp 0-3: Collaborative WGMMA for matmul

Phase A (scatter+RoPE):
  Warp 0: pairs [0, N/4)
  Warp 1: pairs [N/4, N/2)
  Warp 2: pairs [N/2, 3N/4)
  Warp 3: pairs [3N/4, N)
  __syncthreads()

Phase B+C (projection + attention — fused per position):
  All 4 warps: collaborative WGMMA on single position
  Warp 0: issues TMA for next K/V tile (producer)
  Warp 1-3: consume current tile WGMMA (consumers)
```

---

## 5. Paged KV Cache Handling

The paged cache adds complexity: logical positions map to physical blocks via
`block_table`. The megakernel must handle this translation inline.

```cuda
// Paged cache address translation (inline helper)
__device__ __forceinline__
int paged_offset(
    const int32_t* block_table,
    int logical_pos,
    int block_size,
    int kv_idx,           // 0=K, 1=V
    int head_idx,
    int num_kv_heads,
    int head_dim
) {
    int logical_block = logical_pos / block_size;
    int block_offset  = logical_pos % block_size;
    int physical_block = block_table[logical_block];

    // Layout: [num_blocks, 2, num_kv_heads, block_size, head_dim]
    return physical_block * (2 * num_kv_heads * block_size * head_dim)
         + kv_idx * (num_kv_heads * block_size * head_dim)
         + head_idx * (block_size * head_dim)
         + block_offset * head_dim;
}
```

For TMA loads of K,V tiles from paged cache during attention (Phase C), tiles may
span block boundaries. Two strategies:

**Option A — Gather into contiguous shared buffer:**
For each tile position, compute physical address, load individual rows.
Pro: Simple. Con: Can't use TMA bulk copy for non-contiguous rows.

**Option B — Virtual-to-physical tile mapping:**
Pre-sort compute positions by physical block. Process positions within the same
physical block together. Pro: Enables TMA. Con: Changes attention order (may
affect causal mask logic).

**Recommendation:** Option A for v1 (correctness first), Option B for v2 (perf).

---

## 6. GQA (Grouped Query Attention) Support

Qwen2.5-7B uses GQA: 28 attention heads, 4 KV heads (7:1 ratio).
Multiple Q heads share the same K,V head.

**Impact on megakernel:**
- Phase A (scatter): Grid over kv_heads (4), not q_heads (28)
- Phase B (projection): Q projects to 28 heads, K,V to 4 heads
  - Split weight into W_q [28*128, 3584] and W_kv [2*4*128, 3584]
  - WGMMA for Q: 28 head outputs
  - WGMMA for K,V: 4 head outputs (shared across 7 Q heads)
- Phase C (attention): Grid over q_heads (28), each loads from its kv_head group

```
Grid: (num_q_heads=28, num_layers=28)
  Phase A: if (head_idx < num_kv_heads) → scatter/RoPE for this KV head
           else → skip (barrier sync only)
  Phase B: all heads compute Q; kv_head = head_idx / gqa_ratio computes K,V
  Phase C: all heads compute attention using K,V from their kv_head group
```

---

## 7. Layer-Level Bathtub Skip

The bathtub curve may flag some layers for full recomputation. The megakernel
should support this:

```cuda
// Per-layer recompute flag (passed as kernel arg)
const bool* __restrict__ layer_recompute_flags;  // [num_layers]

// In kernel:
if (layer_recompute_flags[layer_idx]) {
    // Skip entirely — vLLM standard prefill handles this layer
    return;
}
```

This means the megakernel grid is launched for ALL layers but immediately returns
for recompute layers. Alternative: compact the grid to only non-recompute layers
and pass a layer index mapping.

---

## 8. Python Binding

```python
# tk_partial_prefill.py — Python wrapper

import torch
from torch.utils.cpp_extension import load

# Compile TK kernel
_tk_module = load(
    name="tk_partial_prefill",
    sources=["kernels/tk_partial_prefill.cu"],
    extra_include_paths=["ThunderKittens/include"],
    extra_cuda_cflags=[
        "-std=c++20",
        "-arch=sm_90a",  # H100
        "--expt-relaxed-constexpr",
        "-O3",
    ],
)


def tk_partial_prefill(
    hidden_states: torch.Tensor,     # [seq_len, hidden_dim] bf16
    donor_kv: torch.Tensor,          # [2, num_kv_heads, donor_len, head_dim] bf16
    qkv_weight: torch.Tensor,        # [3*num_heads*head_dim, hidden_dim] bf16
    kv_cache: torch.Tensor,          # [num_blocks, 2, num_kv_heads, block_size, head_dim] bf16
    block_table: torch.Tensor,       # [num_blocks_per_seq] int32
    compute_positions: torch.Tensor,  # [num_compute] int32 — compact list
    donor_positions: torch.Tensor,    # [num_reuse] int32
    target_positions: torch.Tensor,   # [num_reuse] int32
    layer_recompute: torch.Tensor,   # [num_layers] bool
    inv_freq: torch.Tensor,          # [head_dim/2] float32
    scale: float,
    num_q_heads: int,
    num_kv_heads: int,
    block_size: int,
) -> torch.Tensor:
    """Fused partial prefill: scatter + RoPE + projection + attention.

    Returns:
        output: [seq_len, num_heads * head_dim] bf16.
            Only compute_positions have meaningful values.
    """
    return _tk_module.partial_prefill(
        hidden_states, donor_kv, qkv_weight, kv_cache,
        block_table, compute_positions, donor_positions,
        target_positions, layer_recompute, inv_freq,
        scale, num_q_heads, num_kv_heads, block_size,
    )
```

### 8.1 Integration with vLLM Hook

```python
# In model_runner_hook.py — drop-in replacement

class PartialAttentionHook:
    def apply_to_kv_cache(self, kv_cache, hidden_states):
        if HAS_TK and torch.cuda.get_device_capability()[0] >= 9:
            # Use fused TK megakernel (H100+)
            output = tk_partial_prefill(
                hidden_states=hidden_states,
                donor_kv=self._donor_kv,
                qkv_weight=self._qkv_weight,
                kv_cache=kv_cache,
                block_table=self._block_table,
                compute_positions=self._compute_positions,
                donor_positions=self._donor_positions,
                target_positions=self._target_positions,
                layer_recompute=self._layer_recompute_flags,
                inv_freq=self._inv_freq,
                scale=self._scale,
                num_q_heads=self._num_q_heads,
                num_kv_heads=self._num_kv_heads,
                block_size=self._block_size,
            )
            return output
        else:
            # Fallback to existing Triton kernels (A10G, T4)
            return self._triton_path(kv_cache, hidden_states)
```

---

## 9. Performance Estimates

### 9.1 Projected Latency (H100, Qwen2.5-7B, seq_len=4096, 60% reuse)

| Component | Current (Triton) | Fused (TK) | Savings |
|-----------|-----------------|-------------|---------|
| Kernel launch overhead | 4 * 8us = 32us | 1 * 8us = 8us | 24us |
| Scatter + RoPE | ~50us | ~30us | 20us (fused, no intermediate write) |
| QKV projection | ~200us | ~80us | 120us (skip 60% of GEMM tiles) |
| Attention | ~800us | ~600us | 200us (no mask scan, better TMA pipelining) |
| Intermediate memory | ~170 MB @ 2TB/s = 85us | 0 | 85us |
| **Total** | **~1167us** | **~718us** | **~449us (38% faster)** |

### 9.2 Scaling with Reuse Ratio

| Reuse % | Current (us) | Fused TK (us) | Speedup |
|---------|-------------|---------------|---------|
| 20% | 1600 | 1200 | 1.33x |
| 40% | 1300 | 900 | 1.44x |
| 60% | 1100 | 700 | 1.57x |
| 80% | 800 | 400 | 2.00x |
| 90% | 600 | 250 | 2.40x |

Higher reuse ratios benefit more because:
1. More GEMM tiles skipped in projection (Phase B)
2. Fewer compute positions → less attention work
3. Fixed overhead (kernel launch, weight load) amortized over more savings

### 9.3 Impact on End-to-End TTFT

For a SemBlend hit with 60% reuse on 4K context:
- Current TTFT: ~800ms (dominated by KV retrieval from LMCache, not kernels)
- Kernel portion of TTFT: ~1.2ms (0.15% of total)
- Fused savings: ~0.45ms

**Honest assessment:** The megakernel saves ~0.45ms per prefill on H100. This is
meaningful at scale (millions of requests) but not a game-changer for single-request
latency. The real wins are:

1. **Longer contexts** (16K-128K): Attention is O(n^2), so savings grow quadratically
2. **Higher throughput**: Fewer kernel launches → better GPU utilization under load
3. **Correctness**: Eliminating intermediate buffers removes a class of race conditions
4. **B200**: TK's Blackwell support (TCGEN05, tensor memory) gives another 2x on top

---

## 10. Implementation Plan

### Phase 1: Standalone Attention Kernel (2-3 weeks)
- Implement Phase C only (partial causal attention) in TK
- Benchmark against Triton `partial_prefill_attention`
- Validate numerical correctness (bitwise comparison at float32)
- Target: prove TK gives >30% speedup on attention alone

### Phase 2: Fused Scatter+RoPE+Attention (2-3 weeks)
- Add Phase A (scatter + RoPE in registers)
- Fuse with Phase C (Q from scatter, not from global memory... but Q comes from
  projection, so this is really about K,V being in shared memory already)
- Actually: fuse scatter → immediate K,V availability in shared mem for attention
- Target: eliminate the scatter → global write → attention global read round-trip

### Phase 3: Full Megakernel with Projection (3-4 weeks)
- Add Phase B (masked QKV projection via WGMMA)
- Fuse Q from projection → attention (register transfer, no global memory)
- Handle GQA head grouping
- Paged cache address translation
- Target: full fused pipeline, benchmark end-to-end

### Phase 4: Production Integration (2-3 weeks)
- Python bindings (PyBind11)
- vLLM hook integration with runtime dispatch (TK for sm_90+, Triton for sm_75-86)
- Bathtub layer-skip support
- Edge cases: empty reuse set, empty compute set, single-token sequences
- CI: correctness tests against Triton reference implementation

### Phase 5: B200 Optimization (2-3 weeks)
- Port to sm_100 with TCGEN05 tensor core instructions
- Use tensor memory (256KB) for persistent weight storage
- CTA pairs for inter-SM coordination
- NVFP4 precision for weight tiles (mixed precision)

**Total estimated effort: 11-16 weeks** for a senior CUDA engineer familiar with
TK abstractions.

---

## 11. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| TK API instability (v2 just released) | Build breaks | Pin TK version, vendor the include dir |
| Paged cache non-contiguous TMA | Phase C slower than expected | Option A gather for v1, optimize later |
| GQA head grouping complexity | Warp divergence in Phase B | Separate Q and KV projection kernels |
| Shared memory pressure | Limits TILE_K or requires spilling | Profile and tune tile sizes per model |
| Numerical precision (bf16 accum) | Quality regression vs float32 Triton | Accumulate in float32, cast on store |
| H100-only limits adoption | Most users on A10G | Keep Triton fallback as default path |

---

## 12. Alternative: Incremental TK Adoption

If the full megakernel is too ambitious, a staged approach that replaces individual
Triton kernels with TK equivalents:

1. **TK partial attention only** — highest ROI, replace the O(n) mask scan +
   attention with TK's WGMMA-based attention. Keep scatter/RoPE/projection in Triton.
   Effort: 3-4 weeks. Expected speedup: 20-30% on attention kernel.

2. **TK fused scatter+RoPE** — combine two small kernels. Low effort, modest gain.
   Effort: 1-2 weeks. Expected speedup: eliminates one kernel launch + one round-trip.

3. **TK masked GEMM** — replace Triton `masked_qkv_projection` with WGMMA-based
   projection that truly skips reuse tiles (vs compute-then-zero).
   Effort: 2-3 weeks. Expected speedup: proportional to reuse ratio (up to 2.5x).

**Recommended starting point:** Option 1 (TK partial attention). It has the highest
compute intensity, benefits most from WGMMA, and can be validated independently.

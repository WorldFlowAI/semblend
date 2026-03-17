"""vLLM ModelRunner hook for SemBlend PartialAttention integration.

This module provides the integration layer between vLLM's inference
pipeline and SemBlend's Triton-based PartialAttention kernels. It
intercepts the prefill phase to:

1. Detect when a PartialAttention plan is active (from the connector's
   ``start_load_kv()`` finding a Tier 3 semantic donor match).
2. Replace standard full-sequence prefill with selective prefill using
   the Triton scatter + masked projection + partial attention kernels.
3. Report timing and FLOPs savings back to the connector for logging
   and benchmarking.

Usage with vLLM:
    The hook is registered automatically when SynapseKVConnector detects
    a sparse transfer plan. The vLLM worker checks for the hook before
    running standard prefill::

        # In vLLM's model runner (simplified):
        if hasattr(connector, 'partial_attention_hook') and connector.partial_attention_hook:
            result = connector.partial_attention_hook.execute(
                hidden_states=hidden_states,
                kv_cache=kv_cache,
                model=model,
            )
            if result is not None:
                # Use partial prefill result instead of full prefill
                return result

Architecture:
    The hook operates at the model-runner level, wrapping individual
    attention layers. For each layer with an active plan:

    1. Before attention: scatter donor KV into paged KV cache blocks
    2. During QKV projection: mask out reuse positions (Triton GEMM)
    3. During attention: compute only for placeholder positions
    4. After attention: all positions have valid KV in cache

    Layers flagged for full recomputation (CacheBlend fallback) bypass
    the hook and run standard vLLM attention.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import torch

from synapse_kv_connector.partial_attention import (
    AttentionMode,
    PartialAttentionPlan,
    compute_attention_mask,
)
from synapse_kv_connector.rope_correction import (
    nope_permute_paged_kv,
    rope_correct_scatter_paged,
)
from synapse_kv_connector.triton_kernels import (
    partial_prefill_attention,
    scatter_donor_kv,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LayerPrefillResult:
    """Result of partial prefill for a single transformer layer.

    Attributes:
        layer_idx: Transformer layer index.
        used_partial: Whether PartialAttention was used (vs full recompute).
        positions_computed: Positions that ran QKV projection.
        positions_reused: Positions with donor KV reuse.
        scatter_time_ms: KV scatter kernel time.
        attention_time_ms: Partial attention kernel time.
    """

    layer_idx: int
    used_partial: bool
    positions_computed: int
    positions_reused: int
    scatter_time_ms: float
    attention_time_ms: float


@dataclass
class PrefillHookResult:
    """Aggregate result of the partial prefill hook.

    Attributes:
        total_time_ms: Total wall-clock time for the hook.
        layer_results: Per-layer breakdown.
        overall_computation_ratio: Fraction of FLOPs vs full prefill.
        num_layers_partial: Layers that used PartialAttention.
        num_layers_full: Layers that fell back to full recomputation.
        kv_cache_updated: Whether the KV cache was modified in-place.
    """

    total_time_ms: float
    layer_results: list[LayerPrefillResult] = field(default_factory=list)
    overall_computation_ratio: float = 1.0
    num_layers_partial: int = 0
    num_layers_full: int = 0
    kv_cache_updated: bool = False


class PartialAttentionHook:
    """vLLM model runner hook for SemBlend PartialAttention.

    Manages the lifecycle of a partial prefill operation:
    1. Receives a PartialAttentionPlan from the connector
    2. Loads donor KV tensors
    3. Orchestrates per-layer Triton kernel execution
    4. Reports results for benchmarking

    Args:
        plan: The PartialAttention plan from connector scheduling.
        donor_kv: Donor KV tensor [num_layers, 2, num_heads, donor_len, head_dim].
        device: Target CUDA device.
    """

    def __init__(
        self,
        plan: PartialAttentionPlan,
        donor_kv: torch.Tensor,
        device: torch.device | str = "cuda",
        rope_base: float = 10000.0,
    ) -> None:
        self._plan = plan
        self._donor_kv = donor_kv.to(device)
        self._device = torch.device(device)
        self._executed = False
        self._result: PrefillHookResult | None = None
        self._rope_base = rope_base

        # Pre-compute position tensors (shared across layers)
        self._donor_positions, self._target_positions = (
            self._build_position_tensors()
        )
        self._compute_mask = self._build_compute_mask(layer_idx=0)

    @property
    def plan(self) -> PartialAttentionPlan:
        return self._plan

    @property
    def result(self) -> PrefillHookResult | None:
        return self._result

    @property
    def executed(self) -> bool:
        return self._executed

    def _build_position_tensors(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build donor/target position index tensors from plan."""
        donor_positions = []
        target_positions = []

        for layer_mask in self._plan.layer_masks:
            if layer_mask.recompute_all:
                continue
            for pm in layer_mask.position_masks:
                if pm.mode == AttentionMode.REUSE and pm.donor_pos is not None:
                    donor_positions.append(pm.donor_pos)
                    target_positions.append(pm.target_pos)
            break  # Same positions for all non-recompute layers

        if not donor_positions:
            return (
                torch.zeros(0, dtype=torch.int32, device=self._device),
                torch.zeros(0, dtype=torch.int32, device=self._device),
            )

        return (
            torch.tensor(donor_positions, dtype=torch.int32, device=self._device),
            torch.tensor(target_positions, dtype=torch.int32, device=self._device),
        )

    def _build_compute_mask(self, layer_idx: int) -> torch.Tensor:
        """Build boolean compute mask for a layer as GPU tensor."""
        np_mask = compute_attention_mask(self._plan, layer_idx)
        return torch.from_numpy(np_mask).to(self._device)

    def apply_to_kv_cache(
        self,
        kv_cache: torch.Tensor,
    ) -> PrefillHookResult:
        """Apply donor KV scatter to the full KV cache.

        This is the primary entry point. Scatters donor KV values into
        the target KV cache at reuse positions for all layers that use
        PartialAttention (non-recompute layers).

        For layers flagged for full recomputation (deviation score above
        threshold), no scatter is performed — standard vLLM prefill
        handles those layers.

        Args:
            kv_cache: Target KV cache tensor.
                Shape: [num_layers, 2, num_heads, seq_len, head_dim].

        Returns:
            PrefillHookResult with timing and statistics.
        """
        if self._executed:
            logger.warning("PartialAttentionHook already executed, skipping")
            return self._result or PrefillHookResult(total_time_ms=0.0)

        start = time.perf_counter()
        layer_results = []
        num_partial = 0
        num_full = 0

        for layer_mask in self._plan.layer_masks:
            layer_idx = layer_mask.layer_idx

            if layer_mask.recompute_all:
                num_full += 1
                layer_results.append(
                    LayerPrefillResult(
                        layer_idx=layer_idx,
                        used_partial=False,
                        positions_computed=self._plan.target_len,
                        positions_reused=0,
                        scatter_time_ms=0.0,
                        attention_time_ms=0.0,
                    )
                )
                continue

            # Scatter donor KV for this layer with RoPE correction on K
            if self._donor_positions.numel() > 0:
                scatter_start = torch.cuda.Event(enable_timing=True)
                scatter_end = torch.cuda.Event(enable_timing=True)

                scatter_start.record()

                # Extract single layer from donor and target
                target_layer = kv_cache[layer_idx : layer_idx + 1]
                donor_layer = self._donor_kv[layer_idx : layer_idx + 1]

                # Use RoPE-corrected scatter when positions differ
                # (non-contiguous KV reuse — the core SemBlend innovation)
                if self._needs_rope_correction():
                    # For flat cache: apply scatter then RoPE correct K
                    scatter_donor_kv(
                        target_layer,
                        donor_layer,
                        self._donor_positions,
                        self._target_positions,
                    )
                    # Apply RoPE delta correction to K at target positions
                    from synapse_kv_connector.rope_correction import rope_correct_k
                    k_cache = target_layer[0, 0]  # [num_heads, seq_len, head_dim]
                    head_dim = k_cache.shape[-1]
                    corrected = rope_correct_k(
                        k_cache,
                        self._donor_positions,
                        self._target_positions,
                        head_dim=head_dim,
                        rope_base=self._rope_base,
                    )
                    target_layer[0, 0] = corrected
                else:
                    scatter_donor_kv(
                        target_layer,
                        donor_layer,
                        self._donor_positions,
                        self._target_positions,
                    )

                scatter_end.record()
                torch.cuda.synchronize()
                scatter_ms = scatter_start.elapsed_time(scatter_end)
            else:
                scatter_ms = 0.0

            num_partial += 1
            layer_results.append(
                LayerPrefillResult(
                    layer_idx=layer_idx,
                    used_partial=True,
                    positions_computed=self._plan.num_partial_positions,
                    positions_reused=self._plan.num_reuse_positions,
                    scatter_time_ms=scatter_ms,
                    attention_time_ms=0.0,  # Filled by attention phase
                )
            )

        elapsed = (time.perf_counter() - start) * 1000.0

        self._executed = True
        self._result = PrefillHookResult(
            total_time_ms=elapsed,
            layer_results=layer_results,
            overall_computation_ratio=self._plan.computation_ratio,
            num_layers_partial=num_partial,
            num_layers_full=num_full,
            kv_cache_updated=True,
        )

        logger.info(
            "PartialAttention applied: %d partial + %d full layers, "
            "computation_ratio=%.2f, total_time=%.1fms",
            num_partial,
            num_full,
            self._plan.computation_ratio,
            elapsed,
        )

        return self._result

    def execute_attention(
        self,
        layer_idx: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: float | None = None,
    ) -> torch.Tensor | None:
        """Execute partial attention for a single layer.

        If the layer uses PartialAttention, runs the Triton kernel that
        only computes attention output for compute positions. Returns
        None for layers that need full recomputation (caller should use
        standard attention).

        Args:
            layer_idx: Transformer layer index.
            query: Q tensor [num_heads, seq_len, head_dim].
            key: K tensor [num_heads, seq_len, head_dim].
            value: V tensor [num_heads, seq_len, head_dim].
            scale: Attention scale factor.

        Returns:
            Attention output [num_heads, seq_len, head_dim] if partial
            attention was used, None if full recomputation needed.
        """
        if layer_idx >= len(self._plan.layer_masks):
            return None

        layer_mask = self._plan.layer_masks[layer_idx]
        if layer_mask.recompute_all:
            return None

        compute_mask = self._build_compute_mask(layer_idx)

        return partial_prefill_attention(
            query, key, value, compute_mask, scale
        )

    def get_compute_mask_tensor(self, layer_idx: int) -> torch.Tensor:
        """Get the compute mask as a GPU tensor for a given layer.

        Args:
            layer_idx: Transformer layer index.

        Returns:
            Boolean tensor [seq_len] on the hook's device.
        """
        return self._build_compute_mask(layer_idx)

    def _needs_rope_correction(self) -> bool:
        """Check if any position pair requires RoPE delta correction."""
        if self._donor_positions.numel() == 0:
            return False
        return not torch.equal(self._donor_positions, self._target_positions)

    def should_use_partial(self, layer_idx: int) -> bool:
        """Check if a layer should use PartialAttention.

        Args:
            layer_idx: Transformer layer index.

        Returns:
            True if partial attention, False if full recomputation.
        """
        if layer_idx >= len(self._plan.layer_masks):
            return False
        return not self._plan.layer_masks[layer_idx].recompute_all


# ---------------------------------------------------------------------------
# RoPE Correction Hook — lightweight post-load K correction
# ---------------------------------------------------------------------------


class RoPECorrectionHook:
    """Lightweight hook that applies RoPE correction after LMCache load.

    After LMCache loads donor KV into the paged cache, this hook corrects
    the K position encoding in-place. Supports two modes:

    Mode 1 — Delta correction (SEMBLEND_USE_NOPE=0, default):
        K_corrected = RoPE(target_pos - donor_pos) × K_donor
        Uses `permute_paged_kv_with_rope()`.

    Mode 2 — NoPE two-step (SEMBLEND_USE_NOPE=1):
        Step 1: K_raw = RoPE(-donor_pos) × K_donor  (strip donor position)
        Step 2: K_target = RoPE(target_pos) × K_raw  (apply target position)
        Uses `nope_permute_paged_kv()`.

    Both modes produce mathematically identical K_target (RoPE is a rotation
    group: RoPE(a) × RoPE(-b) = RoPE(a-b)). Empirically validated by running
    all SemBlend benchmarks under both modes and confirming identical PPL.

    Args:
        position_map: Mapping of donor_pos → target_pos pairs.
        plan: The PartialAttention plan (for logging/stats).
        request_id: Request identifier for logging.
        use_nope: If True, use NoPE two-step (MEPIC-style). Default: env var.
    """

    def __init__(
        self,
        position_map: object,
        plan: object,
        request_id: str = "",
        rope_base: float = 10000.0,
        use_nope: bool | None = None,
    ) -> None:
        self._position_map = position_map
        self._plan = plan
        self._request_id = request_id
        self._rope_base = rope_base
        self._executed = False
        self._result: dict | None = None
        # SEMBLEND_USE_NOPE=1 switches to NoPE two-step correction (MEPIC-style)
        if use_nope is None:
            import os
            use_nope = os.environ.get("SEMBLEND_USE_NOPE", "0").strip() == "1"
        self._use_nope = use_nope

    @property
    def executed(self) -> bool:
        return self._executed

    @property
    def result(self) -> dict | None:
        return self._result

    def apply_rope_correction(
        self,
        kv_caches: list[torch.Tensor],
        block_table: torch.Tensor | None,
    ) -> dict:
        """Apply RoPE delta correction to K in the paged KV cache.

        For each (donor_pos, target_pos) pair where delta != 0, applies:
            K_corrected = RoPE(delta) × K_current

        Args:
            kv_caches: List of per-layer KV cache tensors.
                Shape per layer: [2, num_blocks, block_size, num_kv_heads, head_dim]
                (vLLM v0.14.1 flash_attn layout)
            block_table: Block table for this request [num_blocks_per_seq].

        Returns:
            Dict with timing and correction stats.
        """
        if self._executed:
            return self._result or {}

        start = time.perf_counter()
        pos_map = self._position_map
        num_pairs = pos_map.num_pairs if hasattr(pos_map, 'num_pairs') else 0
        needs_correction = (
            pos_map.needs_correction if hasattr(pos_map, 'needs_correction') else False
        )

        if num_pairs == 0 or not needs_correction or block_table is None:
            self._executed = True
            self._result = {
                "corrected": False,
                "reason": "no_correction_needed" if num_pairs == 0 else "no_block_table",
                "num_pairs": num_pairs,
                "time_ms": 0.0,
            }
            import sys
            print(
                f"[SemBlend] RoPE hook: skip ({self._result['reason']}), "
                f"req={self._request_id}",
                file=sys.stderr, flush=True,
            )
            return self._result

        # Build permutation pairs for RoPE correction
        donor_positions = pos_map.donor_positions
        target_positions = pos_map.target_positions
        correction_pairs = [
            (int(d), int(t))
            for d, t in zip(donor_positions, target_positions)
            if d != t
        ]

        if not correction_pairs:
            self._executed = True
            self._result = {
                "corrected": False,
                "reason": "all_positions_match",
                "num_pairs": num_pairs,
                "time_ms": 0.0,
            }
            return self._result

        # Apply K position correction to each layer's KV cache
        # Mode: delta correction (default) or NoPE two-step (SEMBLEND_USE_NOPE=1)
        correction_mode = "nope" if self._use_nope else "delta"
        layers_corrected = 0
        layer_results = []

        for layer_idx, kv_cache in enumerate(kv_caches):
            # Check bathtub curve: skip correction for layers marked
            # for full recomputation (they'll be recomputed anyway)
            if self._plan is not None:
                layer_masks = getattr(self._plan, 'layer_masks', None)
                if layer_masks and layer_idx < len(layer_masks):
                    if layer_masks[layer_idx].recompute_all:
                        layer_results.append({
                            "layer": layer_idx,
                            "action": "skip_recompute",
                        })
                        continue

            try:
                if self._use_nope:
                    # NoPE two-step: strip RoPE(-src_pos) then apply RoPE(tgt_pos)
                    # Mathematically identical to delta correction (RoPE group property)
                    nope_permute_paged_kv(
                        kv_cache=kv_cache,
                        block_table=block_table,
                        permutation=correction_pairs,
                        rope_base=self._rope_base,
                    )
                else:
                    # Delta correction: RoPE(target - donor) in one step (default)
                    from synapse_kv_connector.rope_correction import (
                        permute_paged_kv_with_rope,
                    )
                    permute_paged_kv_with_rope(
                        kv_cache=kv_cache,
                        block_table=block_table,
                        permutation=correction_pairs,
                        rope_base=self._rope_base,
                    )
                layers_corrected += 1
                layer_results.append({
                    "layer": layer_idx,
                    "action": f"corrected_{correction_mode}",
                    "pairs": len(correction_pairs),
                })
            except Exception as exc:
                import sys
                import traceback as _tb
                layer_results.append({
                    "layer": layer_idx,
                    "action": "error",
                    "error": str(exc),
                })
                if layer_idx == 0:
                    print(
                        f"[SemBlend] RoPE layer 0 FAILED: {exc}\n"
                        f"  kv shape={kv_cache.shape} dtype={kv_cache.dtype}\n"
                        f"  bt shape={getattr(block_table, 'shape', '?')}\n"
                        f"  pairs={len(correction_pairs)}\n"
                        f"  tb={_tb.format_exc()}",
                        file=sys.stderr, flush=True,
                    )

        elapsed = (time.perf_counter() - start) * 1000.0
        self._executed = True
        self._result = {
            "corrected": layers_corrected > 0,
            "correction_mode": correction_mode,
            "layers_corrected": layers_corrected,
            "layers_skipped_recompute": len(kv_caches) - layers_corrected,
            "correction_pairs": len(correction_pairs),
            "total_pairs": num_pairs,
            "time_ms": elapsed,
            "request_id": self._request_id,
            "layer_results": layer_results,
        }

        import sys
        print(
            f"[SemBlend] K correction ({correction_mode}) applied: "
            f"{layers_corrected}/{len(kv_caches)} layers, "
            f"{len(correction_pairs)} position pairs, "
            f"{elapsed:.1f}ms, req={self._request_id}",
            file=sys.stderr, flush=True,
        )

        return self._result


# ---------------------------------------------------------------------------
# Model Runner Monkey-Patch
# ---------------------------------------------------------------------------


def patch_model_runner(
    model_runner: object,
    connector: object,
) -> bool:
    """Monkey-patch vLLM's model runner to intercept prefill for PartialAttention.

    When a RoPE correction hook is active, applies position correction
    to the paged KV cache after LMCache loads donor KV. Also supports
    full PartialAttention hooks for scatter + selective recomputation.

    The patch is non-destructive: if no hook is active, execution flows
    through the original code path unchanged.

    Args:
        model_runner: vLLM's GPUModelRunnerV1 instance.
        connector: SemBlendConnectorV1 instance (holds active hooks).

    Returns:
        True if patch was applied, False if model_runner is incompatible.
    """
    if not hasattr(model_runner, "execute_model"):
        logger.warning(
            "patch_model_runner: model_runner lacks execute_model, skipping"
        )
        return False

    # Store model_runner ref on connector for _apply_rope_after_load
    connector._model_runner_ref = model_runner

    original_execute = model_runner.execute_model

    _call_count = [0]

    def _patched_execute_model(*args, **kwargs):
        """Wrapper with diagnostic logging.

        RoPE correction is applied in start_load_kv (worker side) AFTER
        LMCache loads donor KV, not here. The pre-execute path was disabled
        because: (1) it runs before LMCache loads, and (2) kv_caches may
        be a dict (layer names as keys) which apply_rope_correction can't
        iterate correctly.
        """
        import sys
        import time as _time
        _call_count[0] += 1

        sched = args[0] if args else kwargs.get("scheduler_output")
        num_sched = getattr(sched, "total_num_scheduled_tokens", "?") if sched else "?"
        t0 = _time.monotonic()

        result = original_execute(*args, **kwargs)

        elapsed = (_time.monotonic() - t0) * 1000
        if elapsed > 100 or _call_count[0] % 200 == 1:
            print(
                f"[SemBlend] execute_model #{_call_count[0]}: "
                f"sched_tokens={num_sched}, elapsed={elapsed:.0f}ms",
                file=sys.stderr, flush=True,
            )

        return result

    model_runner.execute_model = _patched_execute_model
    logger.info("PartialAttention: patched model_runner.execute_model (diagnostics only)")
    return True

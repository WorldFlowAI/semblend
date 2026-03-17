"""Tests for SemBlend Triton CUDA kernels.

Tests both the Triton GPU path and the PyTorch/numpy fallback path.
GPU tests are skipped if CUDA is not available.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

# Import with graceful degradation
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch not available")

from synapse_kv_connector.triton_kernels import (
    HAS_TRITON,
    PartialPrefillResult,
    masked_qkv_projection,
    partial_prefill,
    partial_prefill_attention,
    scatter_donor_kv,
    scatter_donor_kv_paged,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _device():
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_kv_cache(
    num_layers: int = 4,
    num_heads: int = 8,
    seq_len: int = 64,
    head_dim: int = 64,
    device: str | None = None,
) -> torch.Tensor:
    """Create a zero-initialized KV cache tensor."""
    dev = device or str(_device())
    return torch.zeros(
        (num_layers, 2, num_heads, seq_len, head_dim),
        dtype=torch.float16,
        device=dev,
    )


def _make_donor_kv(
    num_layers: int = 4,
    num_heads: int = 8,
    donor_len: int = 32,
    head_dim: int = 64,
    device: str | None = None,
) -> torch.Tensor:
    """Create a random donor KV tensor."""
    dev = device or str(_device())
    return torch.randn(
        (num_layers, 2, num_heads, donor_len, head_dim),
        dtype=torch.float16,
        device=dev,
    )


# ---------------------------------------------------------------------------
# Kernel 1: scatter_donor_kv
# ---------------------------------------------------------------------------


class TestScatterDonorKv:
    """Tests for the scatter_donor_kv kernel."""

    def test_basic_scatter(self):
        """Scatter 4 positions from donor to target."""
        dev = _device()
        target = _make_kv_cache(num_layers=2, num_heads=4, seq_len=16, head_dim=32)
        donor = _make_donor_kv(num_layers=2, num_heads=4, donor_len=8, head_dim=32)

        donor_pos = torch.tensor([0, 2, 4, 6], dtype=torch.int32, device=dev)
        target_pos = torch.tensor([1, 3, 7, 12], dtype=torch.int32, device=dev)

        result = scatter_donor_kv(target, donor, donor_pos, target_pos)

        # Verify copied positions match
        for i in range(4):
            dp = donor_pos[i].item()
            tp = target_pos[i].item()
            torch.testing.assert_close(
                result[:, :, :, tp, :],
                donor[:, :, :, dp, :],
                atol=0,
                rtol=0,
            )

        # Verify non-copied positions are still zero
        non_target = [p for p in range(16) if p not in [1, 3, 7, 12]]
        for p in non_target[:3]:
            assert torch.all(result[:, :, :, p, :] == 0)

    def test_empty_pairs(self):
        """Empty position tensors should be a no-op."""
        target = _make_kv_cache(num_layers=1, num_heads=2, seq_len=8, head_dim=16)
        donor = _make_donor_kv(num_layers=1, num_heads=2, donor_len=4, head_dim=16)

        empty = torch.zeros(0, dtype=torch.int32, device=target.device)

        result = scatter_donor_kv(target, donor, empty, empty)
        assert torch.all(result == 0)

    def test_single_pair(self):
        """Single position scatter."""
        dev = _device()
        target = _make_kv_cache(num_layers=1, num_heads=1, seq_len=4, head_dim=8)
        donor = _make_donor_kv(num_layers=1, num_heads=1, donor_len=2, head_dim=8)

        dp = torch.tensor([1], dtype=torch.int32, device=dev)
        tp = torch.tensor([3], dtype=torch.int32, device=dev)

        scatter_donor_kv(target, donor, dp, tp)

        torch.testing.assert_close(
            target[0, :, 0, 3, :],
            donor[0, :, 0, 1, :],
            atol=0,
            rtol=0,
        )

    def test_full_sequence_scatter(self):
        """Scatter all positions (100% reuse)."""
        dev = _device()
        seq_len = 16
        target = _make_kv_cache(num_layers=2, num_heads=4, seq_len=seq_len, head_dim=32)
        donor = _make_donor_kv(num_layers=2, num_heads=4, donor_len=seq_len, head_dim=32)

        positions = torch.arange(seq_len, dtype=torch.int32, device=dev)
        scatter_donor_kv(target, donor, positions, positions)

        torch.testing.assert_close(target, donor, atol=0, rtol=0)

    def test_multi_layer_consistency(self):
        """Verify scatter is consistent across all layers."""
        dev = _device()
        target = _make_kv_cache(num_layers=8, num_heads=4, seq_len=32, head_dim=64)
        donor = _make_donor_kv(num_layers=8, num_heads=4, donor_len=16, head_dim=64)

        dp = torch.tensor([0, 5, 10], dtype=torch.int32, device=dev)
        tp = torch.tensor([2, 8, 20], dtype=torch.int32, device=dev)

        scatter_donor_kv(target, donor, dp, tp)

        # All layers should have identical scatter
        for layer in range(8):
            for i in range(3):
                torch.testing.assert_close(
                    target[layer, :, :, tp[i].item(), :],
                    donor[layer, :, :, dp[i].item(), :],
                    atol=0,
                    rtol=0,
                )


# ---------------------------------------------------------------------------
# Kernel 1b: scatter_donor_kv_paged
# ---------------------------------------------------------------------------


class TestScatterDonorKvPaged:
    """Tests for the paged KV cache scatter kernel."""

    def _make_paged_kv(
        self,
        num_blocks: int = 8,
        num_heads: int = 4,
        block_size: int = 16,
        head_dim: int = 32,
    ) -> torch.Tensor:
        """Create a zero-initialized paged KV cache."""
        dev = _device()
        return torch.zeros(
            (num_blocks, 2, num_heads, block_size, head_dim),
            dtype=torch.float16,
            device=dev,
        )

    def _make_paged_donor(
        self,
        num_heads: int = 4,
        donor_len: int = 16,
        head_dim: int = 32,
    ) -> torch.Tensor:
        """Create a random donor KV for paged scatter (no layer dim)."""
        dev = _device()
        return torch.randn(
            (2, num_heads, donor_len, head_dim),
            dtype=torch.float16,
            device=dev,
        )

    def test_basic_paged_scatter(self):
        """Scatter into paged KV cache with block table translation."""
        dev = _device()
        block_size = 4
        kv_cache = self._make_paged_kv(
            num_blocks=8, num_heads=2, block_size=block_size, head_dim=16
        )
        donor_kv = self._make_paged_donor(num_heads=2, donor_len=8, head_dim=16)

        # Block table: logical block 0 → physical 3, block 1 → physical 5
        block_table = torch.tensor([3, 5], dtype=torch.int32, device=dev)

        # Scatter donor pos 0 → target pos 1 (block 0 offset 1 → physical 3 offset 1)
        # Scatter donor pos 4 → target pos 5 (block 1 offset 1 → physical 5 offset 1)
        donor_pos = torch.tensor([0, 4], dtype=torch.int32, device=dev)
        target_pos = torch.tensor([1, 5], dtype=torch.int32, device=dev)

        scatter_donor_kv_paged(kv_cache, donor_kv, block_table, donor_pos, target_pos)

        # Verify: target pos 1 → physical block 3, offset 1
        torch.testing.assert_close(
            kv_cache[3, :, :, 1, :],
            donor_kv[:, :, 0, :],
            atol=0,
            rtol=0,
        )
        # Verify: target pos 5 → physical block 5, offset 1
        torch.testing.assert_close(
            kv_cache[5, :, :, 1, :],
            donor_kv[:, :, 4, :],
            atol=0,
            rtol=0,
        )

    def test_empty_pairs(self):
        """Empty position tensors should be a no-op."""
        dev = _device()
        kv_cache = self._make_paged_kv(num_blocks=4, num_heads=2, block_size=4, head_dim=8)
        donor_kv = self._make_paged_donor(num_heads=2, donor_len=4, head_dim=8)
        block_table = torch.tensor([0, 1], dtype=torch.int32, device=dev)
        empty = torch.zeros(0, dtype=torch.int32, device=dev)

        result = scatter_donor_kv_paged(kv_cache, donor_kv, block_table, empty, empty)
        assert torch.all(result == 0)

    def test_multiple_positions_same_block(self):
        """Multiple target positions mapping to the same physical block."""
        dev = _device()
        block_size = 8
        kv_cache = self._make_paged_kv(
            num_blocks=4, num_heads=2, block_size=block_size, head_dim=16
        )
        donor_kv = self._make_paged_donor(num_heads=2, donor_len=8, head_dim=16)

        # Block table: logical block 0 → physical 2
        block_table = torch.tensor([2], dtype=torch.int32, device=dev)

        # All target positions within block 0 (offsets 0, 3, 7)
        donor_pos = torch.tensor([0, 1, 2], dtype=torch.int32, device=dev)
        target_pos = torch.tensor([0, 3, 7], dtype=torch.int32, device=dev)

        scatter_donor_kv_paged(kv_cache, donor_kv, block_table, donor_pos, target_pos)

        for i, (dp, tp) in enumerate(zip([0, 1, 2], [0, 3, 7])):
            offset = tp % block_size
            torch.testing.assert_close(
                kv_cache[2, :, :, offset, :],
                donor_kv[:, :, dp, :],
                atol=0,
                rtol=0,
            )

    def test_out_of_bounds_block_skipped(self):
        """Target positions beyond block table range are skipped."""
        dev = _device()
        block_size = 4
        kv_cache = self._make_paged_kv(
            num_blocks=4, num_heads=1, block_size=block_size, head_dim=8
        )
        donor_kv = self._make_paged_donor(num_heads=1, donor_len=4, head_dim=8)

        # Only 1 block mapped
        block_table = torch.tensor([1], dtype=torch.int32, device=dev)

        # Second target pos (8) maps to logical block 2 which is OOB
        donor_pos = torch.tensor([0, 1], dtype=torch.int32, device=dev)
        target_pos = torch.tensor([2, 8], dtype=torch.int32, device=dev)

        scatter_donor_kv_paged(kv_cache, donor_kv, block_table, donor_pos, target_pos)

        # First scatter should work (pos 2 → block 0 offset 2 → physical 1 offset 2)
        torch.testing.assert_close(
            kv_cache[1, :, :, 2, :],
            donor_kv[:, :, 0, :],
            atol=0,
            rtol=0,
        )


# ---------------------------------------------------------------------------
# Kernel 2: masked_qkv_projection
# ---------------------------------------------------------------------------


class TestMaskedQkvProjection:
    """Tests for the masked QKV linear projection."""

    def test_all_compute(self):
        """All positions compute — should match standard linear."""
        dev = _device()
        seq_len = 8
        hidden_dim = 32
        out_features = 48  # 3 * num_heads * head_dim (e.g., 3*4*4)

        hidden = torch.randn(seq_len, hidden_dim, dtype=torch.float16, device=dev)
        weight = torch.randn(out_features, hidden_dim, dtype=torch.float16, device=dev)
        mask = torch.ones(seq_len, dtype=torch.bool, device=dev)

        result = masked_qkv_projection(hidden, weight, mask)

        expected = torch.nn.functional.linear(hidden, weight)

        torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)

    def test_all_reuse(self):
        """All positions reuse — output should be zero."""
        dev = _device()
        seq_len = 8
        hidden_dim = 32
        out_features = 48

        hidden = torch.randn(seq_len, hidden_dim, dtype=torch.float16, device=dev)
        weight = torch.randn(out_features, hidden_dim, dtype=torch.float16, device=dev)
        mask = torch.zeros(seq_len, dtype=torch.bool, device=dev)

        result = masked_qkv_projection(hidden, weight, mask)

        assert torch.all(result == 0)

    def test_mixed_mask(self):
        """Some compute, some reuse — only masked positions should have output."""
        dev = _device()
        seq_len = 8
        hidden_dim = 32
        out_features = 48

        hidden = torch.randn(seq_len, hidden_dim, dtype=torch.float16, device=dev)
        weight = torch.randn(out_features, hidden_dim, dtype=torch.float16, device=dev)
        mask = torch.tensor(
            [True, False, True, False, True, False, True, False],
            dtype=torch.bool,
            device=dev,
        )

        result = masked_qkv_projection(hidden, weight, mask)

        # Reuse positions should be zero
        for i in [1, 3, 5, 7]:
            assert torch.all(result[i] == 0), f"Position {i} should be zero"

        # Compute positions should be non-zero
        for i in [0, 2, 4, 6]:
            assert not torch.all(result[i] == 0), f"Position {i} should be non-zero"

    def test_with_bias(self):
        """Masked projection with bias term."""
        dev = _device()
        seq_len = 4
        hidden_dim = 16
        out_features = 24

        hidden = torch.randn(seq_len, hidden_dim, dtype=torch.float16, device=dev)
        weight = torch.randn(out_features, hidden_dim, dtype=torch.float16, device=dev)
        bias = torch.randn(out_features, dtype=torch.float16, device=dev)
        mask = torch.ones(seq_len, dtype=torch.bool, device=dev)

        result = masked_qkv_projection(hidden, weight, mask, qkv_bias=bias)
        expected = torch.nn.functional.linear(hidden, weight, bias)

        torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# Kernel 3: partial_prefill_attention
# ---------------------------------------------------------------------------


class TestPartialPrefillAttention:
    """Tests for the partial prefill attention kernel."""

    def test_all_compute_matches_standard(self):
        """All-compute mask should match standard causal attention."""
        dev = _device()
        num_heads = 2
        seq_len = 8
        head_dim = 16

        q = torch.randn(num_heads, seq_len, head_dim, dtype=torch.float32, device=dev)
        k = torch.randn(num_heads, seq_len, head_dim, dtype=torch.float32, device=dev)
        v = torch.randn(num_heads, seq_len, head_dim, dtype=torch.float32, device=dev)
        mask = torch.ones(seq_len, dtype=torch.bool, device=dev)

        scale = 1.0 / math.sqrt(head_dim)

        result = partial_prefill_attention(q, k, v, mask, scale)

        # Manual reference: standard causal attention
        for h in range(num_heads):
            for pos in range(seq_len):
                q_vec = q[h, pos : pos + 1, :]
                k_causal = k[h, : pos + 1, :]
                v_causal = v[h, : pos + 1, :]

                scores = torch.matmul(q_vec, k_causal.T) * scale
                weights = torch.softmax(scores, dim=-1)
                expected = torch.matmul(weights, v_causal)

                torch.testing.assert_close(
                    result[h, pos, :],
                    expected.squeeze(0),
                    atol=1e-4,
                    rtol=1e-3,
                )

    def test_reuse_positions_are_zero(self):
        """Reuse positions should have zero output."""
        dev = _device()
        num_heads = 2
        seq_len = 8
        head_dim = 16

        q = torch.randn(num_heads, seq_len, head_dim, dtype=torch.float32, device=dev)
        k = torch.randn(num_heads, seq_len, head_dim, dtype=torch.float32, device=dev)
        v = torch.randn(num_heads, seq_len, head_dim, dtype=torch.float32, device=dev)
        mask = torch.tensor(
            [True, False, True, False, True, False, True, False],
            dtype=torch.bool,
            device=dev,
        )

        result = partial_prefill_attention(q, k, v, mask)

        # Reuse positions should be zero
        for pos in [1, 3, 5, 7]:
            assert torch.all(result[:, pos, :] == 0)

    def test_compute_positions_attend_to_full_context(self):
        """Compute positions should attend to ALL positions (including reuse)."""
        dev = _device()
        num_heads = 1
        seq_len = 4
        head_dim = 8

        q = torch.randn(num_heads, seq_len, head_dim, dtype=torch.float32, device=dev)
        k = torch.randn(num_heads, seq_len, head_dim, dtype=torch.float32, device=dev)
        v = torch.randn(num_heads, seq_len, head_dim, dtype=torch.float32, device=dev)

        # Position 3 computes, positions 0,1,2 are reuse
        mask = torch.tensor([False, False, False, True], dtype=torch.bool, device=dev)

        result = partial_prefill_attention(q, k, v, mask)

        # Position 3 should attend to ALL 4 positions (causal)
        scale = 1.0 / math.sqrt(head_dim)
        q_vec = q[0, 3:4, :]
        k_all = k[0, :4, :]
        v_all = v[0, :4, :]
        scores = torch.matmul(q_vec, k_all.T) * scale
        weights = torch.softmax(scores, dim=-1)
        expected = torch.matmul(weights, v_all)

        torch.testing.assert_close(
            result[0, 3, :],
            expected.squeeze(0),
            atol=1e-4,
            rtol=1e-3,
        )

    def test_empty_compute_mask(self):
        """All-reuse mask should return all zeros."""
        dev = _device()
        q = torch.randn(2, 4, 8, dtype=torch.float32, device=dev)
        k = torch.randn(2, 4, 8, dtype=torch.float32, device=dev)
        v = torch.randn(2, 4, 8, dtype=torch.float32, device=dev)
        mask = torch.zeros(4, dtype=torch.bool, device=dev)

        result = partial_prefill_attention(q, k, v, mask)
        assert torch.all(result == 0)


# ---------------------------------------------------------------------------
# End-to-End: partial_prefill orchestrator
# ---------------------------------------------------------------------------


class TestPartialPrefill:
    """Tests for the full partial_prefill orchestrator."""

    def test_basic_partial_prefill(self):
        """Basic end-to-end partial prefill with 50% reuse."""
        dev = _device()
        seq_len = 8
        hidden_dim = 32
        num_heads = 4
        head_dim = hidden_dim // num_heads  # 8
        num_layers = 2
        donor_len = 8

        hidden = torch.randn(seq_len, hidden_dim, dtype=torch.float16, device=dev)
        donor_kv = _make_donor_kv(
            num_layers=num_layers,
            num_heads=num_heads,
            donor_len=donor_len,
            head_dim=head_dim,
        )

        qkv_weight = torch.randn(
            3 * num_heads * head_dim, hidden_dim,
            dtype=torch.float16, device=dev,
        )
        out_proj = torch.randn(
            hidden_dim, num_heads * head_dim,
            dtype=torch.float16, device=dev,
        )

        # 50% reuse: positions 0,1,2,3 from donor; 4,5,6,7 computed
        donor_pos = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=dev)
        target_pos = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=dev)
        compute_mask = torch.tensor(
            [False, False, False, False, True, True, True, True],
            dtype=torch.bool, device=dev,
        )

        result = partial_prefill(
            hidden_states=hidden,
            donor_kv=donor_kv,
            qkv_weight=qkv_weight,
            output_projection=out_proj,
            donor_positions=donor_pos,
            target_positions=target_pos,
            compute_mask=compute_mask,
            num_heads=num_heads,
            head_dim=head_dim,
        )

        assert isinstance(result, PartialPrefillResult)
        assert result.positions_computed == 4
        assert result.positions_reused == 4
        assert abs(result.computation_ratio - 0.5) < 1e-6
        assert result.output.shape == (seq_len, hidden_dim)
        assert result.kv_cache.shape == (num_layers, 2, num_heads, seq_len, head_dim)

    def test_full_reuse(self):
        """100% reuse — no computation needed."""
        dev = _device()
        seq_len = 4
        hidden_dim = 16
        num_heads = 2
        head_dim = 8
        num_layers = 1

        hidden = torch.randn(seq_len, hidden_dim, dtype=torch.float16, device=dev)
        donor_kv = _make_donor_kv(
            num_layers=num_layers,
            num_heads=num_heads,
            donor_len=seq_len,
            head_dim=head_dim,
        )
        qkv_weight = torch.randn(
            3 * num_heads * head_dim, hidden_dim,
            dtype=torch.float16, device=dev,
        )
        out_proj = torch.randn(
            hidden_dim, num_heads * head_dim,
            dtype=torch.float16, device=dev,
        )

        donor_pos = torch.arange(seq_len, dtype=torch.int32, device=dev)
        target_pos = torch.arange(seq_len, dtype=torch.int32, device=dev)
        compute_mask = torch.zeros(seq_len, dtype=torch.bool, device=dev)

        result = partial_prefill(
            hidden_states=hidden,
            donor_kv=donor_kv,
            qkv_weight=qkv_weight,
            output_projection=out_proj,
            donor_positions=donor_pos,
            target_positions=target_pos,
            compute_mask=compute_mask,
            num_heads=num_heads,
            head_dim=head_dim,
        )

        assert result.positions_computed == 0
        assert result.positions_reused == seq_len
        assert result.computation_ratio == 0.0

    def test_no_reuse(self):
        """0% reuse — full computation."""
        dev = _device()
        seq_len = 4
        hidden_dim = 16
        num_heads = 2
        head_dim = 8

        hidden = torch.randn(seq_len, hidden_dim, dtype=torch.float16, device=dev)
        donor_kv = _make_donor_kv(
            num_layers=1, num_heads=num_heads, donor_len=4, head_dim=head_dim
        )
        qkv_weight = torch.randn(
            3 * num_heads * head_dim, hidden_dim,
            dtype=torch.float16, device=dev,
        )
        out_proj = torch.randn(
            hidden_dim, num_heads * head_dim,
            dtype=torch.float16, device=dev,
        )

        empty = torch.zeros(0, dtype=torch.int32, device=dev)
        compute_mask = torch.ones(seq_len, dtype=torch.bool, device=dev)

        result = partial_prefill(
            hidden_states=hidden,
            donor_kv=donor_kv,
            qkv_weight=qkv_weight,
            output_projection=out_proj,
            donor_positions=empty,
            target_positions=empty,
            compute_mask=compute_mask,
            num_heads=num_heads,
            head_dim=head_dim,
        )

        assert result.positions_computed == seq_len
        assert result.positions_reused == 0
        assert result.computation_ratio == 1.0

    def test_sparse_reuse_pattern(self):
        """Non-contiguous reuse pattern (core SemBlend case)."""
        dev = _device()
        seq_len = 16
        hidden_dim = 32
        num_heads = 4
        head_dim = 8
        num_layers = 2

        hidden = torch.randn(seq_len, hidden_dim, dtype=torch.float16, device=dev)
        donor_kv = _make_donor_kv(
            num_layers=num_layers,
            num_heads=num_heads,
            donor_len=10,
            head_dim=head_dim,
        )
        qkv_weight = torch.randn(
            3 * num_heads * head_dim, hidden_dim,
            dtype=torch.float16, device=dev,
        )
        out_proj = torch.randn(
            hidden_dim, num_heads * head_dim,
            dtype=torch.float16, device=dev,
        )

        # Sparse reuse: positions 0,3,6,9,12 from donor 0,2,4,6,8
        donor_pos = torch.tensor([0, 2, 4, 6, 8], dtype=torch.int32, device=dev)
        target_pos = torch.tensor([0, 3, 6, 9, 12], dtype=torch.int32, device=dev)

        compute_mask = torch.ones(seq_len, dtype=torch.bool, device=dev)
        compute_mask[target_pos.long()] = False

        result = partial_prefill(
            hidden_states=hidden,
            donor_kv=donor_kv,
            qkv_weight=qkv_weight,
            output_projection=out_proj,
            donor_positions=donor_pos,
            target_positions=target_pos,
            compute_mask=compute_mask,
            num_heads=num_heads,
            head_dim=head_dim,
        )

        assert result.positions_reused == 5
        assert result.positions_computed == 11
        # Verify donor KV was copied to target positions
        for i in range(5):
            dp = donor_pos[i].item()
            tp = target_pos[i].item()
            torch.testing.assert_close(
                result.kv_cache[:, :, :, tp, :],
                donor_kv[:, :, :, dp, :],
                atol=0,
                rtol=0,
            )

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available",
    )
    def test_timing_reported(self):
        """GPU timing should be non-zero on CUDA."""
        dev = torch.device("cuda")
        seq_len = 64
        hidden_dim = 128
        num_heads = 8
        head_dim = 16

        hidden = torch.randn(seq_len, hidden_dim, dtype=torch.float16, device=dev)
        donor_kv = _make_donor_kv(
            num_layers=4, num_heads=num_heads, donor_len=32, head_dim=head_dim
        )
        qkv_weight = torch.randn(
            3 * num_heads * head_dim, hidden_dim,
            dtype=torch.float16, device=dev,
        )
        out_proj = torch.randn(
            hidden_dim, num_heads * head_dim,
            dtype=torch.float16, device=dev,
        )

        donor_pos = torch.arange(16, dtype=torch.int32, device=dev)
        target_pos = torch.arange(16, dtype=torch.int32, device=dev)
        compute_mask = torch.ones(seq_len, dtype=torch.bool, device=dev)
        compute_mask[:16] = False

        result = partial_prefill(
            hidden_states=hidden,
            donor_kv=donor_kv,
            qkv_weight=qkv_weight,
            output_projection=out_proj,
            donor_positions=donor_pos,
            target_positions=target_pos,
            compute_mask=compute_mask,
            num_heads=num_heads,
            head_dim=head_dim,
        )

        assert result.total_time_ms > 0
        assert result.scatter_time_ms >= 0


# ---------------------------------------------------------------------------
# Model Runner Patch
# ---------------------------------------------------------------------------


class TestModelRunnerPatch:
    """Tests for the model runner monkey-patch."""

    def test_patch_applies_to_mock_runner(self):
        """patch_model_runner should wrap execute_model on a compatible object."""
        from synapse_kv_connector.model_runner_hook import patch_model_runner

        class MockRunner:
            def __init__(self):
                self.model = object()
                self.call_count = 0

            def execute_model(self, *args, **kwargs):
                self.call_count += 1
                return "result"

        class MockConnector:
            _active_hook = None

        runner = MockRunner()
        connector = MockConnector()
        result = patch_model_runner(runner, connector)

        assert result is True
        # Patched execute_model should still work
        out = runner.execute_model()
        assert out == "result"
        assert runner.call_count == 1

    def test_patch_skips_incompatible_runner(self):
        """patch_model_runner should return False for objects without execute_model."""
        from synapse_kv_connector.model_runner_hook import patch_model_runner

        result = patch_model_runner(object(), object())
        assert result is False


# ---------------------------------------------------------------------------
# Pipeline PartialAttention Plan Building
# ---------------------------------------------------------------------------


class TestPipelinePlanBuilding:
    """Tests for SemBlendPipeline.build_partial_attention_plan."""

    def test_build_plan_from_pipeline_result(self):
        """Build PartialAttention plan from a successful pipeline result."""
        from synapse_kv_connector.pipeline import PipelineResult, SemBlendPipeline

        pipeline = SemBlendPipeline(
            embedder_type="jaccard",
            model_name="Qwen/Qwen2.5-7B-Instruct-AWQ",
        )

        result = PipelineResult(
            found=True,
            donor_id="donor-1",
            similarity=0.9,
            reuse_ratio=0.6,
            donor_tokens=list(range(100)),
            slot_actions=[
                {"action": "copy_from_donor", "targetPos": i, "donorPos": i}
                for i in range(60)
            ] + [
                {"action": "recompute", "targetPos": i, "donorPos": None}
                for i in range(60, 100)
            ],
            layer_deviations=[
                {
                    "layerIdx": i,
                    "deviationScore": 0.5 if i < 3 or i > 24 else 0.1,
                    "shouldRecompute": i < 3 or i > 24,
                }
                for i in range(28)
            ],
        )

        plan = pipeline.build_partial_attention_plan(result)

        assert plan is not None
        assert plan.donor_id == "donor-1"
        assert plan.num_reuse_positions == 60
        assert plan.num_partial_positions == 40
        assert plan.num_full_layers > 0  # Early/late layers flagged
        assert plan.computation_ratio < 1.0

    def test_build_plan_from_not_found(self):
        """Should return None for a not-found result."""
        from synapse_kv_connector.pipeline import PipelineResult, SemBlendPipeline

        pipeline = SemBlendPipeline(embedder_type="jaccard")
        result = PipelineResult(found=False)
        plan = pipeline.build_partial_attention_plan(result)
        assert plan is None

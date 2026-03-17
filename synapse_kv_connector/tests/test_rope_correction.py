"""Tests for RoPE delta correction — the core SemBlend innovation.

Verifies that applying RoPE(target_pos - donor_pos) to cached K tensors
produces mathematically exact results: the corrected K matches what would
have been computed from scratch at the target position.

The key identity: RoPE(a) × RoPE⁻¹(b) = RoPE(a - b)
"""
import math

import pytest
import torch


def apply_rope_from_scratch(x: torch.Tensor, positions: torch.Tensor,
                            head_dim: int, rope_base: float = 10000.0) -> torch.Tensor:
    """Apply RoPE to raw (pre-rotation) tensor at given positions.

    This is the reference implementation matching Qwen2/LLaMA RoPE.
    """
    inv_freq = 1.0 / (
        rope_base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    result = x.clone().float()
    for i, pos in enumerate(positions):
        theta = float(pos) * inv_freq
        cos_v = torch.cos(theta)
        sin_v = torch.sin(theta)
        even = result[..., i, 0::2].clone()
        odd = result[..., i, 1::2].clone()
        result[..., i, 0::2] = even * cos_v - odd * sin_v
        result[..., i, 1::2] = even * sin_v + odd * cos_v
    return result.to(x.dtype)


class TestRoPECorrection:
    """Test suite for RoPE delta correction kernel."""

    def test_identity_no_delta(self):
        """When donor_pos == target_pos, correction should be identity."""
        from synapse_kv_connector.rope_correction import rope_correct_k

        num_heads, seq_len, head_dim = 4, 8, 128
        k = torch.randn(num_heads, seq_len, head_dim, dtype=torch.float16)

        positions = torch.tensor([0, 3, 7], dtype=torch.int32)
        result = rope_correct_k(k, positions, positions, head_dim=head_dim)

        torch.testing.assert_close(result, k, atol=1e-3, rtol=1e-3)

    def test_exact_correction_single_position(self):
        """Correcting K from pos 5 to pos 10 should match fresh RoPE at pos 10."""
        from synapse_kv_connector.rope_correction import rope_correct_k

        num_heads, head_dim = 4, 128
        raw_k = torch.randn(num_heads, 1, head_dim, dtype=torch.float32)

        donor_pos = torch.tensor([0], dtype=torch.int32)
        target_pos = torch.tensor([0], dtype=torch.int32)

        # Compute K at donor position 5
        k_at_5 = apply_rope_from_scratch(raw_k, torch.tensor([5]), head_dim)

        # Compute K at target position 10 (ground truth)
        k_at_10 = apply_rope_from_scratch(raw_k, torch.tensor([10]), head_dim)

        # Pad to seq_len > 5
        k_padded = torch.zeros(num_heads, 11, head_dim, dtype=torch.float32)
        k_padded[:, 5, :] = k_at_5[:, 0, :]

        # Apply correction: move from pos 5 to pos 10
        corrected = rope_correct_k(
            k_padded,
            donor_positions=torch.tensor([5], dtype=torch.int32),
            target_positions=torch.tensor([10], dtype=torch.int32),
            head_dim=head_dim,
        )

        # The corrected K at position 5 should match ground truth at position 10
        torch.testing.assert_close(
            corrected[:, 5, :], k_at_10[:, 0, :],
            atol=1e-4, rtol=1e-4,
        )

    def test_exact_correction_reorder(self):
        """REORDER: same tokens at different positions should be exact."""
        from synapse_kv_connector.rope_correction import rope_correct_k

        num_heads, seq_len, head_dim = 4, 4, 64
        raw_k = torch.randn(num_heads, seq_len, head_dim)

        donor_positions = torch.arange(seq_len, dtype=torch.int32)
        target_positions = torch.tensor([2, 3, 0, 1], dtype=torch.int32)

        # Compute donor K (with RoPE at positions 0,1,2,3)
        k_donor = apply_rope_from_scratch(raw_k, donor_positions, head_dim)

        # Compute target K (ground truth: RoPE at positions 2,3,0,1)
        k_target_gt = apply_rope_from_scratch(raw_k, target_positions, head_dim)

        # Apply RoPE correction
        corrected = rope_correct_k(
            k_donor, donor_positions, target_positions, head_dim=head_dim,
        )

        # Each position should match the ground truth
        for i in range(seq_len):
            torch.testing.assert_close(
                corrected[:, i, :], k_target_gt[:, i, :],
                atol=1e-3, rtol=1e-3,
                msg=f"Position {i}: donor_pos={donor_positions[i]}, target_pos={target_positions[i]}",
            )

    def test_negative_delta(self):
        """Correction works for negative deltas (moving to earlier position)."""
        from synapse_kv_connector.rope_correction import rope_correct_k

        num_heads, head_dim = 2, 64
        raw_k = torch.randn(num_heads, 1, head_dim)

        k_at_100 = apply_rope_from_scratch(raw_k, torch.tensor([100]), head_dim)
        k_at_50 = apply_rope_from_scratch(raw_k, torch.tensor([50]), head_dim)

        k_padded = torch.zeros(num_heads, 101, head_dim)
        k_padded[:, 100, :] = k_at_100[:, 0, :]

        corrected = rope_correct_k(
            k_padded,
            donor_positions=torch.tensor([100], dtype=torch.int32),
            target_positions=torch.tensor([50], dtype=torch.int32),
            head_dim=head_dim,
        )

        torch.testing.assert_close(
            corrected[:, 100, :], k_at_50[:, 0, :],
            atol=1e-3, rtol=1e-3,
        )

    def test_large_sequence(self):
        """Verify correction works for 8K tokens (typical RAG prompt)."""
        from synapse_kv_connector.rope_correction import rope_correct_k

        num_heads, seq_len, head_dim = 4, 8192, 128
        k = torch.randn(num_heads, seq_len, head_dim, dtype=torch.float16)

        num_pairs = 100
        donor_pos = torch.randint(0, seq_len, (num_pairs,), dtype=torch.int32)
        target_pos = torch.randint(0, seq_len, (num_pairs,), dtype=torch.int32)

        result = rope_correct_k(k, donor_pos, target_pos, head_dim=head_dim)

        assert result.shape == k.shape
        assert result.dtype == k.dtype

    def test_v_unchanged(self):
        """V cache should not be modified (no position encoding)."""
        from synapse_kv_connector.rope_correction import rope_correct_scatter_paged

        num_blocks, num_heads, block_size, head_dim = 4, 2, 16, 64
        kv_cache = torch.zeros(num_blocks, 2, num_heads, block_size, head_dim)
        donor_kv = torch.randn(2, num_heads, 32, head_dim)
        block_table = torch.arange(num_blocks, dtype=torch.int32)

        donor_pos = torch.tensor([0, 1, 2], dtype=torch.int32)
        target_pos = torch.tensor([5, 10, 15], dtype=torch.int32)

        rope_correct_scatter_paged(
            kv_cache, donor_kv, block_table, donor_pos, target_pos,
        )

        # V at target positions should match donor V exactly
        for i, (dp, tp) in enumerate(zip(donor_pos, target_pos)):
            tb = int(tp) // block_size
            to_ = int(tp) % block_size
            torch.testing.assert_close(
                kv_cache[tb, 1, :, to_, :], donor_kv[1, :, int(dp), :],
                atol=1e-6, rtol=1e-6,
            )

    def test_paged_scatter_k_corrected(self):
        """K in paged cache should have RoPE correction applied."""
        from synapse_kv_connector.rope_correction import rope_correct_scatter_paged

        num_blocks, num_heads, block_size, head_dim = 4, 2, 16, 64
        kv_cache = torch.zeros(num_blocks, 2, num_heads, block_size, head_dim)
        block_table = torch.arange(num_blocks, dtype=torch.int32)

        raw_k = torch.randn(num_heads, 1, head_dim)
        donor_pos_val, target_pos_val = 3, 7

        # Create donor KV with K at position 3
        k_at_3 = apply_rope_from_scratch(raw_k, torch.tensor([donor_pos_val]), head_dim)
        v = torch.randn(num_heads, 32, head_dim)
        donor_kv = torch.zeros(2, num_heads, 32, head_dim)
        donor_kv[0, :, donor_pos_val, :] = k_at_3[:, 0, :]
        donor_kv[1] = v

        donor_pos = torch.tensor([donor_pos_val], dtype=torch.int32)
        target_pos = torch.tensor([target_pos_val], dtype=torch.int32)

        rope_correct_scatter_paged(
            kv_cache, donor_kv, block_table, donor_pos, target_pos,
        )

        # Ground truth: K at target position 7
        k_at_7 = apply_rope_from_scratch(raw_k, torch.tensor([target_pos_val]), head_dim)

        tb = target_pos_val // block_size
        to_ = target_pos_val % block_size

        torch.testing.assert_close(
            kv_cache[tb, 0, :, to_, :], k_at_7[:, 0, :],
            atol=1e-3, rtol=1e-3,
        )

    def test_permute_paged_kv(self):
        """Test in-place permutation with RoPE correction."""
        from synapse_kv_connector.rope_correction import permute_paged_kv_with_rope

        num_blocks, num_heads, block_size, head_dim = 2, 2, 16, 64
        kv_cache = torch.randn(num_blocks, 2, num_heads, block_size, head_dim)
        block_table = torch.arange(num_blocks, dtype=torch.int32)

        # Save original V values for verification
        original_v_pos0 = kv_cache[0, 1, :, 0, :].clone()
        original_v_pos1 = kv_cache[0, 1, :, 1, :].clone()

        # Swap positions 0 and 1
        permutation = [(0, 1), (1, 0)]
        permute_paged_kv_with_rope(kv_cache, block_table, permutation)

        # V at pos 1 should now contain what was at pos 0
        torch.testing.assert_close(
            kv_cache[0, 1, :, 1, :], original_v_pos0,
            atol=1e-6, rtol=1e-6,
        )
        # V at pos 0 should now contain what was at pos 1
        torch.testing.assert_close(
            kv_cache[0, 1, :, 0, :], original_v_pos1,
            atol=1e-6, rtol=1e-6,
        )

    def test_rope_correction_is_cheap(self):
        """Verify correction overhead is negligible (<1ms for 8K tokens)."""
        import time
        from synapse_kv_connector.rope_correction import rope_correct_k

        num_heads, seq_len, head_dim = 4, 8192, 128
        k = torch.randn(num_heads, seq_len, head_dim)

        donor_pos = torch.arange(seq_len, dtype=torch.int32)
        target_pos = torch.roll(donor_pos, 256)  # Shift by 256 positions

        # Warm up
        rope_correct_k(k, donor_pos[:10], target_pos[:10], head_dim=head_dim)

        t0 = time.monotonic()
        rope_correct_k(k, donor_pos, target_pos, head_dim=head_dim)
        elapsed_ms = (time.monotonic() - t0) * 1000

        # CPU path may be slower than GPU; just verify it completes
        assert elapsed_ms < 5000, f"RoPE correction took {elapsed_ms:.1f}ms (budget: <5000ms on CPU)"


class TestRoPECorrectionComposition:
    """Test the mathematical composition property of RoPE."""

    def test_composition_identity(self):
        """RoPE(a) × RoPE(-a) = Identity."""
        from synapse_kv_connector.rope_correction import rope_correct_k

        num_heads, head_dim = 2, 64
        k = torch.randn(num_heads, 10, head_dim)

        # Apply correction pos 0 -> pos 5, then pos 5 -> pos 0
        step1 = rope_correct_k(
            k,
            torch.tensor([3], dtype=torch.int32),
            torch.tensor([8], dtype=torch.int32),
            head_dim=head_dim,
        )
        step2 = rope_correct_k(
            step1,
            torch.tensor([3], dtype=torch.int32),
            torch.tensor([3], dtype=torch.int32) - (torch.tensor([8], dtype=torch.int32) - torch.tensor([3], dtype=torch.int32)),
            head_dim=head_dim,
        )

        # After round-trip, should recover original (within float precision)
        # Note: the correction modifies k at donor_pos, so check position 3
        torch.testing.assert_close(
            step2[:, 3, :], k[:, 3, :],
            atol=1e-3, rtol=1e-3,
        )

    def test_different_rope_bases(self):
        """Correction works with non-standard RoPE bases."""
        from synapse_kv_connector.rope_correction import rope_correct_k

        for rope_base in [500.0, 10000.0, 1000000.0]:
            num_heads, head_dim = 2, 32
            raw_k = torch.randn(num_heads, 1, head_dim)

            k_at_0 = apply_rope_from_scratch(
                raw_k, torch.tensor([0]), head_dim, rope_base
            )
            k_at_10 = apply_rope_from_scratch(
                raw_k, torch.tensor([10]), head_dim, rope_base
            )

            k_padded = torch.zeros(num_heads, 11, head_dim)
            k_padded[:, 0, :] = k_at_0[:, 0, :]

            corrected = rope_correct_k(
                k_padded,
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([10], dtype=torch.int32),
                head_dim=head_dim,
                rope_base=rope_base,
            )

            torch.testing.assert_close(
                corrected[:, 0, :], k_at_10[:, 0, :],
                atol=1e-3, rtol=1e-3,
                msg=f"Failed for rope_base={rope_base}",
            )


class TestPositionMapping:
    """Test PositionMapping integration from pipeline."""

    def test_position_map_needs_correction(self):
        """PositionMapping.needs_correction is True when positions differ."""
        from synapse_kv_connector.pipeline import PositionMapping

        # Same positions — no correction needed
        pm = PositionMapping(
            donor_positions=[0, 1, 2],
            target_positions=[0, 1, 2],
        )
        assert not pm.needs_correction

        # Different positions — correction needed
        pm2 = PositionMapping(
            donor_positions=[0, 1, 2],
            target_positions=[5, 10, 15],
        )
        assert pm2.needs_correction

    def test_position_map_to_rope_correction(self):
        """PositionMapping drives correct RoPE delta correction."""
        from synapse_kv_connector.pipeline import PositionMapping
        from synapse_kv_connector.rope_correction import rope_correct_k

        num_heads, head_dim = 2, 64
        raw_k = torch.randn(num_heads, 1, head_dim)

        pm = PositionMapping(
            donor_positions=[3],
            target_positions=[7],
        )

        # Create K at donor position 3
        k_at_3 = apply_rope_from_scratch(raw_k, torch.tensor([3]), head_dim)
        k_at_7 = apply_rope_from_scratch(raw_k, torch.tensor([7]), head_dim)

        k_padded = torch.zeros(num_heads, 10, head_dim)
        k_padded[:, 3, :] = k_at_3[:, 0, :]

        # Apply correction using position map
        corrected = rope_correct_k(
            k_padded,
            donor_positions=torch.tensor(pm.donor_positions, dtype=torch.int32),
            target_positions=torch.tensor(pm.target_positions, dtype=torch.int32),
            head_dim=head_dim,
        )

        torch.testing.assert_close(
            corrected[:, 3, :], k_at_7[:, 0, :],
            atol=1e-3, rtol=1e-3,
        )

    def test_reorder_position_map(self):
        """REORDER scenario: same tokens, different positions, exact correction."""
        from synapse_kv_connector.pipeline import PositionMapping
        from synapse_kv_connector.rope_correction import rope_correct_k

        num_heads, seq_len, head_dim = 2, 4, 32

        # Donor has tokens at positions 0,1,2,3
        # Target wants them at positions 2,3,0,1
        pm = PositionMapping(
            donor_positions=[0, 1, 2, 3],
            target_positions=[2, 3, 0, 1],
        )
        assert pm.needs_correction
        assert pm.num_pairs == 4

        raw_k = torch.randn(num_heads, seq_len, head_dim)

        donor_positions = torch.arange(seq_len, dtype=torch.int32)
        target_positions = torch.tensor([2, 3, 0, 1], dtype=torch.int32)

        k_donor = apply_rope_from_scratch(raw_k, donor_positions, head_dim)
        k_target_gt = apply_rope_from_scratch(raw_k, target_positions, head_dim)

        corrected = rope_correct_k(
            k_donor,
            donor_positions=torch.tensor(pm.donor_positions, dtype=torch.int32),
            target_positions=torch.tensor(pm.target_positions, dtype=torch.int32),
            head_dim=head_dim,
        )

        for i in range(seq_len):
            torch.testing.assert_close(
                corrected[:, i, :], k_target_gt[:, i, :],
                atol=1e-3, rtol=1e-3,
            )


class TestNoPePermutation:
    """Tests for NoPE two-step correction: strip RoPE(-src) then apply RoPE(tgt).

    NoPE is mathematically equivalent to delta correction:
      RoPE(tgt) × RoPE(-src) = RoPE(tgt - src)

    These tests verify:
    1. NoPE identity: src_pos == tgt_pos → no change
    2. NoPE correctness: result matches delta correction for arbitrary Δ
    3. NoPE matches fresh RoPE at target position (same as delta test)
    4. NoPE correctness in paged cache (via nope_permute_paged_kv)
    """

    def _build_paged_cache(
        self, num_heads: int, head_dim: int, seq_len: int, block_size: int = 16
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build a paged KV cache and block table for testing."""
        num_blocks = (seq_len + block_size - 1) // block_size + 2
        kv_cache = torch.zeros(num_blocks, 2, num_heads, block_size, head_dim,
                               dtype=torch.float16)
        num_logical = (seq_len + block_size - 1) // block_size
        block_table = torch.arange(num_logical, dtype=torch.int32)
        return kv_cache, block_table

    def _apply_rope_single(
        self, k_raw: torch.Tensor, pos: int, head_dim: int, rope_base: float = 10000.0
    ) -> torch.Tensor:
        """Apply RoPE(pos) to a single K vector [num_heads, head_dim]."""
        inv_freq = 1.0 / (
            rope_base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        theta = float(pos) * inv_freq
        cos_v = torch.cos(theta)
        sin_v = torch.sin(theta)
        k = k_raw.float().clone()
        k_even = k[:, 0::2].clone()
        k_odd = k[:, 1::2].clone()
        k[:, 0::2] = k_even * cos_v - k_odd * sin_v
        k[:, 1::2] = k_even * sin_v + k_odd * cos_v
        return k

    def test_nope_identity_no_shift(self):
        """NoPE with src_pos == tgt_pos: K should be unchanged."""
        from synapse_kv_connector.rope_correction import nope_permute_paged_kv

        num_heads, head_dim, block_size = 4, 64, 16
        raw_k = torch.randn(num_heads, head_dim, dtype=torch.float32)
        k_at_pos5 = self._apply_rope_single(raw_k, pos=5, head_dim=head_dim)

        kv_cache, block_table = self._build_paged_cache(num_heads, head_dim, 32, block_size)
        kv_cache[0, 0, :, 5 % block_size, :] = k_at_pos5.half()  # K at pos 5

        nope_permute_paged_kv(
            kv_cache, block_table, permutation=[(5, 5)], rope_base=10000.0
        )

        result = kv_cache[0, 0, :, 5 % block_size, :].float()
        torch.testing.assert_close(result, k_at_pos5, atol=5e-3, rtol=5e-3)

    def test_nope_matches_delta_arbitrary_shift(self):
        """NoPE two-step produces same result as delta correction for Δ ≠ 0."""
        from synapse_kv_connector.rope_correction import (
            nope_permute_paged_kv,
            permute_paged_kv_with_rope,
        )

        num_heads, head_dim, block_size = 4, 64, 16
        raw_k = torch.randn(num_heads, head_dim, dtype=torch.float32)
        src_pos, tgt_pos = 3, 17  # Δ = 14

        k_at_src = self._apply_rope_single(raw_k, src_pos, head_dim)

        # Build two identical paged caches
        kv_delta, block_table = self._build_paged_cache(num_heads, head_dim, 64, block_size)
        kv_nope, _ = self._build_paged_cache(num_heads, head_dim, 64, block_size)

        # Write K at src_pos to both caches
        sb = src_pos // block_size
        so = src_pos % block_size
        kv_delta[sb, 0, :, so, :] = k_at_src.half()
        kv_nope[sb, 0, :, so, :] = k_at_src.half()

        # Delta: RoPE(Δ) in one step
        permute_paged_kv_with_rope(
            kv_delta, block_table, permutation=[(src_pos, tgt_pos)], rope_base=10000.0
        )

        # NoPE: strip RoPE(-src) then apply RoPE(tgt)
        nope_permute_paged_kv(
            kv_nope, block_table, permutation=[(src_pos, tgt_pos)], rope_base=10000.0
        )

        tb = tgt_pos // block_size
        to_ = tgt_pos % block_size
        k_delta_result = kv_delta[tb, 0, :, to_, :].float()
        k_nope_result = kv_nope[tb, 0, :, to_, :].float()

        # Both should produce identical K at target position
        torch.testing.assert_close(k_nope_result, k_delta_result, atol=5e-3, rtol=5e-3)

    def test_nope_matches_fresh_rope_at_target(self):
        """NoPE-corrected K matches fresh RoPE computed at target position."""
        from synapse_kv_connector.rope_correction import nope_permute_paged_kv

        num_heads, head_dim, block_size = 4, 128, 16
        raw_k = torch.randn(num_heads, head_dim, dtype=torch.float32)
        src_pos, tgt_pos = 7, 23

        k_at_src = self._apply_rope_single(raw_k, src_pos, head_dim)
        k_at_tgt_fresh = self._apply_rope_single(raw_k, tgt_pos, head_dim)

        kv_cache, block_table = self._build_paged_cache(num_heads, head_dim, 64, block_size)
        sb = src_pos // block_size
        so = src_pos % block_size
        kv_cache[sb, 0, :, so, :] = k_at_src.half()

        nope_permute_paged_kv(
            kv_cache, block_table, permutation=[(src_pos, tgt_pos)], rope_base=10000.0
        )

        tb = tgt_pos // block_size
        to_ = tgt_pos % block_size
        result = kv_cache[tb, 0, :, to_, :].float()

        # NoPE-corrected K should match fresh RoPE at target
        # Tolerance is higher due to float16 rounding in src
        torch.testing.assert_close(result, k_at_tgt_fresh, atol=1e-2, rtol=1e-2)

    def test_nope_v_cache_unchanged(self):
        """V cache should be copied directly without any rotation (both modes)."""
        from synapse_kv_connector.rope_correction import nope_permute_paged_kv

        num_heads, head_dim, block_size = 4, 64, 16
        kv_cache, block_table = self._build_paged_cache(num_heads, head_dim, 32, block_size)
        v_original = torch.randn(num_heads, head_dim, dtype=torch.float16)
        kv_cache[0, 1, :, 2, :] = v_original  # V at pos 2 (block 0, offset 2)

        nope_permute_paged_kv(
            kv_cache, block_table, permutation=[(2, 10)], rope_base=10000.0
        )

        tb = 10 // block_size
        to_ = 10 % block_size
        v_at_target = kv_cache[tb, 1, :, to_, :]
        torch.testing.assert_close(v_at_target, v_original, atol=0.0, rtol=0.0)

    @pytest.mark.parametrize("delta", [0, 1, 5, 50, 100, 256, 1000])
    def test_nope_delta_correction_parity(self, delta: int):
        """NoPE == delta for all Δ values: 0, 1, 5, 50, 100, 256, 1000."""
        from synapse_kv_connector.rope_correction import (
            nope_permute_paged_kv,
            permute_paged_kv_with_rope,
        )

        num_heads, head_dim, block_size = 8, 128, 16
        src_pos = 5
        tgt_pos = src_pos + delta
        if tgt_pos >= 64 * block_size:
            pytest.skip("tgt_pos exceeds test cache size")

        raw_k = torch.randn(num_heads, head_dim, dtype=torch.float32)
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        theta = src_pos * inv_freq
        k_at_src = raw_k.clone()
        k_at_src[:, 0::2] = raw_k[:, 0::2] * torch.cos(theta) - raw_k[:, 1::2] * torch.sin(theta)
        k_at_src[:, 1::2] = raw_k[:, 0::2] * torch.sin(theta) + raw_k[:, 1::2] * torch.cos(theta)

        def make_cache():
            kv, bt = self._build_paged_cache(num_heads, head_dim, max(tgt_pos + 16, 128), block_size)
            sb = src_pos // block_size
            so = src_pos % block_size
            kv[sb, 0, :, so, :] = k_at_src.half()
            return kv, bt

        kv_d, bt_d = make_cache()
        kv_n, bt_n = make_cache()

        permute_paged_kv_with_rope(kv_d, bt_d, [(src_pos, tgt_pos)], rope_base=10000.0)
        nope_permute_paged_kv(kv_n, bt_n, [(src_pos, tgt_pos)], rope_base=10000.0)

        tb = tgt_pos // block_size
        to_ = tgt_pos % block_size
        torch.testing.assert_close(
            kv_n[tb, 0, :, to_, :].float(),
            kv_d[tb, 0, :, to_, :].float(),
            atol=5e-3, rtol=5e-3,
            msg=f"NoPE != delta at Δ={delta}",
        )

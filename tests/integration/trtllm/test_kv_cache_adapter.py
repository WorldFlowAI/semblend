"""Tests for TRT-LLM KV cache layout adapter.

Verifies stride computation for TRT-LLM's PyTorch backend KV cache layout
and cross-layout compatibility with SemBlend's Triton kernels.

These tests run on CPU (no TRT-LLM or GPU required).
"""
from __future__ import annotations

import pytest

# Mock torch for CPU-only testing
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestTRTLLMKVStrides:
    """Test stride computation for TRT-LLM KV cache layout."""

    def test_trtllm_strides_basic(self):
        """Verify stride values for a standard TRT-LLM layout."""
        from semblend.integration.trtllm.kv_cache_adapter import trtllm_kv_strides

        # TRT-LLM: [num_blocks, 2, tokens_per_block, num_kv_heads, head_dim]
        kv = torch.zeros(100, 2, 128, 4, 128, dtype=torch.float16)
        strides = trtllm_kv_strides(kv)

        assert strides.kv_stride_block == kv.stride(0)
        assert strides.kv_stride_kv == kv.stride(1)
        assert strides.kv_stride_pos == kv.stride(2)
        assert strides.kv_stride_head == kv.stride(3)
        assert strides.kv_stride_dim == kv.stride(4)

    def test_trtllm_strides_contiguous(self):
        """Verify strides are correct for contiguous tensors."""
        from semblend.integration.trtllm.kv_cache_adapter import trtllm_kv_strides

        num_blocks, kv_dim, tpb, n_heads, head_dim = 50, 2, 64, 8, 128
        kv = torch.zeros(num_blocks, kv_dim, tpb, n_heads, head_dim)
        strides = trtllm_kv_strides(kv)

        # For contiguous tensor, stride = product of trailing dims
        assert strides.kv_stride_dim == 1
        assert strides.kv_stride_head == head_dim
        assert strides.kv_stride_pos == n_heads * head_dim
        assert strides.kv_stride_kv == tpb * n_heads * head_dim
        assert strides.kv_stride_block == kv_dim * tpb * n_heads * head_dim

    def test_vllm_strides_differ(self):
        """Verify vLLM strides have head and pos swapped vs TRT-LLM."""
        from semblend.integration.trtllm.kv_cache_adapter import (
            trtllm_kv_strides,
            vllm_kv_strides,
        )

        # Same total size, different layout
        trtllm_kv = torch.zeros(100, 2, 128, 4, 128)  # TRT-LLM
        vllm_kv = torch.zeros(100, 2, 4, 128, 128)    # vLLM

        t_strides = trtllm_kv_strides(trtllm_kv)
        v_strides = vllm_kv_strides(vllm_kv)

        # Block and KV strides should match (same total size per block)
        assert t_strides.kv_stride_block == v_strides.kv_stride_block
        assert t_strides.kv_stride_kv == v_strides.kv_stride_kv

        # Head and pos strides should differ (swapped dims)
        # TRT-LLM: pos=128*4*128=65536? No...
        # TRT-LLM: head_dim=128, n_heads=4, tpb=128
        # stride_pos = n_heads * head_dim = 4*128 = 512
        # stride_head = head_dim = 128
        assert t_strides.kv_stride_pos == 4 * 128  # n_heads * head_dim
        assert t_strides.kv_stride_head == 128      # head_dim

        # vLLM: head_dim=128, block_size=128, n_heads=4
        # stride_head = block_size * head_dim = 128*128 = 16384
        # stride_pos = head_dim = 128
        assert v_strides.kv_stride_head == 128 * 128  # block_size * head_dim
        assert v_strides.kv_stride_pos == 128          # head_dim


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestDetectTRTLLMLayout:
    """Test TRT-LLM layout detection."""

    def test_valid_trtllm_layout(self):
        """Detect standard TRT-LLM 5D layout."""
        from semblend.integration.trtllm.kv_cache_adapter import detect_trtllm_layout

        kv = torch.zeros(100, 2, 128, 4, 128)
        layout = detect_trtllm_layout(kv)

        assert layout.num_blocks == 100
        assert layout.tokens_per_block == 128
        assert layout.num_kv_heads == 4
        assert layout.head_dim == 128
        assert layout.layout_name == "trtllm_pytorch"

    def test_invalid_ndim(self):
        """Reject non-5D tensors."""
        from semblend.integration.trtllm.kv_cache_adapter import detect_trtllm_layout

        kv = torch.zeros(100, 2, 128, 128)
        with pytest.raises(ValueError, match="Expected 5D"):
            detect_trtllm_layout(kv)

    def test_invalid_kv_dim(self):
        """Reject tensors where dim[1] != 2."""
        from semblend.integration.trtllm.kv_cache_adapter import detect_trtllm_layout

        kv = torch.zeros(100, 3, 128, 4, 128)
        with pytest.raises(ValueError, match="Expected dim.*=2"):
            detect_trtllm_layout(kv)

    def test_qwen2_5_7b_awq(self):
        """Detect layout for Qwen2.5-7B-Instruct-AWQ (GQA: 4 KV heads)."""
        from semblend.integration.trtllm.kv_cache_adapter import detect_trtllm_layout

        # Qwen2.5-7B: 4 KV heads (GQA), head_dim=128, tpb=128
        kv = torch.zeros(200, 2, 128, 4, 128)
        layout = detect_trtllm_layout(kv)

        assert layout.num_kv_heads == 4
        assert layout.head_dim == 128
        assert layout.tokens_per_block == 128

    def test_llama3_8b(self):
        """Detect layout for LLaMA-3-8B (GQA: 8 KV heads)."""
        from semblend.integration.trtllm.kv_cache_adapter import detect_trtllm_layout

        # LLaMA-3-8B: 8 KV heads (GQA), head_dim=128, tpb=128
        kv = torch.zeros(200, 2, 128, 8, 128)
        layout = detect_trtllm_layout(kv)

        assert layout.num_kv_heads == 8
        assert layout.head_dim == 128


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestIsTRTLLMLayout:
    """Test heuristic TRT-LLM layout detection."""

    def test_trtllm_detected(self):
        """TRT-LLM layout is detected correctly."""
        from semblend.integration.trtllm.kv_cache_adapter import is_trtllm_layout

        # TRT-LLM: [100, 2, 128, 4, 128] -- tpb=128 > n_kv_heads=4
        kv = torch.zeros(100, 2, 128, 4, 128)
        assert is_trtllm_layout(kv) is True

    def test_vllm_not_detected(self):
        """vLLM layout is NOT detected as TRT-LLM."""
        from semblend.integration.trtllm.kv_cache_adapter import is_trtllm_layout

        # vLLM: [100, 2, 32, 16, 128] -- n_heads=32, block_size=16
        kv = torch.zeros(100, 2, 32, 16, 128)
        assert is_trtllm_layout(kv) is False

    def test_non_5d_rejected(self):
        """Non-5D tensors are rejected."""
        from semblend.integration.trtllm.kv_cache_adapter import is_trtllm_layout

        kv = torch.zeros(100, 2, 128, 128)
        assert is_trtllm_layout(kv) is False

    def test_trtllm_64_block(self):
        """TRT-LLM with 64-token blocks is detected."""
        from semblend.integration.trtllm.kv_cache_adapter import is_trtllm_layout

        kv = torch.zeros(100, 2, 64, 8, 128)
        assert is_trtllm_layout(kv) is True


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestStrideConsistency:
    """Verify that stride-based access produces correct values."""

    def test_write_read_roundtrip_trtllm(self):
        """Write data at specific positions using TRT-LLM strides, read back."""
        from semblend.integration.trtllm.kv_cache_adapter import trtllm_kv_strides

        # Create KV cache and write a known value
        kv = torch.zeros(10, 2, 128, 4, 128, dtype=torch.float32)
        strides = trtllm_kv_strides(kv)

        # Write to block=3, kv=0 (K), pos=42, head=2, dim=0..127
        block, kv_idx, pos, head = 3, 0, 42, 2
        offset = (
            block * strides.kv_stride_block
            + kv_idx * strides.kv_stride_kv
            + pos * strides.kv_stride_pos
            + head * strides.kv_stride_head
        )

        # Write via flat view
        flat = kv.view(-1)
        for d in range(128):
            flat[offset + d * strides.kv_stride_dim] = float(d + 1)

        # Read back via normal indexing
        expected = torch.arange(1, 129, dtype=torch.float32)
        actual = kv[block, kv_idx, pos, head, :]
        assert torch.allclose(actual, expected), f"Mismatch: {actual[:5]} vs {expected[:5]}"

    def test_write_read_roundtrip_vllm(self):
        """Write data at specific positions using vLLM strides, read back."""
        from semblend.integration.trtllm.kv_cache_adapter import vllm_kv_strides

        # vLLM: [num_blocks, 2, num_heads, block_size, head_dim]
        kv = torch.zeros(10, 2, 4, 128, 128, dtype=torch.float32)
        strides = vllm_kv_strides(kv)

        block, kv_idx, head, pos = 5, 1, 3, 77
        offset = (
            block * strides.kv_stride_block
            + kv_idx * strides.kv_stride_kv
            + head * strides.kv_stride_head
            + pos * strides.kv_stride_pos
        )

        flat = kv.view(-1)
        for d in range(128):
            flat[offset + d * strides.kv_stride_dim] = float(d + 100)

        expected = torch.arange(100, 228, dtype=torch.float32)
        actual = kv[block, kv_idx, head, pos, :]
        assert torch.allclose(actual, expected)

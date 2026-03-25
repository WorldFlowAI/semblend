"""Tests for SemShareKV embedding capture buffer."""
from __future__ import annotations

import numpy as np
import pytest

from semblend_core.semshare.embedding_capture import (
    EmbeddingCaptureBuffer,
    _is_layer_zero,
)


class TestEmbeddingCaptureBuffer:
    def test_capture_and_pop(self) -> None:
        buf = EmbeddingCaptureBuffer(max_entries=10)
        emb = np.random.randn(100, 128).astype(np.float32)

        buf.capture("req1", emb)
        assert buf.size == 1

        result = buf.pop("req1")
        assert result is not None
        assert result.shape == (100, 128)
        assert buf.size == 0

    def test_pop_returns_none_for_missing(self) -> None:
        buf = EmbeddingCaptureBuffer()
        assert buf.pop("nonexistent") is None

    def test_get_without_removing(self) -> None:
        buf = EmbeddingCaptureBuffer()
        emb = np.random.randn(50, 64).astype(np.float32)

        buf.capture("req1", emb)
        result = buf.get("req1")
        assert result is not None
        assert buf.size == 1  # still there

    def test_3d_input_reshaped(self) -> None:
        """3D input [seq_len, num_heads, head_dim] should be flattened."""
        buf = EmbeddingCaptureBuffer()
        emb = np.random.randn(50, 8, 64).astype(np.float32)  # 8 heads, 64 dim

        buf.capture("req1", emb)
        result = buf.pop("req1")
        assert result is not None
        assert result.shape == (50, 512)  # 8 * 64

    def test_capacity_eviction(self) -> None:
        buf = EmbeddingCaptureBuffer(max_entries=3)

        for i in range(5):
            buf.capture(f"req{i}", np.random.randn(10, 64).astype(np.float32))

        assert buf.size == 3
        # Oldest should have been evicted
        assert buf.get("req0") is None
        assert buf.get("req1") is None
        assert buf.get("req4") is not None

    def test_duplicate_capture_ignored(self) -> None:
        buf = EmbeddingCaptureBuffer()
        emb1 = np.ones((10, 64), dtype=np.float32)
        emb2 = np.zeros((10, 64), dtype=np.float32)

        buf.capture("req1", emb1)
        buf.capture("req1", emb2)  # should be ignored
        assert buf.size == 1

        result = buf.pop("req1")
        np.testing.assert_array_equal(result, emb1)  # first one kept

    def test_clear(self) -> None:
        buf = EmbeddingCaptureBuffer()
        buf.capture("req1", np.random.randn(10, 64).astype(np.float32))
        buf.capture("req2", np.random.randn(10, 64).astype(np.float32))
        assert buf.size == 2

        buf.clear()
        assert buf.size == 0


class TestIsLayerZero:
    def test_standard_vllm_layer_names(self) -> None:
        assert _is_layer_zero("model.layers.0.self_attn.kv_proj") is True
        assert _is_layer_zero("model.layers.1.self_attn.kv_proj") is False
        assert _is_layer_zero("model.layers.27.self_attn.kv_proj") is False

    def test_alternate_naming(self) -> None:
        assert _is_layer_zero("layers.0.attn") is True
        assert _is_layer_zero("layer.0.attention") is True

    def test_not_layer_zero(self) -> None:
        assert _is_layer_zero("model.layers.10.kv") is False
        assert _is_layer_zero("embeddings") is False

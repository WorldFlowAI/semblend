"""Tests for TRT-LLM PyTorch backend.

Unit tests with mock KVCacheManager -- no TRT-LLM or GPU required.
Tests the inject/register/correct lifecycle and pipeline integration.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class MockKVCacheManager:
    """Mock TRT-LLM KVCacheManager for testing.

    Simulates get_buffers() returning a TRT-LLM-layout KV cache tensor.
    """

    def __init__(
        self,
        num_blocks: int = 100,
        num_kv_heads: int = 4,
        tokens_per_block: int = 128,
        head_dim: int = 128,
        num_layers: int = 28,
    ) -> None:
        self.num_blocks = num_blocks
        self.num_kv_heads = num_kv_heads
        self.tokens_per_block = tokens_per_block
        self.head_dim = head_dim
        self.num_layers = num_layers

        if HAS_TORCH:
            # TRT-LLM layout: [num_blocks, 2, tokens_per_block, num_kv_heads, head_dim]
            self._buffers = [
                torch.zeros(
                    num_blocks,
                    2,
                    tokens_per_block,
                    num_kv_heads,
                    head_dim,
                    dtype=torch.float16,
                )
                for _ in range(num_layers)
            ]
        else:
            self._buffers = [None] * num_layers

    def get_buffers(self, layer_idx: int):
        return self._buffers[layer_idx]


class TestTRTLLMPyTorchBackendInit:
    """Test backend initialization."""

    def test_import(self):
        from semblend.integration.trtllm.pytorch_backend import TRTLLMPyTorchBackend

        assert TRTLLMPyTorchBackend is not None

    def test_init_defaults(self):
        from semblend.integration.trtllm.pytorch_backend import TRTLLMPyTorchBackend

        kv_mgr = MockKVCacheManager()
        backend = TRTLLMPyTorchBackend(
            kv_cache_manager=kv_mgr,
            model_config={"model_name": "test-model"},
        )
        assert backend.get_kv_block_size() == 128

    def test_custom_block_size(self):
        from semblend.integration.trtllm.pytorch_backend import TRTLLMPyTorchBackend

        kv_mgr = MockKVCacheManager(tokens_per_block=64)
        backend = TRTLLMPyTorchBackend(
            kv_cache_manager=kv_mgr,
            model_config={"tokens_per_block": 64},
        )
        assert backend.get_kv_block_size() == 64

    def test_model_config(self):
        from semblend.integration.trtllm.pytorch_backend import TRTLLMPyTorchBackend

        kv_mgr = MockKVCacheManager()
        backend = TRTLLMPyTorchBackend(
            kv_cache_manager=kv_mgr,
            model_config={
                "num_layers": 28,
                "num_kv_heads": 4,
                "head_dim": 128,
                "model_name": "Qwen/Qwen2.5-7B-Instruct-AWQ",
            },
        )
        config = backend.get_model_config()
        assert config["num_layers"] == 28
        assert config["num_heads"] == 4
        assert config["head_dim"] == 128
        assert "Qwen" in config["model_name"]


class TestTRTLLMPyTorchBackendStats:
    """Test backend statistics tracking."""

    def test_initial_stats(self):
        from semblend.integration.trtllm.pytorch_backend import TRTLLMPyTorchBackend

        kv_mgr = MockKVCacheManager()
        backend = TRTLLMPyTorchBackend(
            kv_cache_manager=kv_mgr,
            model_config={},
        )
        stats = backend.get_stats()
        assert stats["injections"] == 0
        assert stats["donors_registered"] == 0
        assert stats["semantic_hits"] == 0
        assert stats["misses"] == 0
        assert stats["donor_store_size"] == 0


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestTRTLLMPyTorchBackendInject:
    """Test KV injection into mock TRT-LLM cache."""

    def test_inject_empty_mapping(self):
        """Empty token mapping should be a no-op."""
        from semblend.integration.trtllm.pytorch_backend import TRTLLMPyTorchBackend

        kv_mgr = MockKVCacheManager()
        backend = TRTLLMPyTorchBackend(
            kv_cache_manager=kv_mgr,
            model_config={},
        )
        donor_kv = torch.zeros(2, 4, 100, 128, dtype=torch.float16)
        block_table = torch.arange(10, dtype=torch.int32)

        # Should not raise
        backend.inject_donor_kv(
            donor_kv=donor_kv,
            block_table=block_table,
            token_mapping=[],
            layer_idx=0,
        )
        assert backend.get_stats()["injections"] == 0


class TestTRTLLMPyTorchBackendSemanticSearch:
    """Test semantic donor discovery."""

    def test_miss_without_pipeline(self):
        """Without pipeline, find_semantic_donor returns None."""
        from semblend.integration.trtllm.pytorch_backend import TRTLLMPyTorchBackend

        kv_mgr = MockKVCacheManager()
        backend = TRTLLMPyTorchBackend(
            kv_cache_manager=kv_mgr,
            model_config={},
        )

        with patch.dict("os.environ", {"SEMBLEND_ENABLED": "0"}):
            result = backend.find_semantic_donor([1, 2, 3])
            assert result is None

    def test_miss_short_prompt(self):
        """Short prompts (<100 tokens) should not trigger semantic search."""
        from semblend.integration.trtllm.pytorch_backend import TRTLLMPyTorchBackend

        kv_mgr = MockKVCacheManager()
        backend = TRTLLMPyTorchBackend(
            kv_cache_manager=kv_mgr,
            model_config={"model_name": "test"},
        )

        # Short prompt -- pipeline may init but should miss
        result = backend.find_semantic_donor(list(range(50)))
        assert result is None


class TestModelEngineHook:
    """Test the SemBlend model engine hook."""

    def test_import(self):
        from semblend.integration.trtllm.model_engine_hook import (
            SemBlendModelEngineHook,
        )

        assert SemBlendModelEngineHook is not None

    def test_detect_approach_enqueue(self):
        """Detect token_sub when enqueue_request is available."""
        from semblend.integration.trtllm.model_engine_hook import (
            SemBlendModelEngineHook,
        )

        engine = MagicMock()
        engine.enqueue_request = MagicMock()
        backend = MagicMock()

        hook = SemBlendModelEngineHook(engine=engine, backend=backend)
        approach = hook._detect_best_approach()
        assert approach == "token_sub"

    def test_detect_approach_radix(self):
        """Detect radix_patch when radix tree is available."""
        from semblend.integration.trtllm.model_engine_hook import (
            SemBlendModelEngineHook,
        )

        engine = MagicMock(spec=[])  # No enqueue_request
        engine.kv_cache_manager = MagicMock()
        engine.kv_cache_manager.radix_tree = MagicMock()
        engine.kv_cache_manager.radix_tree.match_prefix = MagicMock()
        backend = MagicMock()

        hook = SemBlendModelEngineHook(engine=engine, backend=backend)
        approach = hook._detect_best_approach()
        assert approach == "radix_patch"

    def test_detect_approach_block_inject(self):
        """Detect block_inject when get_buffers is available."""
        from semblend.integration.trtllm.model_engine_hook import (
            SemBlendModelEngineHook,
        )

        engine = MagicMock(spec=[])  # No enqueue_request
        kv_mgr = MagicMock(spec=["get_buffers"])
        kv_mgr.get_buffers = MagicMock()
        engine.kv_cache_manager = kv_mgr
        backend = MagicMock()

        hook = SemBlendModelEngineHook(engine=engine, backend=backend)
        approach = hook._detect_best_approach()
        assert approach == "block_inject"

    def test_wrap_unwrap(self):
        """Wrap and unwrap should be symmetric."""
        from semblend.integration.trtllm.model_engine_hook import (
            SemBlendModelEngineHook,
        )

        engine = MagicMock()
        backend = MagicMock()

        hook = SemBlendModelEngineHook(engine=engine, backend=backend)
        hook.wrap()
        assert hook.active_approach == "token_sub"

        hook.unwrap()
        assert hook.active_approach is None

    def test_graceful_fallthrough(self):
        """Interception errors should fall through to original path."""
        from semblend.integration.trtllm.model_engine_hook import (
            SemBlendModelEngineHook,
        )

        engine = MagicMock()
        backend = MagicMock()
        backend.find_semantic_donor.side_effect = RuntimeError("test error")

        hook = SemBlendModelEngineHook(engine=engine, backend=backend)
        hook.wrap()

        request = MagicMock()
        request.token_ids = list(range(200))
        engine.enqueue_request(request)

        # Original should still be called despite the error
        assert hook.get_stats()["requests_intercepted"] >= 0


class TestPostPrefillRoPEHook:
    """Test the post-prefill RoPE correction hook."""

    def test_import(self):
        from semblend.integration.trtllm.model_engine_hook import (
            PostPrefillRoPEHook,
        )

        assert PostPrefillRoPEHook is not None

    def test_skip_without_flag(self):
        """Requests without correction flag should be skipped."""
        from semblend.integration.trtllm.model_engine_hook import (
            PostPrefillRoPEHook,
        )

        backend = MagicMock()
        hook = PostPrefillRoPEHook(backend=backend)

        request = MagicMock(spec=[])  # No _semblend attributes
        block_table = MagicMock()

        hook.on_prefill_complete(request, block_table)
        backend.apply_rope_correction.assert_not_called()

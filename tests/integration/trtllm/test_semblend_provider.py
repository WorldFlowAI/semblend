"""Tests for the SemBlend TRT-LLM provider (reference implementation).

Tests the concrete SemanticCacheLookupProvider and PostPrefixLoadHook
implementations without requiring TRT-LLM or GPU.
"""
from __future__ import annotations

from unittest.mock import patch


class TestSemBlendProviderImport:
    """Test import and initialization."""

    def test_import(self):
        from semblend.integration.trtllm.semblend_provider import SemBlendProvider
        assert SemBlendProvider is not None

    def test_init_defaults(self):
        from semblend.integration.trtllm.semblend_provider import SemBlendProvider
        provider = SemBlendProvider()
        assert provider is not None

    def test_init_custom(self):
        from semblend.integration.trtllm.semblend_provider import SemBlendProvider
        provider = SemBlendProvider(
            model_name="test-model",
            min_similarity=0.70,
            min_reuse_ratio=0.40,
            max_donors=5000,
            chunk_size=64,
        )
        stats = provider.get_stats()
        assert stats["queries"] == 0
        assert stats["hits"] == 0

    def test_implements_abc(self):
        from semblend.integration.trtllm.semblend_provider import SemBlendProvider
        from semblend.integration.trtllm.upstream_interface import (
            SemanticCacheLookupProvider,
        )
        provider = SemBlendProvider()
        assert isinstance(provider, SemanticCacheLookupProvider)


class TestSemBlendProviderSearch:
    """Test semantic search through the provider."""

    def test_miss_disabled(self):
        from semblend.integration.trtllm.semblend_provider import SemBlendProvider
        provider = SemBlendProvider()
        with patch.dict("os.environ", {"SEMBLEND_ENABLED": "0"}):
            result = provider.find_semantic_match([1, 2, 3], "test")
            assert result is None
            assert provider.get_stats()["misses"] == 1

    def test_miss_short_tokens(self):
        from semblend.integration.trtllm.semblend_provider import SemBlendProvider
        provider = SemBlendProvider()
        result = provider.find_semantic_match(list(range(50)), "short")
        assert result is None

    def test_miss_empty_store(self):
        from semblend.integration.trtllm.semblend_provider import SemBlendProvider
        provider = SemBlendProvider(model_name="test")
        # No donors registered, so search will miss
        result = provider.find_semantic_match(list(range(200)), "a long prompt text " * 20)
        assert result is None
        assert provider.get_stats()["misses"] == 1


class TestSemBlendProviderRegistration:
    """Test donor registration."""

    def test_register_skips_short(self):
        from semblend.integration.trtllm.semblend_provider import SemBlendProvider
        provider = SemBlendProvider()
        provider.register_completed("req-1", list(range(50)), "short")
        assert provider.get_stats()["registrations"] == 0

    def test_on_eviction(self):
        from semblend.integration.trtllm.semblend_provider import SemBlendProvider
        provider = SemBlendProvider()
        provider.on_eviction("req-1")
        assert provider.get_stats()["evictions"] == 1


class TestSemBlendPostLoadHook:
    """Test the RoPE correction post-load hook."""

    def test_import(self):
        from semblend.integration.trtllm.semblend_provider import (
            SemBlendPostLoadHook,
        )
        assert SemBlendPostLoadHook is not None

    def test_implements_abc(self):
        from semblend.integration.trtllm.semblend_provider import (
            SemBlendPostLoadHook,
        )
        from semblend.integration.trtllm.upstream_interface import (
            PostPrefixLoadHook,
        )
        hook = SemBlendPostLoadHook()
        assert isinstance(hook, PostPrefixLoadHook)

    def test_skip_empty_mapping(self):
        from semblend.integration.trtllm.semblend_provider import (
            SemBlendPostLoadHook,
        )
        hook = SemBlendPostLoadHook()
        hook.on_prefix_loaded(
            kv_buffers=[],
            block_table=None,
            position_mapping=[],
            rope_config={},
        )
        assert hook.get_stats()["corrections_applied"] == 0

    def test_skip_identity_mapping(self):
        from semblend.integration.trtllm.semblend_provider import (
            SemBlendPostLoadHook,
        )
        hook = SemBlendPostLoadHook()
        # All positions match (no delta needed)
        hook.on_prefix_loaded(
            kv_buffers=[None],
            block_table=None,
            position_mapping=[(0, 0), (1, 1), (2, 2)],
            rope_config={},
        )
        assert hook.get_stats()["corrections_applied"] == 0

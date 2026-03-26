"""Tests for proposed upstream TRT-LLM semantic cache interfaces.

Verifies the ABCs can be imported and implemented correctly.
These tests validate the interface design, not TRT-LLM internals.
"""

from __future__ import annotations

import pytest


class TestSemanticCacheLookupProvider:
    """Test the SemanticCacheLookupProvider ABC."""

    def test_import(self):
        from semblend.integration.trtllm.upstream_interface import (
            SemanticCacheLookupProvider,
        )

        assert SemanticCacheLookupProvider is not None

    def test_cannot_instantiate_directly(self):
        from semblend.integration.trtllm.upstream_interface import (
            SemanticCacheLookupProvider,
        )

        with pytest.raises(TypeError, match="abstract"):
            SemanticCacheLookupProvider()

    def test_concrete_implementation(self):
        from semblend.integration.trtllm.upstream_interface import (
            SemanticCacheLookupProvider,
            SemanticMatchResult,
        )

        class TestProvider(SemanticCacheLookupProvider):
            def __init__(self):
                self.registered = []

            def find_semantic_match(self, token_ids, prompt_text):
                if len(token_ids) > 100:
                    return SemanticMatchResult(
                        donor_token_ids=[1, 2, 3],
                        similarity=0.85,
                        reuse_ratio=0.70,
                        position_mapping=[(0, 0), (1, 1)],
                        donor_id="test-donor",
                    )
                return None

            def register_completed(self, request_id, token_ids, prompt_text):
                self.registered.append(request_id)

        provider = TestProvider()
        result = provider.find_semantic_match(list(range(200)), "test prompt")
        assert result is not None
        assert result.similarity == 0.85
        assert result.donor_token_ids == [1, 2, 3]

        provider.register_completed("req-1", [1, 2, 3], "test")
        assert "req-1" in provider.registered

    def test_on_eviction_default_noop(self):
        from semblend.integration.trtllm.upstream_interface import (
            SemanticCacheLookupProvider,
        )

        class MinimalProvider(SemanticCacheLookupProvider):
            def find_semantic_match(self, token_ids, prompt_text):
                return None

            def register_completed(self, request_id, token_ids, prompt_text):
                pass

        provider = MinimalProvider()
        # Should not raise -- default on_eviction is a no-op
        provider.on_eviction("test-request")


class TestPostPrefixLoadHook:
    """Test the PostPrefixLoadHook ABC."""

    def test_import(self):
        from semblend.integration.trtllm.upstream_interface import (
            PostPrefixLoadHook,
        )

        assert PostPrefixLoadHook is not None

    def test_cannot_instantiate_directly(self):
        from semblend.integration.trtllm.upstream_interface import (
            PostPrefixLoadHook,
        )

        with pytest.raises(TypeError, match="abstract"):
            PostPrefixLoadHook()

    def test_concrete_implementation(self):
        from semblend.integration.trtllm.upstream_interface import (
            PostPrefixLoadHook,
        )

        class TestHook(PostPrefixLoadHook):
            def __init__(self):
                self.calls = []

            def on_prefix_loaded(self, kv_buffers, block_table, position_mapping, rope_config):
                self.calls.append(
                    {
                        "num_layers": len(kv_buffers),
                        "num_pairs": len(position_mapping),
                        "rope_base": rope_config.get("rope_base", 10000.0),
                    }
                )

        hook = TestHook()
        hook.on_prefix_loaded(
            kv_buffers=[None] * 28,
            block_table=None,
            position_mapping=[(0, 5), (1, 6), (2, 7)],
            rope_config={"rope_base": 10000.0, "head_dim": 128},
        )
        assert len(hook.calls) == 1
        assert hook.calls[0]["num_layers"] == 28
        assert hook.calls[0]["num_pairs"] == 3


class TestSemanticMatchResult:
    """Test the SemanticMatchResult dataclass."""

    def test_import(self):
        from semblend.integration.trtllm.upstream_interface import (
            SemanticMatchResult,
        )

        assert SemanticMatchResult is not None

    def test_creation(self):
        from semblend.integration.trtllm.upstream_interface import (
            SemanticMatchResult,
        )

        result = SemanticMatchResult(
            donor_token_ids=[1, 2, 3, 4, 5],
            similarity=0.92,
            reuse_ratio=0.85,
        )
        assert result.donor_token_ids == [1, 2, 3, 4, 5]
        assert result.similarity == 0.92
        assert result.reuse_ratio == 0.85
        assert result.position_mapping == []
        assert result.donor_id == ""
        assert result.metadata == {}

    def test_frozen(self):
        from semblend.integration.trtllm.upstream_interface import (
            SemanticMatchResult,
        )

        result = SemanticMatchResult(
            donor_token_ids=[1],
            similarity=0.5,
            reuse_ratio=0.5,
        )
        with pytest.raises(AttributeError):
            result.similarity = 0.9

    def test_with_position_mapping(self):
        from semblend.integration.trtllm.upstream_interface import (
            SemanticMatchResult,
        )

        mapping = [(0, 5), (1, 6), (2, 7), (3, 8)]
        result = SemanticMatchResult(
            donor_token_ids=[10, 20, 30, 40],
            similarity=0.88,
            reuse_ratio=0.75,
            position_mapping=mapping,
            donor_id="donor-123",
        )
        assert len(result.position_mapping) == 4
        assert result.position_mapping[0] == (0, 5)
        assert result.donor_id == "donor-123"

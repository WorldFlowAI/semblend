"""Tests for the Dynamo KvIndexer wrapper with SemBlend semantic fallback.

Uses a mock KvIndexer to test the wrapper logic without Dynamo installed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class MockKvIndexer:
    """Mock Dynamo KvIndexer for testing."""

    def __init__(self, overlap_scores: dict | None = None):
        self._scores = overlap_scores or {}
        self._calls = []

    def find_matches_for_request(self, tokens, lora_name=None):
        self._calls.append(("find_matches", len(tokens), lora_name))
        return dict(self._scores)

    def apply_event(self, event):
        self._calls.append(("apply_event", event))

    def remove_worker(self, worker_id):
        self._calls.append(("remove_worker", worker_id))

    def remove_worker_dp_rank(self, worker_id, dp_rank):
        self._calls.append(("remove_worker_dp_rank", worker_id, dp_rank))

    def shutdown(self):
        self._calls.append(("shutdown",))


class TestSemBlendKvIndexerWrapperImport:
    """Test import and basic initialization."""

    def test_import(self):
        from semblend.integration.dynamo.kv_indexer_wrapper import (
            SemBlendKvIndexerWrapper,
        )

        assert SemBlendKvIndexerWrapper is not None

    def test_init(self):
        from semblend.integration.dynamo.kv_indexer_wrapper import (
            SemBlendKvIndexerWrapper,
        )

        mock = MockKvIndexer()
        wrapper = SemBlendKvIndexerWrapper(
            inner_indexer=mock,
            kv_block_size=32,
            min_similarity=0.60,
        )
        assert wrapper is not None

    def test_stats_initial(self):
        from semblend.integration.dynamo.kv_indexer_wrapper import (
            SemBlendKvIndexerWrapper,
        )

        mock = MockKvIndexer()
        wrapper = SemBlendKvIndexerWrapper(inner_indexer=mock)
        stats = wrapper.get_stats()
        assert stats["total_queries"] == 0
        assert stats["exact_hits"] == 0
        assert stats["semantic_hits"] == 0


class TestExactPrefixPassthrough:
    """Test that high-overlap queries pass through to inner indexer."""

    def test_high_overlap_returns_exact(self):
        from semblend.integration.dynamo.kv_indexer_wrapper import (
            SemBlendKvIndexerWrapper,
        )

        # Mock returns high overlap (worker-0 has 20 out of 32 blocks)
        mock = MockKvIndexer(overlap_scores={"worker-0": 20})
        wrapper = SemBlendKvIndexerWrapper(
            inner_indexer=mock,
            kv_block_size=32,
            min_overlap_ratio=0.50,
        )

        # 1024 tokens = 32 blocks at block_size=32
        tokens = list(range(1024))
        scores = wrapper.find_matches_for_request(tokens)

        assert scores == {"worker-0": 20}
        assert wrapper.get_stats()["exact_hits"] == 1
        assert wrapper.get_stats()["total_queries"] == 1

    def test_zero_overlap_triggers_semantic(self):
        from semblend.integration.dynamo.kv_indexer_wrapper import (
            SemBlendKvIndexerWrapper,
        )

        mock = MockKvIndexer(overlap_scores={})
        wrapper = SemBlendKvIndexerWrapper(
            inner_indexer=mock,
            kv_block_size=32,
            min_overlap_ratio=0.50,
        )

        # Short tokens — semantic search skipped for <100 tokens
        tokens = list(range(50))
        scores = wrapper.find_matches_for_request(tokens)

        assert scores == {}
        # Not counted as exact hit (overlap was 0)
        assert wrapper.get_stats()["exact_hits"] == 0
        assert wrapper.get_stats()["semantic_misses"] == 1

    def test_low_overlap_long_prompt_triggers_semantic(self):
        """Low overlap on long prompt should attempt semantic search."""
        from semblend.integration.dynamo.kv_indexer_wrapper import (
            SemBlendKvIndexerWrapper,
        )

        mock = MockKvIndexer(overlap_scores={"worker-0": 1})
        wrapper = SemBlendKvIndexerWrapper(
            inner_indexer=mock,
            kv_block_size=32,
            min_overlap_ratio=0.50,
        )

        # Disable SemBlend to test the fallback path
        with patch.dict("os.environ", {"SEMBLEND_ENABLED": "0"}):
            tokens = list(range(1024))
            scores = wrapper.find_matches_for_request(tokens)

        # Should still return exact scores (semantic disabled)
        assert scores == {"worker-0": 1}
        assert wrapper.get_stats()["semantic_misses"] == 1


class TestEventForwarding:
    """Test that events are forwarded to inner indexer."""

    def test_apply_event_forwarded(self):
        from semblend.integration.dynamo.kv_indexer_wrapper import (
            SemBlendKvIndexerWrapper,
        )

        mock = MockKvIndexer()
        wrapper = SemBlendKvIndexerWrapper(inner_indexer=mock)

        event = {"type": "stored", "blocks": [1, 2, 3]}
        wrapper.apply_event(event)

        assert ("apply_event", event) in mock._calls

    def test_remove_worker_forwarded(self):
        from semblend.integration.dynamo.kv_indexer_wrapper import (
            SemBlendKvIndexerWrapper,
        )

        mock = MockKvIndexer()
        wrapper = SemBlendKvIndexerWrapper(inner_indexer=mock)
        wrapper.remove_worker("worker-0")

        assert ("remove_worker", "worker-0") in mock._calls

    def test_shutdown_forwarded(self):
        from semblend.integration.dynamo.kv_indexer_wrapper import (
            SemBlendKvIndexerWrapper,
        )

        mock = MockKvIndexer()
        wrapper = SemBlendKvIndexerWrapper(inner_indexer=mock)
        wrapper.shutdown()

        assert ("shutdown",) in mock._calls


class TestDonorRegistration:
    """Test donor registration through the wrapper."""

    def test_register_short_prompt_skipped(self):
        from semblend.integration.dynamo.kv_indexer_wrapper import (
            SemBlendKvIndexerWrapper,
        )

        mock = MockKvIndexer()
        wrapper = SemBlendKvIndexerWrapper(inner_indexer=mock)

        # Short prompt — should be skipped
        wrapper.register_completed_request(
            request_id="req-1",
            tokens=list(range(50)),
            prompt_text="short prompt",
        )
        assert wrapper.get_stats()["donors_registered"] == 0


class TestEventPublisher:
    """Test the event publisher wrapper."""

    def test_import(self):
        from semblend.integration.dynamo.event_publisher import (
            SemBlendEventPublisher,
        )

        assert SemBlendEventPublisher is not None

    def test_publish_stored_forwarded(self):
        from semblend.integration.dynamo.event_publisher import (
            SemBlendEventPublisher,
        )

        mock_inner = MagicMock()
        publisher = SemBlendEventPublisher(
            inner_publisher=mock_inner,
            worker_id="worker-0",
        )

        publisher.publish_stored(
            block_hashes=[1, 2, 3],
            parent_block_hash=0,
            token_ids=list(range(50)),  # Too short for semantic
            block_size=32,
        )

        mock_inner.publish_stored.assert_called_once()
        assert publisher.get_stats()["block_events_published"] == 1

    def test_publish_removed_forwarded(self):
        from semblend.integration.dynamo.event_publisher import (
            SemBlendEventPublisher,
        )

        mock_inner = MagicMock()
        publisher = SemBlendEventPublisher(inner_publisher=mock_inner)

        publisher.publish_removed(block_hashes=[1, 2])
        mock_inner.publish_removed.assert_called_once()

    def test_shutdown_forwarded(self):
        from semblend.integration.dynamo.event_publisher import (
            SemBlendEventPublisher,
        )

        mock_inner = MagicMock()
        publisher = SemBlendEventPublisher(inner_publisher=mock_inner)
        publisher.shutdown()
        mock_inner.shutdown.assert_called_once()

"""Unit tests for the segment-level KV cache client."""

from __future__ import annotations

import base64
import hashlib
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from synapse_kv_connector.segment_client import (
    FindDonorResult,
    KvSlotAction,
    KvTransferPlan,
    LoadSegmentResult,
    SegmentKvStats,
    SegmentSearchHit,
    SegmentSearchResult,
    SegmentTypeCount,
    SynapseSegmentClient,
    compute_segment_hash,
)


# ---------------------------------------------------------------------------
# Segment hash tests
# ---------------------------------------------------------------------------


class TestComputeSegmentHash:
    """Tests for the segment hash function."""

    def test_deterministic(self) -> None:
        """Same inputs always produce the same hash."""
        h1 = compute_segment_hash("system_prompt", "You are a helpful assistant.")
        h2 = compute_segment_hash("system_prompt", "You are a helpful assistant.")
        assert h1 == h2

    def test_different_type_different_hash(self) -> None:
        """Different segment types produce different hashes."""
        h1 = compute_segment_hash("system_prompt", "hello")
        h2 = compute_segment_hash("user_query", "hello")
        assert h1 != h2

    def test_different_text_different_hash(self) -> None:
        """Different text produces different hashes."""
        h1 = compute_segment_hash("system_prompt", "hello")
        h2 = compute_segment_hash("system_prompt", "world")
        assert h1 != h2

    def test_hash_format(self) -> None:
        """Hash is a hex string of expected length."""
        result = compute_segment_hash("rag_context", "some context text")
        assert isinstance(result, str)
        # Blake3 produces 256-bit (64 hex char) digests
        assert len(result) == 64
        # All characters are hex
        assert all(c in "0123456789abcdef" for c in result)

    def test_uses_colon_separator(self) -> None:
        """Hash is computed over 'type:text' with colon separator."""
        # This tests the concatenation format by verifying that
        # "a:b" (type="a", text="b") differs from "ab:" (type="ab", text="")
        h1 = compute_segment_hash("a", "b")
        h2 = compute_segment_hash("ab", "")
        assert h1 != h2


# ---------------------------------------------------------------------------
# Store request construction tests
# ---------------------------------------------------------------------------


class TestStoreSegmentKvRequest:
    """Tests for store_segment_kv request construction."""

    @patch("synapse_kv_connector.segment_client.requests.Session")
    def test_request_body_structure(self, mock_session_cls: MagicMock) -> None:
        """store_segment_kv sends correctly structured JSON."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "segmentHash": "abc123",
            "bytesStored": 128,
            "overwroteExisting": False,
        }
        mock_session.put.return_value = mock_response
        mock_session_cls.return_value = mock_session

        client = SynapseSegmentClient(
            base_url="http://localhost:8080",
            tenant_id="test-tenant",
        )

        kv_data = np.ones((2, 2, 4, 8, 64), dtype=np.float16)
        embedding = [0.1, 0.2, 0.3]

        client.store_segment_kv(
            segment_hash="abc123",
            segment_type="system_prompt",
            kv_data=kv_data,
            num_tokens=8,
            num_layers=2,
            num_heads=4,
            head_dim=64,
            embedding=embedding,
            text_preview="You are a helpful assistant.",
        )

        call_args = mock_session.put.call_args
        url = call_args[0][0]
        body = call_args[1]["json"]

        assert url == "http://localhost:8080/api/v1/kv-cache/segments/abc123"
        assert body["segmentType"] == "system_prompt"
        assert body["numTokens"] == 8
        assert body["numLayers"] == 2
        assert body["numHeads"] == 4
        assert body["headDim"] == 64
        assert body["tenantId"] == "test-tenant"
        assert body["embedding"] == [0.1, 0.2, 0.3]
        assert body["textPreview"] == "You are a helpful assistant."
        assert isinstance(body["kvData"], str)

    @patch("synapse_kv_connector.segment_client.requests.Session")
    def test_optional_fields_omitted(
        self, mock_session_cls: MagicMock
    ) -> None:
        """Optional embedding and textPreview are omitted when None."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "segmentHash": "def456",
            "bytesStored": 64,
            "overwroteExisting": False,
        }
        mock_session.put.return_value = mock_response
        mock_session_cls.return_value = mock_session

        client = SynapseSegmentClient(base_url="http://localhost:8080")

        kv_data = np.zeros((1, 2, 2, 4, 32), dtype=np.float16)
        client.store_segment_kv(
            segment_hash="def456",
            segment_type="rag_context",
            kv_data=kv_data,
            num_tokens=4,
            num_layers=1,
            num_heads=2,
            head_dim=32,
        )

        body = mock_session.put.call_args[1]["json"]
        assert "embedding" not in body
        assert "textPreview" not in body

    @patch("synapse_kv_connector.segment_client.requests.Session")
    def test_rejects_non_float16(self, mock_session_cls: MagicMock) -> None:
        """store_segment_kv rejects non-float16 arrays."""
        mock_session_cls.return_value = MagicMock()
        client = SynapseSegmentClient(base_url="http://localhost:8080")

        kv_data = np.ones((1, 2, 2, 4, 32), dtype=np.float32)
        with pytest.raises(ValueError, match="float16"):
            client.store_segment_kv(
                segment_hash="bad",
                segment_type="test",
                kv_data=kv_data,
                num_tokens=4,
                num_layers=1,
                num_heads=2,
                head_dim=32,
            )


# ---------------------------------------------------------------------------
# Load response parsing tests
# ---------------------------------------------------------------------------


class TestLoadSegmentKvResponse:
    """Tests for load_segment_kv response parsing."""

    @patch("synapse_kv_connector.segment_client.requests.Session")
    def test_parses_response(self, mock_session_cls: MagicMock) -> None:
        """load_segment_kv correctly parses the gateway response."""
        original = np.random.randn(64).astype(np.float16)
        kv_b64 = base64.standard_b64encode(original.tobytes()).decode("ascii")

        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "segmentType": "system_prompt",
            "numTokens": 32,
            "numLayers": 2,
            "numHeads": 4,
            "headDim": 64,
            "kvData": kv_b64,
            "embedding": [0.1, 0.2, 0.3],
        }
        mock_session.get.return_value = mock_response
        mock_session_cls.return_value = mock_session

        client = SynapseSegmentClient(base_url="http://localhost:8080")
        result = client.load_segment_kv("abc123")

        assert result is not None
        assert isinstance(result, LoadSegmentResult)
        assert result.segment_type == "system_prompt"
        assert result.num_tokens == 32
        assert result.num_layers == 2
        assert result.num_heads == 4
        assert result.head_dim == 64
        assert result.embedding == [0.1, 0.2, 0.3]
        np.testing.assert_array_equal(result.kv_data, original)

    @patch("synapse_kv_connector.segment_client.requests.Session")
    def test_returns_none_on_404(self, mock_session_cls: MagicMock) -> None:
        """load_segment_kv returns None when not found."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_session.get.return_value = mock_response
        mock_session_cls.return_value = mock_session

        client = SynapseSegmentClient(base_url="http://localhost:8080")
        assert client.load_segment_kv("nonexistent") is None

    @patch("synapse_kv_connector.segment_client.requests.Session")
    def test_parses_response_without_embedding(
        self, mock_session_cls: MagicMock
    ) -> None:
        """load_segment_kv handles missing embedding field."""
        kv_b64 = base64.standard_b64encode(b"\x00" * 16).decode("ascii")

        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "segmentType": "user_query",
            "numTokens": 4,
            "numLayers": 1,
            "numHeads": 2,
            "headDim": 32,
            "kvData": kv_b64,
        }
        mock_session.get.return_value = mock_response
        mock_session_cls.return_value = mock_session

        client = SynapseSegmentClient(base_url="http://localhost:8080")
        result = client.load_segment_kv("noembedding")

        assert result is not None
        assert result.embedding is None


# ---------------------------------------------------------------------------
# Delete tests
# ---------------------------------------------------------------------------


class TestDeleteSegmentKv:
    """Tests for delete_segment_kv."""

    @patch("synapse_kv_connector.segment_client.requests.Session")
    def test_returns_true_on_success(
        self, mock_session_cls: MagicMock
    ) -> None:
        """delete_segment_kv returns True on 204."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_session.delete.return_value = mock_response
        mock_session_cls.return_value = mock_session

        client = SynapseSegmentClient(base_url="http://localhost:8080")
        assert client.delete_segment_kv("abc123") is True

    @patch("synapse_kv_connector.segment_client.requests.Session")
    def test_returns_false_on_404(
        self, mock_session_cls: MagicMock
    ) -> None:
        """delete_segment_kv returns False when not found."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_session.delete.return_value = mock_response
        mock_session_cls.return_value = mock_session

        client = SynapseSegmentClient(base_url="http://localhost:8080")
        assert client.delete_segment_kv("nonexistent") is False


# ---------------------------------------------------------------------------
# Search request/response tests
# ---------------------------------------------------------------------------


class TestSearchSegments:
    """Tests for search_segments request and response handling."""

    @patch("synapse_kv_connector.segment_client.requests.Session")
    def test_request_body_structure(self, mock_session_cls: MagicMock) -> None:
        """search_segments sends correctly structured JSON."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [],
            "totalSearched": 0,
        }
        mock_session.post.return_value = mock_response
        mock_session_cls.return_value = mock_session

        client = SynapseSegmentClient(
            base_url="http://localhost:8080",
            tenant_id="my-tenant",
        )
        client.search_segments(
            embedding=[0.1, 0.2, 0.3],
            top_k=10,
            threshold=0.9,
            segment_type="rag_context",
        )

        call_args = mock_session.post.call_args
        url = call_args[0][0]
        body = call_args[1]["json"]

        assert url == "http://localhost:8080/api/v1/kv-cache/segments/search"
        assert body["embedding"] == [0.1, 0.2, 0.3]
        assert body["topK"] == 10
        assert body["threshold"] == 0.9
        assert body["segmentType"] == "rag_context"
        assert body["tenantId"] == "my-tenant"

    @patch("synapse_kv_connector.segment_client.requests.Session")
    def test_parses_search_results(
        self, mock_session_cls: MagicMock
    ) -> None:
        """search_segments correctly parses the response."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "segmentHash": "hash1",
                    "segmentType": "system_prompt",
                    "numTokens": 64,
                    "similarity": 0.95,
                    "textPreview": "You are helpful.",
                },
                {
                    "segmentHash": "hash2",
                    "segmentType": "rag_context",
                    "numTokens": 128,
                    "similarity": 0.88,
                },
            ],
            "totalSearched": 100,
        }
        mock_session.post.return_value = mock_response
        mock_session_cls.return_value = mock_session

        client = SynapseSegmentClient(base_url="http://localhost:8080")
        result = client.search_segments(embedding=[0.5, 0.5])

        assert isinstance(result, SegmentSearchResult)
        assert result.total_searched == 100
        assert len(result.results) == 2

        hit0 = result.results[0]
        assert isinstance(hit0, SegmentSearchHit)
        assert hit0.segment_hash == "hash1"
        assert hit0.segment_type == "system_prompt"
        assert hit0.num_tokens == 64
        assert abs(hit0.similarity - 0.95) < 1e-6
        assert hit0.text_preview == "You are helpful."

        hit1 = result.results[1]
        assert hit1.text_preview is None

    @patch("synapse_kv_connector.segment_client.requests.Session")
    def test_omits_segment_type_when_none(
        self, mock_session_cls: MagicMock
    ) -> None:
        """search_segments omits segmentType when not specified."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [],
            "totalSearched": 0,
        }
        mock_session.post.return_value = mock_response
        mock_session_cls.return_value = mock_session

        client = SynapseSegmentClient(base_url="http://localhost:8080")
        client.search_segments(embedding=[1.0])

        body = mock_session.post.call_args[1]["json"]
        assert "segmentType" not in body

    def test_rejects_empty_embedding(self) -> None:
        """search_segments raises ValueError on empty embedding."""
        client = SynapseSegmentClient(base_url="http://localhost:8080")
        with pytest.raises(ValueError, match="must not be empty"):
            client.search_segments(embedding=[])


# ---------------------------------------------------------------------------
# Stats tests
# ---------------------------------------------------------------------------


class TestSegmentGetStats:
    """Tests for get_stats response parsing."""

    @patch("synapse_kv_connector.segment_client.requests.Session")
    def test_parses_stats_with_type_counts(
        self, mock_session_cls: MagicMock
    ) -> None:
        """get_stats correctly parses stats with type breakdowns."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "entryCount": 25,
            "totalBytes": 524288,
            "maxEntries": 131072,
            "fillRatio": 0.000191,
            "typeCounts": [
                {"segmentType": "system_prompt", "count": 10},
                {"segmentType": "rag_context", "count": 15},
            ],
        }
        mock_session.get.return_value = mock_response
        mock_session_cls.return_value = mock_session

        client = SynapseSegmentClient(base_url="http://localhost:8080")
        stats = client.get_stats()

        assert isinstance(stats, SegmentKvStats)
        assert stats.entry_count == 25
        assert stats.total_bytes == 524288
        assert stats.max_entries == 131072
        assert abs(stats.fill_ratio - 0.000191) < 1e-9
        assert len(stats.type_counts) == 2
        assert isinstance(stats.type_counts[0], SegmentTypeCount)
        assert stats.type_counts[0].segment_type == "system_prompt"
        assert stats.type_counts[0].count == 10
        assert stats.type_counts[1].segment_type == "rag_context"
        assert stats.type_counts[1].count == 15

    @patch("synapse_kv_connector.segment_client.requests.Session")
    def test_parses_stats_without_type_counts(
        self, mock_session_cls: MagicMock
    ) -> None:
        """get_stats handles missing typeCounts gracefully."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "entryCount": 0,
            "totalBytes": 0,
            "maxEntries": 131072,
            "fillRatio": 0.0,
        }
        mock_session.get.return_value = mock_response
        mock_session_cls.return_value = mock_session

        client = SynapseSegmentClient(base_url="http://localhost:8080")
        stats = client.get_stats()

        assert stats.entry_count == 0
        assert stats.type_counts == []


# ---------------------------------------------------------------------------
# Transfer plan tests
# ---------------------------------------------------------------------------


class TestRequestTransferPlan:
    """Tests for request_transfer_plan."""

    @patch("synapse_kv_connector.segment_client.requests.Session")
    def test_request_body_structure(self, mock_session_cls: MagicMock) -> None:
        """request_transfer_plan sends correctly structured JSON."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "plan": {
                "donorId": "donor-1",
                "targetLen": 5,
                "donorLen": 5,
                "slotActions": [
                    {"action": "copy_from_donor", "donorPos": 0, "targetPos": 0},
                    {"action": "placeholder", "targetPos": 1},
                ],
                "copyPositions": [0],
                "placeholderPositions": [1],
                "numCopied": 1,
                "numPlaceholders": 1,
                "reuseRatio": 0.5,
                "editScript": {},
            },
            "viable": True,
        }
        mock_session.post.return_value = mock_response
        mock_session_cls.return_value = mock_session

        client = SynapseSegmentClient(base_url="http://localhost:8080")
        plan = client.request_transfer_plan(
            donor_tokens=[1, 2, 3, 4, 5],
            target_tokens=[1, 6, 3, 4, 5],
            donor_id="donor-1",
            min_reuse_ratio=0.4,
        )

        call_args = mock_session.post.call_args
        url = call_args[0][0]
        body = call_args[1]["json"]

        assert url == "http://localhost:8080/api/v1/kv-cache/segments/plan"
        assert body["donorTokens"] == [1, 2, 3, 4, 5]
        assert body["targetTokens"] == [1, 6, 3, 4, 5]
        assert body["donorId"] == "donor-1"
        assert body["minReuseRatio"] == 0.4

        assert isinstance(plan, KvTransferPlan)
        assert plan.donor_id == "donor-1"
        assert plan.viable is True
        assert plan.num_copied == 1
        assert len(plan.slot_actions) == 2

    @patch("synapse_kv_connector.segment_client.requests.Session")
    def test_includes_layer_deviations(
        self, mock_session_cls: MagicMock
    ) -> None:
        """request_transfer_plan includes layer deviations when given."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "plan": {
                "donorId": "d",
                "targetLen": 2,
                "donorLen": 2,
                "slotActions": [],
                "copyPositions": [],
                "placeholderPositions": [],
                "numCopied": 0,
                "numPlaceholders": 0,
                "reuseRatio": 0.0,
            },
            "viable": False,
        }
        mock_session.post.return_value = mock_response
        mock_session_cls.return_value = mock_session

        client = SynapseSegmentClient(base_url="http://localhost:8080")
        client.request_transfer_plan(
            donor_tokens=[1, 2],
            target_tokens=[3, 4],
            donor_id="d",
            layer_deviations=[0.1, 0.5, 0.2],
            num_layers=3,
        )

        body = mock_session.post.call_args[1]["json"]
        assert body["layerDeviations"] == [0.1, 0.5, 0.2]
        assert body["numLayers"] == 3

    @patch("synapse_kv_connector.segment_client.requests.Session")
    def test_parses_slot_actions(self, mock_session_cls: MagicMock) -> None:
        """request_transfer_plan correctly parses slot actions."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "plan": {
                "donorId": "d",
                "targetLen": 3,
                "donorLen": 3,
                "slotActions": [
                    {"action": "copy_from_donor", "donorPos": 0, "targetPos": 0},
                    {"action": "placeholder", "targetPos": 1},
                    {"action": "copy_from_donor", "donorPos": 2, "targetPos": 2},
                ],
                "copyPositions": [0, 2],
                "placeholderPositions": [1],
                "numCopied": 2,
                "numPlaceholders": 1,
                "reuseRatio": 0.667,
            },
            "viable": True,
        }
        mock_session.post.return_value = mock_response
        mock_session_cls.return_value = mock_session

        client = SynapseSegmentClient(base_url="http://localhost:8080")
        plan = client.request_transfer_plan(
            donor_tokens=[1, 2, 3],
            target_tokens=[1, 9, 3],
            donor_id="d",
        )

        assert len(plan.slot_actions) == 3
        assert plan.slot_actions[0] == KvSlotAction(
            action="copy_from_donor", donor_pos=0, target_pos=0
        )
        assert plan.slot_actions[1] == KvSlotAction(
            action="placeholder", donor_pos=None, target_pos=1
        )
        assert plan.copy_positions == [0, 2]
        assert plan.placeholder_positions == [1]


class TestFindDonorAndPlan:
    """Tests for find_donor_and_plan."""

    @patch("synapse_kv_connector.segment_client.requests.Session")
    def test_found_with_plan(self, mock_session_cls: MagicMock) -> None:
        """find_donor_and_plan returns plan when donor found."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "found": True,
            "plan": {
                "donorId": "req-42",
                "targetLen": 10,
                "donorLen": 10,
                "slotActions": [],
                "copyPositions": [0, 1, 2],
                "placeholderPositions": [3],
                "numCopied": 3,
                "numPlaceholders": 1,
                "reuseRatio": 0.75,
            },
            "treeSize": 5,
        }
        mock_session.post.return_value = mock_response
        mock_session_cls.return_value = mock_session

        client = SynapseSegmentClient(base_url="http://localhost:8080")
        result = client.find_donor_and_plan(
            target_tokens=[10, 20, 30, 40],
        )

        assert isinstance(result, FindDonorResult)
        assert result.found is True
        assert result.tree_size == 5
        assert result.plan is not None
        assert result.plan.donor_id == "req-42"
        assert result.plan.num_copied == 3

    @patch("synapse_kv_connector.segment_client.requests.Session")
    def test_not_found(self, mock_session_cls: MagicMock) -> None:
        """find_donor_and_plan returns empty when no donor."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "found": False,
            "plan": None,
            "treeSize": 0,
        }
        mock_session.post.return_value = mock_response
        mock_session_cls.return_value = mock_session

        client = SynapseSegmentClient(base_url="http://localhost:8080")
        result = client.find_donor_and_plan(target_tokens=[1, 2, 3])

        assert result.found is False
        assert result.plan is None
        assert result.tree_size == 0

    @patch("synapse_kv_connector.segment_client.requests.Session")
    def test_request_body(self, mock_session_cls: MagicMock) -> None:
        """find_donor_and_plan sends correct body."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "found": False,
            "plan": None,
            "treeSize": 0,
        }
        mock_session.post.return_value = mock_response
        mock_session_cls.return_value = mock_session

        client = SynapseSegmentClient(base_url="http://localhost:8080")
        client.find_donor_and_plan(
            target_tokens=[5, 6, 7],
            min_reuse_ratio=0.6,
            layer_deviation_threshold=0.4,
        )

        body = mock_session.post.call_args[1]["json"]
        assert body["targetTokens"] == [5, 6, 7]
        assert body["minReuseRatio"] == 0.6
        assert body["layerDeviationThreshold"] == 0.4


class TestInsertDeltaNode:
    """Tests for insert_delta_node."""

    @patch("synapse_kv_connector.segment_client.requests.Session")
    def test_request_body_structure(self, mock_session_cls: MagicMock) -> None:
        """insert_delta_node sends correctly structured JSON."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "requestId": "req-1",
            "treeSize": 1,
        }
        mock_session.post.return_value = mock_response
        mock_session_cls.return_value = mock_session

        client = SynapseSegmentClient(base_url="http://localhost:8080")
        resp = client.insert_delta_node(
            request_id="req-1",
            token_ids=[10, 20, 30],
            embedding=[0.1, 0.2, 0.3],
            kv_resident=True,
        )

        url = mock_session.post.call_args[0][0]
        body = mock_session.post.call_args[1]["json"]

        assert url == "http://localhost:8080/api/v1/kv-cache/segments/delta-tree/insert"
        assert body["requestId"] == "req-1"
        assert body["tokenIds"] == [10, 20, 30]
        assert body["embedding"] == [0.1, 0.2, 0.3]
        assert body["kvResident"] is True
        assert resp["treeSize"] == 1

    @patch("synapse_kv_connector.segment_client.requests.Session")
    def test_omits_embedding_when_none(
        self, mock_session_cls: MagicMock
    ) -> None:
        """insert_delta_node omits embedding when not provided."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "requestId": "req-2",
            "treeSize": 2,
        }
        mock_session.post.return_value = mock_response
        mock_session_cls.return_value = mock_session

        client = SynapseSegmentClient(base_url="http://localhost:8080")
        client.insert_delta_node(
            request_id="req-2",
            token_ids=[1, 2],
        )

        body = mock_session.post.call_args[1]["json"]
        assert "embedding" not in body

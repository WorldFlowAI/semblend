"""Unit tests for the token-level KV cache client."""

from __future__ import annotations

import base64
import hashlib
import json
import struct
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from synapse_kv_connector.client import (
    KvStats,
    LoadKvResult,
    SynapseKVClient,
    compute_token_hash,
)


# ---------------------------------------------------------------------------
# Token hash tests
# ---------------------------------------------------------------------------


class TestComputeTokenHash:
    """Tests for the token hash function."""

    def test_deterministic(self) -> None:
        """Same token IDs always produce the same hash."""
        ids = [1, 2, 3, 4, 5]
        assert compute_token_hash(ids) == compute_token_hash(ids)

    def test_matches_manual_sha256(self) -> None:
        """Hash matches manual SHA-256 of little-endian packed uint32s."""
        ids = [100, 200, 300]
        packed = struct.pack("<III", 100, 200, 300)
        expected = hashlib.sha256(packed).hexdigest()
        assert compute_token_hash(ids) == expected

    def test_single_token(self) -> None:
        """Single token produces a valid hex digest."""
        result = compute_token_hash([42])
        packed = struct.pack("<I", 42)
        expected = hashlib.sha256(packed).hexdigest()
        assert result == expected
        assert len(result) == 64  # SHA-256 hex digest length

    def test_empty_raises(self) -> None:
        """Empty token list raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            compute_token_hash([])

    def test_order_matters(self) -> None:
        """Different token orderings produce different hashes."""
        assert compute_token_hash([1, 2, 3]) != compute_token_hash([3, 2, 1])

    def test_large_token_ids(self) -> None:
        """Works with token IDs up to uint32 max."""
        ids = [0, 2**32 - 1, 50257]
        result = compute_token_hash(ids)
        assert isinstance(result, str)
        assert len(result) == 64


# ---------------------------------------------------------------------------
# Base64 roundtrip tests
# ---------------------------------------------------------------------------


class TestBase64Roundtrip:
    """Tests for base64 encoding/decoding of KV data."""

    def test_roundtrip_small(self) -> None:
        """Small float16 array survives base64 encode/decode."""
        original = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)
        encoded = base64.standard_b64encode(original.tobytes()).decode("ascii")
        decoded = np.frombuffer(
            base64.standard_b64decode(encoded), dtype=np.float16
        )
        np.testing.assert_array_equal(original, decoded)

    def test_roundtrip_shaped(self) -> None:
        """Shaped KV tensor survives base64 roundtrip."""
        num_layers, num_heads, num_tokens, head_dim = 2, 4, 8, 64
        shape = (num_layers, 2, num_heads, num_tokens, head_dim)
        original = np.random.randn(*shape).astype(np.float16)

        encoded = base64.standard_b64encode(original.tobytes()).decode("ascii")
        decoded = np.frombuffer(
            base64.standard_b64decode(encoded), dtype=np.float16
        ).reshape(shape)

        np.testing.assert_array_equal(original, decoded)

    def test_roundtrip_zeros(self) -> None:
        """All-zero tensor roundtrips correctly."""
        original = np.zeros((2, 2, 4, 16, 64), dtype=np.float16)
        encoded = base64.standard_b64encode(original.tobytes()).decode("ascii")
        decoded = np.frombuffer(
            base64.standard_b64decode(encoded), dtype=np.float16
        ).reshape(original.shape)
        np.testing.assert_array_equal(original, decoded)


# ---------------------------------------------------------------------------
# Request body construction tests
# ---------------------------------------------------------------------------


class TestStoreKvStateRequest:
    """Tests for store_kv_state request construction."""

    @patch("synapse_kv_connector.client.requests.Session")
    def test_request_body_structure(self, mock_session_cls: MagicMock) -> None:
        """store_kv_state sends correctly structured JSON."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "tokenHash": "abc",
            "bytesStored": 64,
            "overwroteExisting": False,
        }
        mock_session.put.return_value = mock_response
        mock_session_cls.return_value = mock_session

        client = SynapseKVClient(
            base_url="http://localhost:8080",
            tenant_id="test-tenant",
        )

        token_ids = [10, 20, 30]
        kv_data = np.ones((2, 2, 4, 3, 64), dtype=np.float16)

        client.store_kv_state(
            token_ids=token_ids,
            kv_data=kv_data,
            num_layers=2,
            num_heads=4,
            head_dim=64,
        )

        call_args = mock_session.put.call_args
        url = call_args[0][0]
        body = call_args[1]["json"]

        expected_hash = compute_token_hash(token_ids)
        assert url == f"http://localhost:8080/api/v1/kv-cache/{expected_hash}"
        assert body["numTokens"] == 3
        assert body["numLayers"] == 2
        assert body["numHeads"] == 4
        assert body["headDim"] == 64
        assert body["tenantId"] == "test-tenant"
        assert isinstance(body["kvData"], str)

        # Verify the base64 data decodes correctly
        decoded = base64.standard_b64decode(body["kvData"])
        restored = np.frombuffer(decoded, dtype=np.float16)
        np.testing.assert_array_equal(restored, kv_data.ravel())

    @patch("synapse_kv_connector.client.requests.Session")
    def test_rejects_non_float16(self, mock_session_cls: MagicMock) -> None:
        """store_kv_state rejects non-float16 arrays."""
        mock_session_cls.return_value = MagicMock()
        client = SynapseKVClient(base_url="http://localhost:8080")

        kv_data = np.ones((2, 2, 4, 3, 64), dtype=np.float32)
        with pytest.raises(ValueError, match="float16"):
            client.store_kv_state(
                token_ids=[1, 2, 3],
                kv_data=kv_data,
                num_layers=2,
                num_heads=4,
                head_dim=64,
            )


# ---------------------------------------------------------------------------
# Response parsing tests
# ---------------------------------------------------------------------------


class TestLoadKvStateResponse:
    """Tests for load_kv_state response parsing."""

    @patch("synapse_kv_connector.client.requests.Session")
    def test_parses_response(self, mock_session_cls: MagicMock) -> None:
        """load_kv_state correctly parses the gateway response."""
        num_layers, num_heads, num_tokens, head_dim = 2, 4, 3, 64
        shape = (num_layers, 2, num_heads, num_tokens, head_dim)
        original = np.random.randn(*shape).astype(np.float16)
        kv_b64 = base64.standard_b64encode(original.tobytes()).decode("ascii")

        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "numTokens": num_tokens,
            "numLayers": num_layers,
            "numHeads": num_heads,
            "headDim": head_dim,
            "kvData": kv_b64,
        }
        mock_session.get.return_value = mock_response
        mock_session_cls.return_value = mock_session

        client = SynapseKVClient(base_url="http://localhost:8080")
        result = client.load_kv_state([10, 20, 30])

        assert result is not None
        assert isinstance(result, LoadKvResult)
        assert result.num_tokens == num_tokens
        assert result.num_layers == num_layers
        assert result.num_heads == num_heads
        assert result.head_dim == head_dim
        assert result.kv_data.shape == shape
        assert result.kv_data.dtype == np.float16
        np.testing.assert_array_equal(result.kv_data, original)

    @patch("synapse_kv_connector.client.requests.Session")
    def test_returns_none_on_404(self, mock_session_cls: MagicMock) -> None:
        """load_kv_state returns None when the entry is not found."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_session.get.return_value = mock_response
        mock_session_cls.return_value = mock_session

        client = SynapseKVClient(base_url="http://localhost:8080")
        result = client.load_kv_state([99, 100])

        assert result is None


# ---------------------------------------------------------------------------
# Delete tests
# ---------------------------------------------------------------------------


class TestDeleteKvState:
    """Tests for delete_kv_state."""

    @patch("synapse_kv_connector.client.requests.Session")
    def test_returns_true_on_success(
        self, mock_session_cls: MagicMock
    ) -> None:
        """delete_kv_state returns True on 204 No Content."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_session.delete.return_value = mock_response
        mock_session_cls.return_value = mock_session

        client = SynapseKVClient(base_url="http://localhost:8080")
        assert client.delete_kv_state([1, 2, 3]) is True

    @patch("synapse_kv_connector.client.requests.Session")
    def test_returns_false_on_404(
        self, mock_session_cls: MagicMock
    ) -> None:
        """delete_kv_state returns False when entry not found."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_session.delete.return_value = mock_response
        mock_session_cls.return_value = mock_session

        client = SynapseKVClient(base_url="http://localhost:8080")
        assert client.delete_kv_state([1, 2, 3]) is False


# ---------------------------------------------------------------------------
# Stats tests
# ---------------------------------------------------------------------------


class TestGetStats:
    """Tests for get_stats response parsing."""

    @patch("synapse_kv_connector.client.requests.Session")
    def test_parses_stats_response(
        self, mock_session_cls: MagicMock
    ) -> None:
        """get_stats correctly parses the gateway response."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "entryCount": 42,
            "totalBytes": 1048576,
            "maxEntries": 65536,
            "fillRatio": 0.000641,
        }
        mock_session.get.return_value = mock_response
        mock_session_cls.return_value = mock_session

        client = SynapseKVClient(base_url="http://localhost:8080")
        stats = client.get_stats()

        assert isinstance(stats, KvStats)
        assert stats.entry_count == 42
        assert stats.total_bytes == 1048576
        assert stats.max_entries == 65536
        assert abs(stats.fill_ratio - 0.000641) < 1e-9


# ---------------------------------------------------------------------------
# load_kv_state_by_hash tests
# ---------------------------------------------------------------------------


class TestLoadKvStateByHash:
    """Tests for load_kv_state_by_hash (direct hash lookup)."""

    @patch("synapse_kv_connector.client.requests.Session")
    def test_parses_response(self, mock_session_cls: MagicMock) -> None:
        """Correctly parses KV data from a direct hash lookup."""
        num_layers, num_heads, num_tokens, head_dim = 2, 4, 5, 64
        shape = (num_layers, 2, num_heads, num_tokens, head_dim)
        original = np.random.randn(*shape).astype(np.float16)
        kv_b64 = base64.standard_b64encode(original.tobytes()).decode(
            "ascii"
        )

        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "numTokens": num_tokens,
            "numLayers": num_layers,
            "numHeads": num_heads,
            "headDim": head_dim,
            "kvData": kv_b64,
        }
        mock_session.get.return_value = mock_response
        mock_session_cls.return_value = mock_session

        client = SynapseKVClient(base_url="http://localhost:8080")
        result = client.load_kv_state_by_hash("abc123def456")

        assert result is not None
        assert isinstance(result, LoadKvResult)
        assert result.num_tokens == num_tokens
        assert result.kv_data.shape == shape
        np.testing.assert_array_equal(result.kv_data, original)

        # Verify the URL uses the hash directly (no recomputation)
        call_url = mock_session.get.call_args[0][0]
        assert call_url.endswith("/api/v1/kv-cache/abc123def456")

    @patch("synapse_kv_connector.client.requests.Session")
    def test_returns_none_on_404(self, mock_session_cls: MagicMock) -> None:
        """Returns None when the hash is not found."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_session.get.return_value = mock_response
        mock_session_cls.return_value = mock_session

        client = SynapseKVClient(base_url="http://localhost:8080")
        result = client.load_kv_state_by_hash("nonexistent")

        assert result is None

    @patch("synapse_kv_connector.client.requests.Session")
    def test_does_not_compute_hash(
        self, mock_session_cls: MagicMock
    ) -> None:
        """Verifies the hash is used as-is without recomputation."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_session.get.return_value = mock_response
        mock_session_cls.return_value = mock_session

        client = SynapseKVClient(base_url="http://localhost:8080")
        client.load_kv_state_by_hash("my-custom-hash-value")

        call_url = mock_session.get.call_args[0][0]
        assert "my-custom-hash-value" in call_url

"""Unit tests for the SynapseKVConnector with Tier 3 integration."""

from __future__ import annotations

import base64
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np

from synapse_kv_connector.client import LoadKvResult
from synapse_kv_connector.connector import SynapseKVConnector
from synapse_kv_connector.partial_attention import AttentionMode
from synapse_kv_connector.segment_client import (
    FindDonorResult,
    KvSlotAction,
    KvTransferPlan,
)


def _make_connector(tier3_enabled: bool = True) -> SynapseKVConnector:
    """Create a connector with mocked HTTP sessions."""
    with patch("requests.Session"):
        return SynapseKVConnector(
            base_url="http://localhost:8080",
            tier3_enabled=tier3_enabled,
            min_reuse_ratio=0.5,
        )


def _mock_tier1_hit(connector: SynapseKVConnector, num_tokens: int) -> None:
    """Mock the token client to return a Tier 1 hit."""
    kv = np.ones((2, 2, 4, num_tokens, 64), dtype=np.float16)
    result = LoadKvResult(
        num_tokens=num_tokens,
        num_layers=2,
        num_heads=4,
        head_dim=64,
        kv_data=kv,
    )
    connector._token_client.load_kv_state = MagicMock(return_value=result)


def _mock_tier1_miss(connector: SynapseKVConnector) -> None:
    """Mock the token client to return a Tier 1 miss."""
    connector._token_client.load_kv_state = MagicMock(return_value=None)


def _mock_tier3_found(
    connector: SynapseKVConnector,
    donor_id: str = "donor-1",
    reuse_ratio: float = 0.8,
    num_copied: int = 4,
    num_placeholders: int = 1,
    viable: bool = True,
) -> None:
    """Mock the segment client to return a Tier 3 donor match."""
    plan = KvTransferPlan(
        donor_id=donor_id,
        target_len=num_copied + num_placeholders,
        donor_len=num_copied + num_placeholders,
        slot_actions=[],
        copy_positions=list(range(num_copied)),
        placeholder_positions=list(
            range(num_copied, num_copied + num_placeholders)
        ),
        num_copied=num_copied,
        num_placeholders=num_placeholders,
        reuse_ratio=reuse_ratio,
        viable=viable,
    )
    result = FindDonorResult(found=True, plan=plan, tree_size=3)
    connector._segment_client.find_donor_and_plan = MagicMock(
        return_value=result
    )


def _mock_tier3_not_found(connector: SynapseKVConnector) -> None:
    """Mock the segment client to return no donor."""
    result = FindDonorResult(found=False, plan=None, tree_size=0)
    connector._segment_client.find_donor_and_plan = MagicMock(
        return_value=result
    )


# ---------------------------------------------------------------------------
# Tier 1 exact match (existing behavior)
# ---------------------------------------------------------------------------


class TestTier1ExactMatch:
    """Tests for Tier 1 exact token-hash matching."""

    def test_exact_match_returns_num_tokens(self) -> None:
        """Tier 1 exact hit returns the full token count."""
        connector = _make_connector(tier3_enabled=False)
        _mock_tier1_hit(connector, num_tokens=10)

        matched = connector.get_num_new_matched_tokens([1, 2, 3])

        assert matched == 10
        assert connector.last_transfer_plan is None

    def test_no_match_returns_zero(self) -> None:
        """No match returns 0."""
        connector = _make_connector(tier3_enabled=False)
        _mock_tier1_miss(connector)

        assert connector.get_num_new_matched_tokens([1, 2, 3]) == 0

    def test_empty_tokens_returns_zero(self) -> None:
        """Empty token list returns 0."""
        connector = _make_connector()
        assert connector.get_num_new_matched_tokens([]) == 0


# ---------------------------------------------------------------------------
# Tier 3 approximate match
# ---------------------------------------------------------------------------


class TestTier3ApproximateMatch:
    """Tests for Tier 3 DELTA tree donor matching."""

    def test_tier3_fallback_on_tier1_miss(self) -> None:
        """Falls back to Tier 3 when Tier 1 misses."""
        connector = _make_connector()
        _mock_tier1_miss(connector)
        _mock_tier3_found(connector, num_copied=4, num_placeholders=1)

        matched = connector.get_num_new_matched_tokens([1, 2, 3, 99, 5])

        assert matched == 4
        assert connector.last_transfer_plan is not None
        assert connector.last_transfer_plan.donor_id == "donor-1"
        assert connector.last_transfer_plan.reuse_ratio == 0.8

    def test_tier3_not_viable(self) -> None:
        """Returns 0 when Tier 3 plan is not viable."""
        connector = _make_connector()
        _mock_tier1_miss(connector)
        _mock_tier3_found(
            connector,
            num_copied=1,
            num_placeholders=9,
            reuse_ratio=0.1,
            viable=False,
        )

        matched = connector.get_num_new_matched_tokens(list(range(10)))

        assert matched == 0
        assert connector.last_transfer_plan is None

    def test_tier3_no_donor_found(self) -> None:
        """Returns 0 when no donor exists in DELTA tree."""
        connector = _make_connector()
        _mock_tier1_miss(connector)
        _mock_tier3_not_found(connector)

        assert connector.get_num_new_matched_tokens([1, 2, 3]) == 0

    def test_tier1_hit_skips_tier3(self) -> None:
        """Tier 1 hit bypasses Tier 3 entirely."""
        connector = _make_connector()
        _mock_tier1_hit(connector, num_tokens=5)

        # Add a mock to verify it's NOT called
        connector._segment_client.find_donor_and_plan = MagicMock()

        matched = connector.get_num_new_matched_tokens([1, 2, 3, 4, 5])

        assert matched == 5
        connector._segment_client.find_donor_and_plan.assert_not_called()

    def test_tier3_disabled(self) -> None:
        """Tier 3 is skipped when disabled."""
        connector = _make_connector(tier3_enabled=False)
        _mock_tier1_miss(connector)
        connector._segment_client.find_donor_and_plan = MagicMock()

        matched = connector.get_num_new_matched_tokens([1, 2, 3])

        assert matched == 0
        connector._segment_client.find_donor_and_plan.assert_not_called()

    def test_tier3_error_is_caught(self) -> None:
        """Tier 3 errors don't crash the connector."""
        connector = _make_connector()
        _mock_tier1_miss(connector)
        connector._segment_client.find_donor_and_plan = MagicMock(
            side_effect=Exception("network error")
        )

        assert connector.get_num_new_matched_tokens([1, 2, 3]) == 0

    def test_plan_cleared_between_calls(self) -> None:
        """Transfer plan is cleared at start of each scheduling call."""
        connector = _make_connector()
        _mock_tier1_miss(connector)
        _mock_tier3_found(connector, num_copied=3, num_placeholders=1)

        connector.get_num_new_matched_tokens([1, 2, 3, 4])
        assert connector.last_transfer_plan is not None

        # Next call with Tier 1 hit should clear the plan
        _mock_tier1_hit(connector, num_tokens=5)
        connector.get_num_new_matched_tokens([1, 2, 3, 4, 5])
        assert connector.last_transfer_plan is None


# ---------------------------------------------------------------------------
# Save + DELTA tree registration
# ---------------------------------------------------------------------------


class TestSaveKvWithDeltaTree:
    """Tests for start_save_kv with DELTA tree registration."""

    def test_save_registers_in_delta_tree(self) -> None:
        """start_save_kv stores KV and registers in DELTA tree."""
        connector = _make_connector()
        connector._token_client.store_kv_state = MagicMock(
            return_value={
                "tokenHash": "abc123",
                "bytesStored": 512,
                "overwroteExisting": False,
            }
        )
        connector._segment_client.insert_delta_node = MagicMock(
            return_value={"requestId": "abc123", "treeSize": 1}
        )

        kv_data = np.ones((2, 2, 4, 5, 64), dtype=np.float16)
        connector.start_save_kv(
            token_ids=[1, 2, 3, 4, 5],
            kv_data=kv_data,
            num_layers=2,
            num_heads=4,
            head_dim=64,
        )

        connector._token_client.store_kv_state.assert_called_once()
        connector._segment_client.insert_delta_node.assert_called_once()

        call_kwargs = (
            connector._segment_client.insert_delta_node.call_args[1]
        )
        assert call_kwargs["request_id"] == "abc123"
        assert call_kwargs["token_ids"] == [1, 2, 3, 4, 5]
        assert call_kwargs["kv_resident"] is True

    def test_save_uses_custom_request_id(self) -> None:
        """start_save_kv uses custom request_id for DELTA tree."""
        connector = _make_connector()
        connector._token_client.store_kv_state = MagicMock(
            return_value={
                "tokenHash": "hash",
                "bytesStored": 64,
                "overwroteExisting": False,
            }
        )
        connector._segment_client.insert_delta_node = MagicMock(
            return_value={"requestId": "custom-id", "treeSize": 2}
        )

        kv_data = np.ones((1, 2, 2, 3, 32), dtype=np.float16)
        connector.start_save_kv(
            token_ids=[1, 2, 3],
            kv_data=kv_data,
            num_layers=1,
            num_heads=2,
            head_dim=32,
            request_id="custom-id",
        )

        call_kwargs = (
            connector._segment_client.insert_delta_node.call_args[1]
        )
        assert call_kwargs["request_id"] == "custom-id"

    def test_save_skips_delta_tree_when_tier3_disabled(self) -> None:
        """start_save_kv skips DELTA tree when Tier 3 is off."""
        connector = _make_connector(tier3_enabled=False)
        connector._token_client.store_kv_state = MagicMock(
            return_value={
                "tokenHash": "h",
                "bytesStored": 32,
                "overwroteExisting": False,
            }
        )
        connector._segment_client.insert_delta_node = MagicMock()

        kv_data = np.ones((1, 2, 2, 3, 32), dtype=np.float16)
        connector.start_save_kv(
            token_ids=[1, 2, 3],
            kv_data=kv_data,
            num_layers=1,
            num_heads=2,
            head_dim=32,
        )

        connector._token_client.store_kv_state.assert_called_once()
        connector._segment_client.insert_delta_node.assert_not_called()

    def test_save_delta_tree_error_caught(self) -> None:
        """DELTA tree registration errors don't crash save."""
        connector = _make_connector()
        connector._token_client.store_kv_state = MagicMock(
            return_value={
                "tokenHash": "h",
                "bytesStored": 32,
                "overwroteExisting": False,
            }
        )
        connector._segment_client.insert_delta_node = MagicMock(
            side_effect=Exception("delta tree error")
        )

        kv_data = np.ones((1, 2, 2, 3, 32), dtype=np.float16)
        # Should not raise
        connector.start_save_kv(
            token_ids=[1, 2, 3],
            kv_data=kv_data,
            num_layers=1,
            num_heads=2,
            head_dim=32,
        )

    def test_save_empty_tokens_is_noop(self) -> None:
        """start_save_kv with empty tokens does nothing."""
        connector = _make_connector()
        connector._token_client.store_kv_state = MagicMock()
        connector._segment_client.insert_delta_node = MagicMock()

        connector.start_save_kv(
            token_ids=[],
            kv_data=np.ones((1,), dtype=np.float16),
            num_layers=1,
            num_heads=1,
            head_dim=1,
        )

        connector._token_client.store_kv_state.assert_not_called()
        connector._segment_client.insert_delta_node.assert_not_called()


# ---------------------------------------------------------------------------
# Tier 3 start_load_kv
# ---------------------------------------------------------------------------


class TestTier3StartLoadKv:
    """Tests for start_load_kv Tier 3 path (donor KV loading)."""

    def test_loads_donor_kv_with_contiguous_prefix(self) -> None:
        """Tier 3 path loads donor KV when copy positions are contiguous."""
        connector = _make_connector()
        _mock_tier1_miss(connector)
        _mock_tier3_found(
            connector,
            donor_id="donor-abc",
            num_copied=4,
            num_placeholders=1,
        )

        # Schedule to populate the transfer plan
        connector.get_num_new_matched_tokens([1, 2, 3, 4, 99])
        assert connector.last_transfer_plan is not None

        # Mock the donor KV lookup
        donor_kv = np.ones((2, 2, 4, 5, 64), dtype=np.float16)
        donor_result = LoadKvResult(
            num_tokens=5,
            num_layers=2,
            num_heads=4,
            head_dim=64,
            kv_data=donor_kv,
        )
        connector._token_client.load_kv_state_by_hash = MagicMock(
            return_value=donor_result
        )

        result = connector.start_load_kv([1, 2, 3, 4, 99])

        assert result is not None
        assert result.shape == (2, 2, 4, 5, 64)
        assert result.dtype == np.float16
        # Copied positions [0,1,2,3] should have donor data (all ones)
        np.testing.assert_array_equal(result[:, :, :, 0, :], 1.0)
        np.testing.assert_array_equal(result[:, :, :, 3, :], 1.0)
        # Placeholder position [4] should be zeros
        np.testing.assert_array_equal(result[:, :, :, 4, :], 0.0)
        # Verify load_kv_state_by_hash was called with the donor hash
        connector._token_client.load_kv_state_by_hash.assert_called_once_with(
            "donor-abc"
        )

    def test_consumes_transfer_plan(self) -> None:
        """Transfer plan is consumed (cleared) after use."""
        connector = _make_connector()
        _mock_tier1_miss(connector)
        _mock_tier3_found(connector, num_copied=3, num_placeholders=1)

        connector.get_num_new_matched_tokens([1, 2, 3, 4])
        assert connector.last_transfer_plan is not None

        # Mock donor not found to test plan consumption
        connector._token_client.load_kv_state_by_hash = MagicMock(
            return_value=None
        )
        connector.start_load_kv([1, 2, 3, 4])

        # Plan should be consumed (None)
        assert connector.last_transfer_plan is None

    def test_returns_none_when_donor_not_found(self) -> None:
        """Returns None when donor KV is not in the cache."""
        connector = _make_connector()
        _mock_tier1_miss(connector)
        _mock_tier3_found(connector, num_copied=3, num_placeholders=1)

        connector.get_num_new_matched_tokens([1, 2, 3, 4])

        connector._token_client.load_kv_state_by_hash = MagicMock(
            return_value=None
        )
        result = connector.start_load_kv([1, 2, 3, 4])

        assert result is None

    def test_sparse_copy_with_partial_attention(self) -> None:
        """Non-contiguous positions use PartialAttention instead of fallback."""
        connector = _make_connector()
        _mock_tier1_miss(connector)

        # Create a plan with non-contiguous copy positions [0, 2, 4]
        plan = KvTransferPlan(
            donor_id="donor-sparse",
            target_len=5,
            donor_len=5,
            slot_actions=[
                KvSlotAction("copy_from_donor", donor_pos=0, target_pos=0),
                KvSlotAction("placeholder", donor_pos=None, target_pos=1),
                KvSlotAction("copy_from_donor", donor_pos=2, target_pos=2),
                KvSlotAction("placeholder", donor_pos=None, target_pos=3),
                KvSlotAction("copy_from_donor", donor_pos=4, target_pos=4),
            ],
            copy_positions=[0, 2, 4],
            placeholder_positions=[1, 3],
            num_copied=3,
            num_placeholders=2,
            reuse_ratio=0.6,
            viable=True,
        )
        result = FindDonorResult(found=True, plan=plan, tree_size=1)
        connector._segment_client.find_donor_and_plan = MagicMock(
            return_value=result
        )

        connector.get_num_new_matched_tokens([1, 2, 3, 4, 5])

        # Mock donor KV
        donor_kv = np.ones((2, 2, 4, 5, 64), dtype=np.float16)
        donor_result = LoadKvResult(
            num_tokens=5,
            num_layers=2,
            num_heads=4,
            head_dim=64,
            kv_data=donor_kv,
        )
        connector._token_client.load_kv_state_by_hash = MagicMock(
            return_value=donor_result
        )

        kv = connector.start_load_kv([1, 2, 3, 4, 5])

        # Should succeed with sparse KV copy
        assert kv is not None
        assert kv.shape == (2, 2, 4, 5, 64)

        # Copied positions [0, 2, 4] should have donor data (ones)
        np.testing.assert_array_equal(kv[:, :, :, 0, :], 1.0)
        np.testing.assert_array_equal(kv[:, :, :, 2, :], 1.0)
        np.testing.assert_array_equal(kv[:, :, :, 4, :], 1.0)

        # Placeholder positions [1, 3] should be zeros
        np.testing.assert_array_equal(kv[:, :, :, 1, :], 0.0)
        np.testing.assert_array_equal(kv[:, :, :, 3, :], 0.0)

        # PartialAttention plan should be populated
        attn_plan = connector.last_attention_plan
        assert attn_plan is not None
        assert attn_plan.donor_id == "donor-sparse"
        assert attn_plan.num_reuse_positions == 3
        assert attn_plan.num_partial_positions == 2
        assert attn_plan.computation_ratio < 1.0

    def test_contiguous_prefix_no_attention_plan(self) -> None:
        """Contiguous prefix copy does not produce an attention plan."""
        connector = _make_connector()
        _mock_tier1_miss(connector)
        _mock_tier3_found(
            connector,
            donor_id="donor-contig",
            num_copied=4,
            num_placeholders=1,
        )

        connector.get_num_new_matched_tokens([1, 2, 3, 4, 99])

        donor_kv = np.ones((2, 2, 4, 5, 64), dtype=np.float16)
        donor_result = LoadKvResult(
            num_tokens=5, num_layers=2, num_heads=4,
            head_dim=64, kv_data=donor_kv,
        )
        connector._token_client.load_kv_state_by_hash = MagicMock(
            return_value=donor_result
        )

        kv = connector.start_load_kv([1, 2, 3, 4, 99])

        assert kv is not None
        # Contiguous prefix: no attention plan needed
        assert connector.last_attention_plan is None

    def test_sparse_copy_fallback_without_slot_actions(self) -> None:
        """Non-contiguous copy with empty slot_actions uses 1:1 mapping."""
        connector = _make_connector()
        _mock_tier1_miss(connector)

        # Plan with non-contiguous positions but no slot_actions
        plan = KvTransferPlan(
            donor_id="donor-sparse2",
            target_len=5,
            donor_len=5,
            slot_actions=[],
            copy_positions=[0, 2, 4],
            placeholder_positions=[1, 3],
            num_copied=3,
            num_placeholders=2,
            reuse_ratio=0.6,
            viable=True,
        )
        result = FindDonorResult(found=True, plan=plan, tree_size=1)
        connector._segment_client.find_donor_and_plan = MagicMock(
            return_value=result
        )

        connector.get_num_new_matched_tokens([1, 2, 3, 4, 5])

        donor_kv = np.ones((2, 2, 4, 5, 64), dtype=np.float16)
        donor_result = LoadKvResult(
            num_tokens=5, num_layers=2, num_heads=4,
            head_dim=64, kv_data=donor_kv,
        )
        connector._token_client.load_kv_state_by_hash = MagicMock(
            return_value=donor_result
        )

        kv = connector.start_load_kv([1, 2, 3, 4, 5])

        assert kv is not None
        # 1:1 fallback: positions 0,2,4 get donor data
        np.testing.assert_array_equal(kv[:, :, :, 0, :], 1.0)
        np.testing.assert_array_equal(kv[:, :, :, 2, :], 1.0)
        assert connector.last_attention_plan is not None

    def test_attention_plan_cleared_between_loads(self) -> None:
        """Attention plan is cleared at the start of each start_load_kv."""
        connector = _make_connector()
        _mock_tier1_miss(connector)

        # First: sparse copy produces attention plan
        plan = KvTransferPlan(
            donor_id="d",
            target_len=3,
            donor_len=3,
            slot_actions=[
                KvSlotAction("copy_from_donor", donor_pos=0, target_pos=0),
                KvSlotAction("placeholder", donor_pos=None, target_pos=1),
                KvSlotAction("copy_from_donor", donor_pos=2, target_pos=2),
            ],
            copy_positions=[0, 2],
            placeholder_positions=[1],
            num_copied=2,
            num_placeholders=1,
            reuse_ratio=0.67,
            viable=True,
        )
        result = FindDonorResult(found=True, plan=plan, tree_size=1)
        connector._segment_client.find_donor_and_plan = MagicMock(
            return_value=result
        )
        connector.get_num_new_matched_tokens([1, 2, 3])

        donor_kv = np.ones((2, 2, 4, 3, 64), dtype=np.float16)
        connector._token_client.load_kv_state_by_hash = MagicMock(
            return_value=LoadKvResult(
                num_tokens=3, num_layers=2, num_heads=4,
                head_dim=64, kv_data=donor_kv,
            )
        )
        connector.start_load_kv([1, 2, 3])
        assert connector.last_attention_plan is not None

        # Second: Tier 1 load should clear the attention plan
        connector._token_client.load_kv_state = MagicMock(
            return_value=LoadKvResult(
                num_tokens=3, num_layers=2, num_heads=4,
                head_dim=64, kv_data=donor_kv,
            )
        )
        connector.start_load_kv([1, 2, 3])
        assert connector.last_attention_plan is None

    def test_error_during_load_returns_none(self) -> None:
        """Errors during donor KV loading return None gracefully."""
        connector = _make_connector()
        _mock_tier1_miss(connector)
        _mock_tier3_found(connector, num_copied=3, num_placeholders=1)

        connector.get_num_new_matched_tokens([1, 2, 3, 4])

        connector._token_client.load_kv_state_by_hash = MagicMock(
            side_effect=Exception("network timeout")
        )
        result = connector.start_load_kv([1, 2, 3, 4])

        assert result is None

    def test_tier1_fallback_when_no_plan(self) -> None:
        """Falls back to Tier 1 exact match when no transfer plan."""
        connector = _make_connector()
        kv = np.ones((2, 2, 4, 5, 64), dtype=np.float16)
        result = LoadKvResult(
            num_tokens=5,
            num_layers=2,
            num_heads=4,
            head_dim=64,
            kv_data=kv,
        )
        connector._token_client.load_kv_state = MagicMock(
            return_value=result
        )

        loaded = connector.start_load_kv([1, 2, 3, 4, 5])

        assert loaded is not None
        np.testing.assert_array_equal(loaded, kv)

# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFlow AI. All rights reserved.
# SPDX-License-Identifier: LicenseRef-WorldFlowAI-Proprietary

"""Tests for the CPU KVCacheConnector."""

import numpy as np
import pytest

from synapse_kv_connector.connector_protocol import KVCacheConnector, KVMetadata
from synapse_kv_connector.backends.cpu import CPUKVCacheConnector


class TestProtocolCompliance:
    """Verify CPUKVCacheConnector satisfies the Protocol."""

    def test_implements_protocol(self):
        connector = CPUKVCacheConnector()
        assert isinstance(connector, KVCacheConnector)

    def test_metadata(self):
        connector = CPUKVCacheConnector(num_layers=12, num_kv_heads=2, head_dim=64)
        meta = connector.get_metadata()
        assert isinstance(meta, KVMetadata)
        assert meta.num_layers == 12
        assert meta.num_kv_heads == 2
        assert meta.head_dim == 64
        assert meta.device == "cpu"


class TestLoadDonorKV:
    """Test KV loading operations."""

    @pytest.fixture
    def connector(self):
        return CPUKVCacheConnector(
            num_layers=4, num_kv_heads=2, head_dim=8,
            block_size=4, max_seq_len=64,
        )

    def test_basic_load(self, connector):
        """Load donor KV and verify it's in the cache."""
        # Create fake donor KV: [layers=4, kv=2, heads=2, seq=8, dim=8]
        donor = np.random.randn(4, 2, 2, 8, 8).astype(np.float16)

        result = connector.load_donor_kv(
            donor_kv=donor,
            donor_positions=[0, 1, 2, 3],
            target_positions=[0, 1, 2, 3],  # Same positions = no RoPE needed
        )

        assert result.tokens_loaded == 4
        assert len(result.layers_loaded) == 4
        assert result.load_time_ms > 0

    def test_load_subset_layers(self, connector):
        """Load only specific layers."""
        donor = np.random.randn(4, 2, 2, 8, 8).astype(np.float16)

        result = connector.load_donor_kv(
            donor_kv=donor,
            donor_positions=[0, 1],
            target_positions=[0, 1],
            layers=[0, 2],  # Only layers 0 and 2
        )

        assert result.layers_loaded == [0, 2]

    def test_load_empty_positions(self, connector):
        """Empty positions should work without error."""
        donor = np.random.randn(4, 2, 2, 8, 8).astype(np.float16)

        result = connector.load_donor_kv(
            donor_kv=donor,
            donor_positions=[],
            target_positions=[],
        )

        assert result.tokens_loaded == 0


class TestExtractKV:
    """Test KV extraction."""

    def test_extract_returns_copy(self):
        connector = CPUKVCacheConnector(
            num_layers=2, num_kv_heads=2, head_dim=4,
            max_seq_len=16,
        )

        # Load some data first
        donor = np.ones((2, 2, 2, 16, 4), dtype=np.float16)
        connector.load_donor_kv(donor, list(range(8)), list(range(8)))

        # Extract
        extracted = connector.extract_kv("req-1", (0, 8))
        assert extracted.shape == (2, 2, 2, 8, 4)

        # Verify it's a copy (modifying extracted doesn't affect cache)
        extracted[:] = 999
        re_extracted = connector.extract_kv("req-1", (0, 8))
        assert not np.all(re_extracted == 999)


class TestRoPECorrection:
    """Test RoPE position correction."""

    def test_identity_correction(self):
        """Same source and target positions = no change."""
        connector = CPUKVCacheConnector(head_dim=8)
        k = np.random.randn(2, 4, 8).astype(np.float32)  # [heads, seq, dim]
        k_original = k.copy()

        result = connector.apply_rope_correction(
            k, source_positions=[0, 1, 2, 3], target_positions=[0, 1, 2, 3]
        )

        np.testing.assert_allclose(result, k_original, atol=1e-6)

    def test_nonzero_correction(self):
        """Different positions should change K."""
        connector = CPUKVCacheConnector(head_dim=8)
        k = np.random.randn(2, 4, 8).astype(np.float32)
        k_original = k.copy()

        result = connector.apply_rope_correction(
            k, source_positions=[0, 1, 2, 3], target_positions=[10, 11, 12, 13]
        )

        # Should be different from original
        assert not np.allclose(result, k_original)

    def test_correction_preserves_norm(self):
        """RoPE rotation should preserve vector norms (it's a rotation)."""
        connector = CPUKVCacheConnector(head_dim=8)
        k = np.random.randn(2, 4, 8).astype(np.float32)

        norms_before = np.linalg.norm(k, axis=-1)

        result = connector.apply_rope_correction(
            k, source_positions=[0, 1, 2, 3], target_positions=[100, 101, 102, 103]
        )

        norms_after = np.linalg.norm(result, axis=-1)
        np.testing.assert_allclose(norms_after, norms_before, atol=1e-5)

    def test_correction_is_reversible(self):
        """Applying correction and then reverse should get back original."""
        connector = CPUKVCacheConnector(head_dim=8)
        k = np.random.randn(2, 4, 8).astype(np.float32)
        k_original = k.copy()

        # Forward: 0→10
        k_fwd = connector.apply_rope_correction(
            k.copy(), source_positions=[0, 1, 2, 3], target_positions=[10, 11, 12, 13]
        )

        # Reverse: 10→0
        k_rev = connector.apply_rope_correction(
            k_fwd, source_positions=[10, 11, 12, 13], target_positions=[0, 1, 2, 3]
        )

        np.testing.assert_allclose(k_rev, k_original, atol=1e-5)

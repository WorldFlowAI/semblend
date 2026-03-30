# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFlow AI. All rights reserved.
# SPDX-License-Identifier: LicenseRef-WorldFlowAI-Proprietary

"""Tests for SemBlendRadixCache components (CPU-only, no SGLang required)."""

import numpy as np
import pytest

from synapse_kv_connector.backends.sglang_cpu import (
    _DonorRecord,
    _LocalDonorStore,
    rope_correct_cpu,
)


class TestLocalDonorStore:
    """Test the local semantic donor index."""

    def test_add_and_search(self):
        store = _LocalDonorStore(min_similarity=0.5)
        emb = np.random.randn(384).astype(np.float32)
        emb /= np.linalg.norm(emb)

        store.add(_DonorRecord(
            donor_id="d1", embedding=emb, token_ids=[1, 2, 3],
            num_tokens=3, registered_at=0.0,
        ))

        result = store.search(emb)  # Same embedding = perfect match
        assert result is not None
        assert result.donor_id == "d1"

    def test_empty_store(self):
        store = _LocalDonorStore()
        emb = np.random.randn(384).astype(np.float32)
        assert store.search(emb) is None

    def test_below_threshold(self):
        store = _LocalDonorStore(min_similarity=0.99)
        emb1 = np.zeros(384, dtype=np.float32)
        emb1[0] = 1.0
        emb2 = np.zeros(384, dtype=np.float32)
        emb2[1] = 1.0  # Orthogonal

        store.add(_DonorRecord("d1", emb1, [1], num_tokens=1, registered_at=0.0))
        assert store.search(emb2) is None

    def test_max_donors_eviction(self):
        store = _LocalDonorStore(max_donors=3)
        for i in range(5):
            emb = np.random.randn(384).astype(np.float32)
            store.add(_DonorRecord(
                f"d{i}", emb, [i], num_tokens=1, registered_at=float(i),
            ))
        assert store.size == 3

    def test_remove(self):
        store = _LocalDonorStore()
        emb = np.random.randn(384).astype(np.float32)
        store.add(_DonorRecord("d1", emb, [1], num_tokens=1, registered_at=0.0))
        assert store.size == 1
        store.remove("d1")
        assert store.size == 0

    def test_best_match_returned(self):
        store = _LocalDonorStore(min_similarity=0.3)
        base = np.random.randn(384).astype(np.float32)
        base /= np.linalg.norm(base)

        # Add a close match and a distant one
        close = base + np.random.randn(384).astype(np.float32) * 0.1
        close /= np.linalg.norm(close)
        distant = np.random.randn(384).astype(np.float32)
        distant /= np.linalg.norm(distant)

        store.add(_DonorRecord("close", close, [1], num_tokens=1, registered_at=0.0))
        store.add(_DonorRecord("distant", distant, [2], num_tokens=1, registered_at=1.0))

        result = store.search(base)
        assert result is not None
        assert result.donor_id == "close"


class TestRoPECorrection:
    """Test the RoPE correction function."""

    def test_identity(self):
        k = np.random.randn(2, 4, 8).astype(np.float32)
        k_orig = k.copy()
        result = rope_correct_cpu(k, np.array([0, 1, 2, 3]), np.array([0, 1, 2, 3]), 8)
        np.testing.assert_allclose(result, k_orig, atol=1e-6)

    def test_nonzero_delta(self):
        k = np.random.randn(2, 4, 8).astype(np.float32)
        k_orig = k.copy()
        result = rope_correct_cpu(k, np.array([0, 1, 2, 3]), np.array([10, 11, 12, 13]), 8)
        assert not np.allclose(result, k_orig)

    def test_preserves_norm(self):
        k = np.random.randn(2, 4, 8).astype(np.float32)
        norms_before = np.linalg.norm(k, axis=-1)
        result = rope_correct_cpu(k, np.array([0, 1, 2, 3]), np.array([100, 101, 102, 103]), 8)
        norms_after = np.linalg.norm(result, axis=-1)
        np.testing.assert_allclose(norms_after, norms_before, atol=1e-5)

    def test_reversible(self):
        k = np.random.randn(2, 4, 8).astype(np.float32)
        k_orig = k.copy()
        fwd = rope_correct_cpu(k.copy(), np.array([0, 1, 2, 3]), np.array([10, 11, 12, 13]), 8)
        rev = rope_correct_cpu(fwd, np.array([10, 11, 12, 13]), np.array([0, 1, 2, 3]), 8)
        np.testing.assert_allclose(rev, k_orig, atol=1e-5)

    def test_empty_positions(self):
        k = np.random.randn(2, 0, 8).astype(np.float32)
        result = rope_correct_cpu(k, np.array([]), np.array([]), 8)
        assert result.shape == (2, 0, 8)

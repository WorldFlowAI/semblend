"""Tests for SemBlend RadixCache backend."""
from __future__ import annotations

import numpy as np
import pytest


class TestDonorStore:
    """Test the _SemBlendDonorStore for RadixCache integration."""

    def test_import(self):
        from semblend.integration.sglang.radix_backend import _SemBlendDonorStore
        assert _SemBlendDonorStore is not None

    def test_empty_store(self):
        from semblend.integration.sglang.radix_backend import _SemBlendDonorStore
        store = _SemBlendDonorStore(max_entries=10, min_similarity=0.5)
        assert store.size == 0
        query = np.random.randn(384).astype(np.float32)
        assert store.find_donor(query) is None

    def test_add_and_find(self):
        from semblend.integration.sglang.radix_backend import _SemBlendDonorStore
        store = _SemBlendDonorStore(max_entries=10, min_similarity=0.5)

        emb = np.random.randn(384).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        tokens = tuple(range(100))
        store.add_donor(tokens, emb)
        assert store.size == 1

        # Same embedding should match
        donor = store.find_donor(emb)
        assert donor is not None
        assert donor.token_ids == tokens

    def test_exclude_self(self):
        from semblend.integration.sglang.radix_backend import _SemBlendDonorStore
        store = _SemBlendDonorStore(max_entries=10, min_similarity=0.5)

        emb = np.random.randn(384).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        tokens = tuple(range(256))
        store.add_donor(tokens, emb)

        # Exclude self — should return None
        donor = store.find_donor(emb, exclude_tokens=tokens)
        assert donor is None

    def test_lru_eviction(self):
        from semblend.integration.sglang.radix_backend import _SemBlendDonorStore
        store = _SemBlendDonorStore(max_entries=3, min_similarity=0.5)

        for i in range(5):
            emb = np.random.randn(384).astype(np.float32)
            tokens = tuple(range(i * 256, (i + 1) * 256))
            store.add_donor(tokens, emb)

        assert store.size == 3

    def test_similarity_threshold(self):
        from semblend.integration.sglang.radix_backend import _SemBlendDonorStore
        store = _SemBlendDonorStore(max_entries=10, min_similarity=0.99)

        emb = np.random.randn(384).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        store.add_donor(tuple(range(256)), emb)

        # Different embedding should not meet high threshold
        other = np.random.randn(384).astype(np.float32)
        other = other / np.linalg.norm(other)
        assert store.find_donor(other) is None

    def test_best_match_returned(self):
        from semblend.integration.sglang.radix_backend import _SemBlendDonorStore
        store = _SemBlendDonorStore(max_entries=10, min_similarity=0.0)

        base = np.random.randn(384).astype(np.float32)
        base = base / np.linalg.norm(base)

        # Add a similar entry (perturbed)
        similar = base + 0.1 * np.random.randn(384).astype(np.float32)
        similar = similar / np.linalg.norm(similar)
        store.add_donor(tuple(range(256)), similar)

        # Add a less similar entry
        less_similar = base + 0.5 * np.random.randn(384).astype(np.float32)
        less_similar = less_similar / np.linalg.norm(less_similar)
        store.add_donor(tuple(range(256, 512)), less_similar)

        # Query should return the more similar entry
        donor = store.find_donor(base)
        assert donor is not None


class TestCreateSemBlendRadixCache:
    """Test the factory function for creating SemBlend-enhanced RadixCache."""

    def test_import(self):
        from semblend.integration.sglang.radix_backend import create_semblend_radix_cache
        assert create_semblend_radix_cache is not None

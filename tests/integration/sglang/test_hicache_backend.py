"""Tests for SemBlend SGLang HiCacheStorage backend."""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestSemBlendHiCacheStorageImport:
    """Test that the HiCache backend can be imported."""

    def test_import(self):
        from semblend.integration.sglang.hicache_backend import SemBlendHiCacheStorage
        assert SemBlendHiCacheStorage is not None

    def test_donor_index_import(self):
        from semblend.integration.sglang.hicache_backend import _SemBlendDonorIndex
        assert _SemBlendDonorIndex is not None


class TestSemBlendDonorIndex:
    """Test the in-process donor index."""

    def test_empty_index(self):
        from semblend.integration.sglang.hicache_backend import _SemBlendDonorIndex
        index = _SemBlendDonorIndex(max_entries=10, min_similarity=0.5)
        assert index.size == 0
        assert index.find_semantic_match(np.random.randn(384).astype(np.float32)) is None

    def test_register_and_find(self):
        from semblend.integration.sglang.hicache_backend import _SemBlendDonorIndex
        index = _SemBlendDonorIndex(max_entries=10, min_similarity=0.5)

        # Register a donor
        emb = np.random.randn(384).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        index.register("key1", emb, [1, 2, 3])
        assert index.size == 1

        # Query with same embedding should match
        result = index.find_semantic_match(emb)
        assert result == "key1"

    def test_threshold_filtering(self):
        from semblend.integration.sglang.hicache_backend import _SemBlendDonorIndex
        index = _SemBlendDonorIndex(max_entries=10, min_similarity=0.95)

        # Register a donor
        emb = np.random.randn(384).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        index.register("key1", emb, [1, 2, 3])

        # Query with different embedding should not match at high threshold
        other = np.random.randn(384).astype(np.float32)
        other = other / np.linalg.norm(other)
        result = index.find_semantic_match(other)
        assert result is None

    def test_lru_eviction(self):
        from semblend.integration.sglang.hicache_backend import _SemBlendDonorIndex
        index = _SemBlendDonorIndex(max_entries=3, min_similarity=0.5)

        for i in range(5):
            emb = np.random.randn(384).astype(np.float32)
            index.register(f"key{i}", emb, [i])

        assert index.size == 3

    def test_duplicate_registration(self):
        from semblend.integration.sglang.hicache_backend import _SemBlendDonorIndex
        index = _SemBlendDonorIndex(max_entries=10, min_similarity=0.5)

        emb = np.random.randn(384).astype(np.float32)
        index.register("key1", emb, [1, 2, 3])
        index.register("key1", emb, [1, 2, 3])

        assert index.size == 1


class TestSemBlendHiCacheStorage:
    """Test the HiCacheStorage backend."""

    def _make_storage(self, **env_overrides):
        from semblend.integration.sglang.hicache_backend import SemBlendHiCacheStorage

        env = {"SEMBLEND_ENABLED": "1", "SEMBLEND_MIN_SIMILARITY": "0.60"}
        env.update(env_overrides)

        with patch.dict(os.environ, env):
            mock_config = MagicMock()
            return SemBlendHiCacheStorage(storage_config=mock_config)

    def test_creation(self):
        storage = self._make_storage()
        assert storage._enabled is True

    def test_creation_disabled(self):
        storage = self._make_storage(SEMBLEND_ENABLED="0")
        assert storage._enabled is False

    def test_set_and_get(self):
        storage = self._make_storage()
        storage.set("hash1", value="kv_data_1")
        result = storage.get("hash1")
        assert result == "kv_data_1"

    def test_get_missing(self):
        storage = self._make_storage()
        result = storage.get("nonexistent")
        assert result is None

    def test_exists(self):
        storage = self._make_storage()
        storage.set("hash1", value="data")
        assert storage.exists("hash1") is True
        assert storage.exists("hash2") is False

    def test_batch_exists_consecutive(self):
        storage = self._make_storage()
        storage.set("h1", value="d1")
        storage.set("h2", value="d2")

        count = storage.batch_exists(["h1", "h2", "h3"])
        assert count == 2

    def test_batch_exists_gap(self):
        storage = self._make_storage()
        storage.set("h1", value="d1")
        storage.set("h3", value="d3")

        # Should stop at first miss (h2)
        count = storage.batch_exists(["h1", "h2", "h3"])
        assert count == 1

    def test_batch_set_and_get(self):
        storage = self._make_storage()
        storage.batch_set(["k1", "k2"], values=["v1", "v2"])
        results = storage.batch_get(["k1", "k2"])
        assert results == ["v1", "v2"]

    def test_clear(self):
        storage = self._make_storage()
        storage.set("h1", value="d1")
        assert storage.exists("h1")
        storage.clear()
        assert not storage.exists("h1")

    def test_stats(self):
        storage = self._make_storage()
        storage.set("h1", value="d1")
        storage.get("h1")
        storage.get("h2")  # miss

        stats = storage.get_stats()
        assert stats["exact_hits"] >= 1
        assert stats["stores"] >= 1

    def test_register_token_ids(self):
        storage = self._make_storage()
        storage.register_token_ids("hash1", [100, 200, 300])
        assert storage._token_map["hash1"] == [100, 200, 300]

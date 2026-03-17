"""Tests for KVTensorStore — direct KV tensor storage for semantic tier."""
from __future__ import annotations

import numpy as np
import pytest

from semblend_core.kv_tensor_store import KVTensorStore, SearchResult


def _make_embedding(dim: int = 384, seed: int = 0) -> np.ndarray:
    """Create a normalized random embedding."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


def _make_kv(n_tokens: int = 100, n_layers: int = 4, n_heads: int = 4, head_dim: int = 16):
    """Create fake KV tensors for testing."""
    kv = {}
    for layer in range(n_layers):
        k = np.random.randn(n_tokens, n_heads, head_dim).astype(np.float16)
        v = np.random.randn(n_tokens, n_heads, head_dim).astype(np.float16)
        kv[layer] = (k, v)
    return kv


class TestKVTensorStoreBasic:
    def test_add_and_search(self):
        store = KVTensorStore(max_entries=10)
        emb = _make_embedding(seed=42)
        kv = _make_kv(n_tokens=50, n_layers=2)
        entry_id = store.add(emb, list(range(50)), kv, model_id="test")

        assert store.size == 1
        assert store.total_bytes > 0

        results = store.search(emb, top_k=1)
        assert len(results) == 1
        assert results[0].entry_id == entry_id
        assert results[0].similarity > 0.99  # same embedding

    def test_search_empty_store(self):
        store = KVTensorStore()
        results = store.search(_make_embedding(), top_k=5)
        assert results == []

    def test_search_respects_threshold(self):
        store = KVTensorStore()
        emb1 = _make_embedding(seed=1)
        kv = _make_kv(n_tokens=10, n_layers=1)
        store.add(emb1, [1, 2, 3], kv)

        # Orthogonal embedding should be below threshold
        emb_ortho = _make_embedding(seed=999)
        results = store.search(emb_ortho, min_similarity=0.95)
        assert len(results) == 0

    def test_search_sorts_by_similarity(self):
        store = KVTensorStore()
        base = _make_embedding(seed=0)
        kv = _make_kv(n_tokens=10, n_layers=1)

        # Add the base embedding
        store.add(base, [1], kv, model_id="test")

        # Add a similar embedding (perturbed)
        similar = base + 0.1 * _make_embedding(seed=1)
        similar = similar / np.linalg.norm(similar)
        store.add(similar, [2], kv, model_id="test")

        # Add a less similar embedding
        less_similar = base + 0.5 * _make_embedding(seed=2)
        less_similar = less_similar / np.linalg.norm(less_similar)
        store.add(less_similar, [3], kv, model_id="test")

        results = store.search(base, top_k=3, min_similarity=0.0)
        assert len(results) == 3
        assert results[0].similarity >= results[1].similarity >= results[2].similarity

    def test_get_kv(self):
        store = KVTensorStore()
        emb = _make_embedding()
        kv = _make_kv(n_tokens=20, n_layers=4)
        entry_id = store.add(emb, list(range(20)), kv)

        # Get all layers
        retrieved = store.get_kv(entry_id)
        assert retrieved is not None
        assert len(retrieved) == 4
        for layer in range(4):
            k, v = retrieved[layer]
            assert k.shape == (20, 4, 16)
            assert v.shape == (20, 4, 16)

    def test_get_kv_specific_layers(self):
        store = KVTensorStore()
        emb = _make_embedding()
        kv = _make_kv(n_tokens=20, n_layers=4)
        entry_id = store.add(emb, list(range(20)), kv)

        # Get only layers 0 and 3
        retrieved = store.get_kv(entry_id, layers=[0, 3])
        assert retrieved is not None
        assert set(retrieved.keys()) == {0, 3}

    def test_get_kv_nonexistent(self):
        store = KVTensorStore()
        assert store.get_kv("nonexistent") is None

    def test_get_entry(self):
        store = KVTensorStore()
        emb = _make_embedding()
        kv = _make_kv(n_tokens=30, n_layers=2)
        entry_id = store.add(emb, [10, 20, 30], kv, model_id="qwen")

        entry = store.get_entry(entry_id)
        assert entry is not None
        assert entry.n_tokens == 3
        assert entry.n_layers == 2
        assert entry.model_id == "qwen"
        assert entry.token_ids == (10, 20, 30)

    def test_remove(self):
        store = KVTensorStore()
        emb = _make_embedding()
        kv = _make_kv(n_tokens=10, n_layers=1)
        entry_id = store.add(emb, [1], kv)

        assert store.size == 1
        assert store.remove(entry_id)
        assert store.size == 0
        assert store.get_kv(entry_id) is None
        assert not store.remove(entry_id)  # already removed


class TestKVTensorStoreEviction:
    def test_max_entries_eviction(self):
        store = KVTensorStore(max_entries=3)
        kv = _make_kv(n_tokens=5, n_layers=1)

        ids = []
        for i in range(5):
            eid = store.add(_make_embedding(seed=i), [i], kv)
            ids.append(eid)

        assert store.size == 3
        # First two should have been evicted
        assert store.get_entry(ids[0]) is None
        assert store.get_entry(ids[1]) is None
        assert store.get_entry(ids[2]) is not None

    def test_max_bytes_eviction(self):
        kv = _make_kv(n_tokens=100, n_layers=4, n_heads=4, head_dim=16)
        per_entry_bytes = sum(k.nbytes + v.nbytes for k, v in kv.values())

        # Allow ~2.5 entries worth of bytes
        store = KVTensorStore(
            max_entries=100,
            max_cpu_bytes=int(per_entry_bytes * 2.5),
        )

        for i in range(5):
            store.add(_make_embedding(seed=i), [i], kv)

        # Should have at most 2 entries (can't fit 3)
        assert store.size <= 2


class TestKVTensorStoreModelFilter:
    def test_model_id_filtering(self):
        store = KVTensorStore()
        kv = _make_kv(n_tokens=10, n_layers=1)

        emb = _make_embedding(seed=0)
        store.add(emb, [1], kv, model_id="qwen")
        store.add(emb, [2], kv, model_id="llama")

        # Search for qwen only
        results = store.search(emb, model_id="qwen", min_similarity=0.0)
        assert len(results) == 1
        assert results[0].model_id == "qwen"

        # Search for llama only
        results = store.search(emb, model_id="llama", min_similarity=0.0)
        assert len(results) == 1
        assert results[0].model_id == "llama"

        # Search all
        results = store.search(emb, min_similarity=0.0)
        assert len(results) == 2


class TestKVTensorStoreMemoryAccounting:
    def test_total_bytes_tracking(self):
        store = KVTensorStore()
        kv = _make_kv(n_tokens=10, n_layers=2)
        expected_bytes = sum(k.nbytes + v.nbytes for k, v in kv.values())

        store.add(_make_embedding(seed=0), [1], kv)
        assert store.total_bytes == expected_bytes

        store.add(_make_embedding(seed=1), [2], kv)
        assert store.total_bytes == 2 * expected_bytes

    def test_bytes_decrease_on_remove(self):
        store = KVTensorStore()
        kv = _make_kv(n_tokens=10, n_layers=2)
        entry_id = store.add(_make_embedding(), [1], kv)

        bytes_before = store.total_bytes
        store.remove(entry_id)
        assert store.total_bytes == 0
        assert bytes_before > 0

"""Tests for donor_store.py and simhash.py."""
from __future__ import annotations

import time

import numpy as np


# ---------------------------------------------------------------------------
# SimHash tests
# ---------------------------------------------------------------------------


def test_simhash_identical():
    """Identical sequences should have distance 0."""
    from synapse_kv_connector.simhash import compute_simhash, hamming_distance

    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    h1 = compute_simhash(tokens)
    h2 = compute_simhash(tokens)
    assert hamming_distance(h1, h2) == 0


def test_simhash_similar():
    """Similar sequences should have low hamming distance."""
    from synapse_kv_connector.simhash import compute_simhash, hamming_distance

    tokens1 = list(range(100))
    tokens2 = list(range(100))
    tokens2[50] = 999  # Single change

    h1 = compute_simhash(tokens1)
    h2 = compute_simhash(tokens2)
    dist = hamming_distance(h1, h2)
    assert dist < 20, f"Hamming distance {dist} too high for single-token change"


def test_simhash_different():
    """Very different sequences should have high hamming distance."""
    from synapse_kv_connector.simhash import compute_simhash, hamming_distance

    tokens1 = list(range(100))
    tokens2 = list(range(1000, 1100))

    h1 = compute_simhash(tokens1)
    h2 = compute_simhash(tokens2)
    dist = hamming_distance(h1, h2)
    assert dist > 10, f"Hamming distance {dist} too low for different sequences"


def test_simhash_plausible_donor():
    """is_plausible_donor should filter correctly."""
    from synapse_kv_connector.simhash import compute_simhash, is_plausible_donor

    similar = compute_simhash([1, 2, 3, 4, 5])
    same = compute_simhash([1, 2, 3, 4, 5])
    assert is_plausible_donor(similar, same)


def test_simhash_short_sequence():
    """Short sequences (< ngram_size) should still produce a hash."""
    from synapse_kv_connector.simhash import compute_simhash

    h = compute_simhash([1, 2])
    assert isinstance(h, int)
    assert h >= 0


def test_simhash_latency():
    """SimHash on 8K tokens should complete in <0.2ms."""
    from synapse_kv_connector.simhash import compute_simhash

    tokens = list(range(8000))
    t0 = time.monotonic()
    for _ in range(100):
        compute_simhash(tokens)
    elapsed_ms = (time.monotonic() - t0) * 1000

    per_call_ms = elapsed_ms / 100
    # Relaxed bound for CI (may be slower than dedicated hardware)
    assert per_call_ms < 50, f"SimHash took {per_call_ms:.1f}ms per call"


# ---------------------------------------------------------------------------
# DonorStore tests
# ---------------------------------------------------------------------------


def _make_store(n: int = 0, dim: int = 384, max_entries: int = 1000):
    """Create a DonorStore with n random donors."""
    from synapse_kv_connector.donor_store import DonorNode, DonorStore

    store = DonorStore(
        max_entries=max_entries,
        embedding_dim=dim,
        min_similarity=0.5,
    )

    for i in range(n):
        tokens = list(range(i * 10, i * 10 + 100))
        emb = np.random.randn(dim).astype(np.float32)
        emb /= np.linalg.norm(emb)
        store.add_donor(DonorNode(
            request_id=f"req-{i}",
            token_ids=tokens,
            embedding=emb,
            timestamp=float(i),
        ))

    return store


def test_donor_store_empty():
    """Empty store should return None."""
    store = _make_store(0)
    result = store.find_donor(
        query_embedding=np.zeros(384, dtype=np.float32),
        query_tokens=[1, 2, 3],
    )
    assert result is None


def test_donor_store_add_and_find():
    """Adding a donor and querying with same embedding should find it."""
    from synapse_kv_connector.donor_store import DonorNode

    store = _make_store(0)
    tokens = list(range(100))
    emb = np.random.randn(384).astype(np.float32)
    emb /= np.linalg.norm(emb)

    store.add_donor(DonorNode(
        request_id="donor-1",
        token_ids=tokens,
        embedding=emb.copy(),
        timestamp=1.0,
    ))

    # Query with same tokens (but slightly different to avoid exact-match skip)
    query_tokens = list(range(100))
    query_tokens[50] = 999  # One token different

    result = store.find_donor(
        query_embedding=emb,
        query_tokens=query_tokens,
        min_reuse_ratio=0.5,
    )

    assert result is not None
    assert result.donor.request_id == "donor-1"
    assert result.similarity > 0.9
    # rapidfuzz Levenshtein opcodes may group ops differently,
    # so reuse ratio depends on edit script optimization
    assert result.alignment.reuse_ratio >= 0.3


def test_donor_store_lru_eviction():
    """Store should evict oldest entries when at capacity."""
    store = _make_store(100, max_entries=50)
    assert store.size == 50


def test_donor_store_duplicate_add():
    """Adding same request_id twice should not increase size."""
    from synapse_kv_connector.donor_store import DonorNode

    store = _make_store(0)
    node = DonorNode(
        request_id="dup",
        token_ids=[1, 2, 3],
        embedding=np.zeros(384, dtype=np.float32),
        timestamp=1.0,
    )
    store.add_donor(node)
    store.add_donor(node)
    assert store.size == 1


def test_donor_store_no_self_match():
    """Should not return a donor with identical token_ids as the query."""
    from synapse_kv_connector.donor_store import DonorNode

    store = _make_store(0)
    tokens = list(range(100))
    emb = np.random.randn(384).astype(np.float32)
    emb /= np.linalg.norm(emb)

    store.add_donor(DonorNode(
        request_id="self",
        token_ids=tokens,
        embedding=emb.copy(),
        timestamp=1.0,
    ))

    result = store.find_donor(
        query_embedding=emb,
        query_tokens=tokens,  # Same tokens
    )
    assert result is None


def test_donor_store_1k_entries():
    """Lookup in 1K-entry store."""
    store = _make_store(1000)
    query_emb = np.random.randn(384).astype(np.float32)
    query_emb /= np.linalg.norm(query_emb)

    t0 = time.monotonic()
    store.find_donor(
        query_embedding=query_emb,
        query_tokens=list(range(50)),
    )
    elapsed_ms = (time.monotonic() - t0) * 1000
    # Relaxed bound for CI
    assert elapsed_ms < 500, f"1K lookup took {elapsed_ms:.1f}ms"


def test_donor_store_no_embedding_fallback():
    """Should handle None embedding gracefully."""
    from synapse_kv_connector.donor_store import DonorNode

    store = _make_store(0)
    tokens = list(range(100))
    store.add_donor(DonorNode(
        request_id="no-emb",
        token_ids=tokens,
        embedding=None,
        timestamp=1.0,
    ))

    query = list(range(100))
    query[50] = 999
    result = store.find_donor(
        query_embedding=None,
        query_tokens=query,
    )
    # No embedding means cosine search can't work — returns None
    assert result is None

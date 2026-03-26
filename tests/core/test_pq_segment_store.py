"""Tests for PQ segment store."""

from __future__ import annotations

import threading

import numpy as np


def _random_normalized(rng, n, dim=384):
    """Generate L2-normalized random embeddings."""
    embs = rng.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / norms


def test_codebook_training():
    from semblend_core.pq_segment_store import train_pq_codebook

    rng = np.random.RandomState(42)
    embs = _random_normalized(rng, 1000)
    cb = train_pq_codebook(embs, n_subquantizers=48, n_centroids=256)
    assert cb.trained
    assert cb.centroids.shape == (48, 256, 8)
    assert cb.nbytes > 0


def test_encode_decode_roundtrip():
    from semblend_core.pq_segment_store import (
        adc_cosine_similarities,
        pq_encode_batch,
        train_pq_codebook,
    )

    rng = np.random.RandomState(42)
    train_embs = _random_normalized(rng, 2000)
    cb = train_pq_codebook(train_embs, n_subquantizers=48, n_centroids=256)

    test_embs = _random_normalized(rng, 50)
    codes = pq_encode_batch(test_embs, cb)
    assert codes.shape == (50, 48)
    assert codes.dtype == np.uint8

    # ADC vs exact cosine
    query = _random_normalized(rng, 1)[0]
    adc_sims = adc_cosine_similarities(query, codes, cb)
    exact_sims = test_embs @ query

    # PQ approximation should correlate reasonably with exact.
    # Random 384-dim vectors have less structure than real embeddings,
    # so correlation is lower (~0.7) than with real data (~0.9+).
    correlation = np.corrcoef(adc_sims, exact_sims)[0, 1]
    assert correlation > 0.60, f"ADC-exact correlation too low: {correlation:.3f}"


def test_segment_store_lifecycle():
    from semblend_core.pq_segment_store import PQSegmentStore

    rng = np.random.RandomState(42)

    store = PQSegmentStore(max_entries=100, train_threshold=5)
    assert store.size == 0
    assert not store.codebook_trained

    # Add donors below threshold (buffered)
    for i in range(4):
        segs = _random_normalized(rng, 10)
        store.add_segments(f"donor-{i}", segs)
    assert store.size == 4
    assert not store.codebook_trained

    # Add 5th donor triggers training
    store.add_segments("donor-4", _random_normalized(rng, 10))
    assert store.codebook_trained
    assert store.size == 5

    # Compare segments
    query = _random_normalized(rng, 5)
    scores = store.compare_segments(query, ["donor-0", "donor-4"])
    assert len(scores) == 2
    assert all(isinstance(s, float) for s in scores)

    # Evict
    store.evict("donor-0")
    scores = store.compare_segments(query, ["donor-0"])
    assert scores[0] == 0.0


def test_segment_store_pre_training_comparison():
    """Before codebook training, uses exact cosine in buffer."""
    from semblend_core.pq_segment_store import PQSegmentStore

    rng = np.random.RandomState(42)

    store = PQSegmentStore(max_entries=100, train_threshold=100)  # high threshold
    segs = _random_normalized(rng, 10)
    store.add_segments("donor-0", segs)

    # Compare against self should give high score
    scores = store.compare_segments(segs[:3], ["donor-0"])
    assert scores[0] > 0.5


def test_memory_efficiency():
    """PQ codes at 1K donors should use < 2MB."""
    from semblend_core.pq_segment_store import PQSegmentStore

    rng = np.random.RandomState(42)

    store = PQSegmentStore(max_entries=1100, train_threshold=50)
    for i in range(1000):
        segs = _random_normalized(rng, 30)
        store.add_segments(f"donor-{i}", segs)

    assert store.codebook_trained
    nbytes = store.nbytes
    assert nbytes < 2_000_000, f"PQ store too large: {nbytes} bytes"


def test_get_segment_similarity():
    from semblend_core.pq_segment_store import PQSegmentStore

    rng = np.random.RandomState(42)

    store = PQSegmentStore(max_entries=100, train_threshold=5)
    for i in range(6):
        store.add_segments(f"donor-{i}", _random_normalized(rng, 10))

    query = _random_normalized(rng, 1)[0]
    sim = store.get_segment_similarity(query, "donor-0", chunk_idx=0)
    assert isinstance(sim, float)

    # Out of range chunk
    sim_oob = store.get_segment_similarity(query, "donor-0", chunk_idx=999)
    assert sim_oob == 0.0

    # Unknown donor
    sim_unknown = store.get_segment_similarity(query, "unknown", chunk_idx=0)
    assert sim_unknown == 0.0


def test_duplicate_add_ignored():
    from semblend_core.pq_segment_store import PQSegmentStore

    rng = np.random.RandomState(42)

    store = PQSegmentStore(max_entries=100, train_threshold=100)
    segs = _random_normalized(rng, 10)
    store.add_segments("donor-0", segs)
    store.add_segments("donor-0", _random_normalized(rng, 10))  # duplicate
    assert store.size == 1


def test_thread_safety():
    """Basic concurrent add + compare."""
    from semblend_core.pq_segment_store import PQSegmentStore

    rng = np.random.RandomState(42)

    store = PQSegmentStore(max_entries=200, train_threshold=10)
    errors = []

    def add_donors(start, count):
        try:
            for i in range(start, start + count):
                segs = _random_normalized(np.random.RandomState(i), 10)
                store.add_segments(f"donor-{i}", segs)
        except Exception as e:
            errors.append(e)

    def compare_donors():
        try:
            query = _random_normalized(np.random.RandomState(999), 5)
            for _ in range(20):
                store.compare_segments(query, ["donor-0", "donor-5"])
        except Exception as e:
            errors.append(e)

    threads = [
        threading.Thread(target=add_donors, args=(0, 50)),
        threading.Thread(target=add_donors, args=(50, 50)),
        threading.Thread(target=compare_donors),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread safety errors: {errors}"

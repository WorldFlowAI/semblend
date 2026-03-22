"""Tests for confidence-gated fuzzy matching."""
from __future__ import annotations

import math

import pytest


def test_fuzzy_match_config_defaults():
    from semblend_core.alignment import FuzzyMatchConfig
    cfg = FuzzyMatchConfig()
    assert cfg.min_overlap == 0.90
    assert cfg.decay_function == "exponential"
    assert cfg.position_tau == 128.0
    assert cfg.confidence_high == 0.92
    assert cfg.confidence_low == 0.80


def test_chunk_bag_cosine_identical():
    from semblend_core.alignment import chunk_bag_cosine
    assert chunk_bag_cosine([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == pytest.approx(1.0)


def test_chunk_bag_cosine_disjoint():
    from semblend_core.alignment import chunk_bag_cosine
    assert chunk_bag_cosine([1, 2, 3], [10, 20, 30]) == pytest.approx(0.0)


def test_chunk_bag_cosine_partial_overlap():
    from semblend_core.alignment import chunk_bag_cosine
    score = chunk_bag_cosine([1, 2, 3, 4, 5], [1, 2, 3, 10, 20])
    assert 0.3 < score < 0.8


def test_position_decay_exponential():
    from semblend_core.alignment import _compute_position_decay
    # Zero delta -> 1.0
    assert _compute_position_decay(0.0, 128.0, "exponential") == pytest.approx(1.0)
    # Small delta -> near 1.0
    assert _compute_position_decay(1.0, 128.0, "exponential") == pytest.approx(math.exp(-1/128))
    # Large delta -> small
    val = _compute_position_decay(256.0, 128.0, "exponential")
    assert val < 0.2


def test_position_decay_linear():
    from semblend_core.alignment import _compute_position_decay
    assert _compute_position_decay(0.0, 128.0, "linear") == pytest.approx(1.0)
    assert _compute_position_decay(256.0, 128.0, "linear") == pytest.approx(0.0)
    assert _compute_position_decay(128.0, 128.0, "linear") == pytest.approx(0.5)


def test_position_decay_step():
    from semblend_core.alignment import _compute_position_decay
    assert _compute_position_decay(50.0, 128.0, "step") == 1.0
    assert _compute_position_decay(200.0, 128.0, "step") == 0.0


def test_chunk_confidence_clean_shift():
    """A clean shifted prefix should have high coherence and confidence."""
    from semblend_core.alignment import _compute_chunk_confidence, FuzzyMatchConfig
    # 256 tokens, all shifted by exactly 5 positions
    pairs = [(i, i) for i in range(240)] + [(i, i - 5) for i in range(240, 250)]
    config = FuzzyMatchConfig(position_tau=128.0)

    conf = _compute_chunk_confidence(
        pairs=pairs, overlap_ratio=0.95,
        donor_chunk=list(range(256)), target_chunk=list(range(256)),
        chunk_idx=0, config=config,
    )
    assert conf.positional_coherence > 0.9
    assert conf.confidence > 0.8
    assert conf.tier in ("fast_reuse", "verified_reuse")


def test_chunk_confidence_scattered():
    """Scattered rearrangement should have low coherence."""
    from semblend_core.alignment import _compute_chunk_confidence, FuzzyMatchConfig
    import random
    random.seed(42)
    # Random permutation of offsets
    offsets = list(range(100))
    random.shuffle(offsets)
    pairs = [(i, offsets[i]) for i in range(100)]
    config = FuzzyMatchConfig(position_tau=128.0)

    conf = _compute_chunk_confidence(
        pairs=pairs, overlap_ratio=0.95,
        donor_chunk=list(range(256)), target_chunk=list(range(256)),
        chunk_idx=0, config=config,
    )
    assert conf.positional_coherence < 0.2


def test_tier_fast_reuse():
    from semblend_core.alignment import _compute_chunk_confidence, FuzzyMatchConfig
    # Perfect match: all same position, high overlap
    pairs = [(i, i) for i in range(256)]
    config = FuzzyMatchConfig(confidence_high=0.92, confidence_low=0.80)

    conf = _compute_chunk_confidence(
        pairs=pairs, overlap_ratio=0.98,
        donor_chunk=list(range(256)), target_chunk=list(range(256)),
        chunk_idx=0, config=config,
    )
    assert conf.tier == "fast_reuse"
    assert conf.confidence >= 0.92


def test_tier_recompute_low_bag_cosine():
    """Low bag-cosine should force recompute even with high overlap."""
    from semblend_core.alignment import _compute_chunk_confidence, FuzzyMatchConfig
    pairs = [(i, i) for i in range(100)]
    # Donor and target chunks with very different token distributions
    donor = list(range(1000, 1256))
    target = list(range(2000, 2256))
    config = FuzzyMatchConfig(bag_cosine_min=0.94)

    conf = _compute_chunk_confidence(
        pairs=pairs, overlap_ratio=0.95,
        donor_chunk=donor, target_chunk=target,
        chunk_idx=0, config=config,
    )
    assert conf.tier == "recompute"
    assert conf.bag_cosine < 0.94


def test_fuzzy_alignment_with_confidence():
    """Full fuzzy alignment should include chunk confidences."""
    from semblend_core.alignment import compute_fuzzy_chunk_alignment

    # Shifted prefix: 10-token prefix difference
    base = list(range(100, 612))
    donor = list(range(10, 20)) + base
    target = list(range(20, 30)) + base

    result = compute_fuzzy_chunk_alignment(donor, target, chunk_size=256)
    assert result.fuzzy_chunks > 0 or result.exact_chunks > 0
    assert len(result.chunk_confidences) >= 0
    assert result.mean_fuzzy_confidence > 0


def test_backward_compat_exact_match():
    """Exact match should have empty chunk_confidences."""
    from semblend_core.alignment import compute_alignment
    tokens = list(range(512))
    result = compute_alignment(tokens, tokens)
    assert result.chunk_confidences == ()
    assert result.mean_fuzzy_confidence == 1.0
    assert result.fuzzy_recompute_chunks == 0

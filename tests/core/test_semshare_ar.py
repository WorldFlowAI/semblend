"""Tests for SemShareKV Attention Recovery metric."""
from __future__ import annotations

import numpy as np
import pytest

from semblend_core.semshare.attention_recovery import (
    AttentionRecoveryResult,
    compute_attention_recovery,
    compute_per_layer_ar,
)


class TestAttentionRecovery:
    def test_uniform_attention(self) -> None:
        """Uniform weights: need 90% of tokens for 90% coverage."""
        weights = np.ones(100, dtype=np.float32)
        result = compute_attention_recovery(weights, threshold=0.90)

        assert result.ar_value == 90
        assert result.threshold_met is True
        assert result.seq_len == 100

    def test_concentrated_attention(self) -> None:
        """One-hot attention: AR = 1."""
        weights = np.zeros(100, dtype=np.float32)
        weights[42] = 1.0
        result = compute_attention_recovery(weights, threshold=0.90)

        assert result.ar_value == 1
        assert result.threshold_met is True

    def test_two_token_concentration(self) -> None:
        """Two tokens hold all attention."""
        weights = np.zeros(100, dtype=np.float32)
        weights[10] = 0.5
        weights[50] = 0.5
        result = compute_attention_recovery(weights, threshold=0.90)

        assert result.ar_value == 2
        assert result.threshold_met is True

    def test_empty_sequence(self) -> None:
        weights = np.array([], dtype=np.float32)
        result = compute_attention_recovery(weights, threshold=0.90)

        assert result.ar_value == 0
        assert result.threshold_met is False
        assert result.seq_len == 0

    def test_zero_attention(self) -> None:
        weights = np.zeros(10, dtype=np.float32)
        result = compute_attention_recovery(weights, threshold=0.90)

        assert result.threshold_met is False

    def test_ar_fraction(self) -> None:
        weights = np.ones(100, dtype=np.float32)
        result = compute_attention_recovery(weights, threshold=0.90)

        assert abs(result.ar_fraction - 0.90) < 0.02

    def test_result_is_frozen(self) -> None:
        weights = np.ones(10, dtype=np.float32)
        result = compute_attention_recovery(weights)
        with pytest.raises(AttributeError):
            result.ar_value = 999  # type: ignore[misc]

    def test_threshold_50_percent(self) -> None:
        """Lower threshold needs fewer tokens."""
        weights = np.ones(100, dtype=np.float32)
        result = compute_attention_recovery(weights, threshold=0.50)
        assert result.ar_value == 50

    def test_skewed_distribution(self) -> None:
        """Zipf-like distribution: few tokens cover most attention."""
        rng = np.random.RandomState(42)
        weights = 1.0 / np.arange(1, 101, dtype=np.float32)
        result = compute_attention_recovery(weights, threshold=0.90)

        # Zipf-like should need fewer than 90 tokens for 90% coverage
        assert result.ar_value < 70
        assert result.threshold_met is True


class TestPerLayerAR:
    def test_deeper_layers_lower_ar(self) -> None:
        """Paper's key observation: deeper layers have lower AR (sparser attention)."""
        layers = [
            np.ones(100, dtype=np.float32),  # uniform (high AR)
            np.ones(100, dtype=np.float32) * 0.5,  # still uniform
        ]
        # Make "deeper" layer more concentrated
        concentrated = np.zeros(100, dtype=np.float32)
        concentrated[:10] = 1.0
        layers.append(concentrated)

        results = compute_per_layer_ar(layers, threshold=0.90)

        assert len(results) == 3
        # Deeper layer (index 2) should have lower AR
        assert results[2].ar_value < results[0].ar_value

    def test_empty_layers(self) -> None:
        results = compute_per_layer_ar([], threshold=0.90)
        assert len(results) == 0

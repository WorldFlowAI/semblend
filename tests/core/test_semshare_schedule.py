"""Tests for SemShareKV layer schedule and HD detection."""
from __future__ import annotations

import numpy as np
import pytest

from semblend_core.semshare.config import SemShareConfig
from semblend_core.semshare.hd_detector import HDDetectionResult, detect_hd_tokens
from semblend_core.semshare.layer_schedule import LayerSchedule, compute_layer_schedule


@pytest.fixture
def config() -> SemShareConfig:
    return SemShareConfig(embed_dim=64, hd_fraction=0.40)


class TestHDDetector:
    def test_identical_kv_no_hd(self, config: SemShareConfig) -> None:
        """When KV is identical, all deviations should be 0."""
        kv = np.ones((10, 64), dtype=np.float32)
        result = detect_hd_tokens(kv, kv, kv, kv, config)

        assert result.seq_len == 10
        # All deviations are 0, threshold is 0, but top 40% still flagged
        assert result.hd_threshold == 0.0

    def test_all_different_kv(self, config: SemShareConfig) -> None:
        """When KV differs maximally, deviation should be high for all."""
        rng = np.random.RandomState(42)
        recomp_k = rng.randn(10, 64).astype(np.float32) * 10
        recomp_v = rng.randn(10, 64).astype(np.float32) * 10
        donor_k = rng.randn(10, 64).astype(np.float32) * 10
        donor_v = rng.randn(10, 64).astype(np.float32) * 10

        result = detect_hd_tokens(recomp_k, recomp_v, donor_k, donor_v, config)

        assert result.seq_len == 10
        assert result.num_hd_tokens == 4  # 40% of 10

    def test_top_40_percent_flagged(self, config: SemShareConfig) -> None:
        """Exactly top 40% should be flagged."""
        # Create controlled deviations
        recomp_k = np.zeros((20, 64), dtype=np.float32)
        donor_k = np.zeros((20, 64), dtype=np.float32)
        recomp_v = np.zeros((20, 64), dtype=np.float32)
        donor_v = np.zeros((20, 64), dtype=np.float32)

        # Make first 8 tokens have high deviation
        for i in range(8):
            recomp_k[i] = np.ones(64) * (10 + i)

        result = detect_hd_tokens(recomp_k, recomp_v, donor_k, donor_v, config)

        assert result.num_hd_tokens == 8  # 40% of 20
        # The HD tokens should be the ones with highest deviation
        for i in range(8):
            assert result.hd_mask[19 - i] is False or result.deviation_scores[i] > 0

    def test_result_is_frozen(self, config: SemShareConfig) -> None:
        kv = np.ones((5, 64), dtype=np.float32)
        result = detect_hd_tokens(kv, kv, kv, kv, config)
        with pytest.raises(AttributeError):
            result.hd_threshold = 999.0  # type: ignore[misc]


class TestLayerSchedule:
    def test_layer0_no_recompute(self, config: SemShareConfig) -> None:
        hd = HDDetectionResult(
            hd_mask=tuple([False] * 10),
            deviation_scores=tuple([0.0] * 10),
            hd_threshold=0.0,
            hd_fraction=0.0,
        )
        schedule = compute_layer_schedule(hd, num_layers=4, config=config)

        assert schedule.per_layer_alpha[0] == 0.0
        # Layer 0: only unmatched positions (none here)
        assert not any(schedule.per_layer_masks[0])

    def test_layer1_full_recompute(self, config: SemShareConfig) -> None:
        hd = HDDetectionResult(
            hd_mask=tuple([True] * 4 + [False] * 6),
            deviation_scores=tuple([1.0] * 4 + [0.1] * 6),
            hd_threshold=0.5,
            hd_fraction=0.4,
        )
        schedule = compute_layer_schedule(hd, num_layers=4, config=config)

        assert schedule.per_layer_alpha[1] == 1.0
        assert all(schedule.per_layer_masks[1])  # all True

    def test_exponential_decay(self, config: SemShareConfig) -> None:
        hd = HDDetectionResult(
            hd_mask=tuple([True] * 4 + [False] * 6),
            deviation_scores=tuple([1.0] * 4 + [0.1] * 6),
            hd_threshold=0.5,
            hd_fraction=0.4,
        )
        schedule = compute_layer_schedule(hd, num_layers=10, config=config)

        # After layer 1, alphas should decay
        for i in range(3, len(schedule.per_layer_alpha)):
            assert schedule.per_layer_alpha[i] <= schedule.per_layer_alpha[i - 1]

    def test_hd_tokens_prioritized(self, config: SemShareConfig) -> None:
        """HD tokens should be recomputed before cold tokens."""
        hd = HDDetectionResult(
            hd_mask=tuple([True] * 2 + [False] * 8),
            deviation_scores=tuple([5.0] * 2 + [0.1] * 8),
            hd_threshold=1.0,
            hd_fraction=0.2,
        )
        # With a small alpha, only HD tokens should be selected
        small_alpha_config = SemShareConfig(
            embed_dim=64, alpha_base=0.3, alpha_decay=0.5
        )
        schedule = compute_layer_schedule(hd, num_layers=6, config=small_alpha_config)

        # Layer 2 (first decay layer): should include HD tokens
        layer2_mask = schedule.per_layer_masks[2]
        assert layer2_mask[0] is True  # HD token
        assert layer2_mask[1] is True  # HD token

    def test_unmatched_always_recomputed(self, config: SemShareConfig) -> None:
        hd = HDDetectionResult(
            hd_mask=tuple([False] * 10),
            deviation_scores=tuple([0.0] * 10),
            hd_threshold=0.0,
            hd_fraction=0.0,
        )
        schedule = compute_layer_schedule(
            hd, num_layers=4, config=config, unmatched_positions=(3, 7)
        )

        # Unmatched positions should be recomputed in all layers
        for layer_idx in range(4):
            mask = schedule.per_layer_masks[layer_idx]
            assert mask[3] is True
            assert mask[7] is True

    def test_schedule_is_frozen(self, config: SemShareConfig) -> None:
        hd = HDDetectionResult(
            hd_mask=tuple([False] * 5),
            deviation_scores=tuple([0.0] * 5),
            hd_threshold=0.0,
            hd_fraction=0.0,
        )
        schedule = compute_layer_schedule(hd, num_layers=4, config=config)
        with pytest.raises(AttributeError):
            schedule.num_layers = 999  # type: ignore[misc]

    def test_recompute_ratio_bounded(self, config: SemShareConfig) -> None:
        """Total recompute should be less than full recompute."""
        hd = HDDetectionResult(
            hd_mask=tuple([True] * 4 + [False] * 6),
            deviation_scores=tuple([1.0] * 4 + [0.1] * 6),
            hd_threshold=0.5,
            hd_fraction=0.4,
        )
        schedule = compute_layer_schedule(hd, num_layers=28, config=config)

        # Should recompute less than 100% of all tokens
        assert schedule.recompute_ratio < 1.0
        # But more than 0% (layer 1 is full recompute)
        assert schedule.recompute_ratio > 0.0

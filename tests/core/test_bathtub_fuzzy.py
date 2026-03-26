"""Tests for fuzzy-aware bathtub curve and configurable recomputation."""

from __future__ import annotations

import pytest


def test_backward_compat_no_fuzzy():
    """With fuzzy_fraction=0, results should match the original implementation."""
    from semblend_core.bathtub import compute_layer_deviations

    devs = compute_layer_deviations(28, mismatch_fraction=0.1, model_name="qwen2.5-7b")
    recompute = [d.layer_idx for d in devs if d.should_recompute]
    # Qwen should recompute late layers [26, 27] and possibly [0]
    assert all(idx in recompute for idx in [26, 27])


def test_fuzzy_increases_recomputation():
    """Fuzzy fraction should cause more layers to be recomputed."""
    from semblend_core.bathtub import compute_layer_deviations

    devs_exact = compute_layer_deviations(
        28,
        mismatch_fraction=0.1,
        model_name="qwen2.5-7b",
    )
    devs_fuzzy = compute_layer_deviations(
        28,
        mismatch_fraction=0.1,
        model_name="qwen2.5-7b",
        fuzzy_fraction=0.5,
        mean_fuzzy_confidence=0.85,
        similarity=0.8,
    )
    n_exact = sum(1 for d in devs_exact if d.should_recompute)
    n_fuzzy = sum(1 for d in devs_fuzzy if d.should_recompute)
    assert n_fuzzy >= n_exact


def test_confidence_penalty_lowers_threshold():
    """Low confidence should lower threshold, increasing recomputation."""
    from semblend_core.bathtub import compute_layer_deviations

    devs_high_conf = compute_layer_deviations(
        28,
        mismatch_fraction=0.1,
        model_name="qwen2.5-7b",
        fuzzy_fraction=0.3,
        mean_fuzzy_confidence=0.95,
        similarity=0.8,
    )
    devs_low_conf = compute_layer_deviations(
        28,
        mismatch_fraction=0.1,
        model_name="qwen2.5-7b",
        fuzzy_fraction=0.3,
        mean_fuzzy_confidence=0.80,
        similarity=0.8,
    )
    n_high = sum(1 for d in devs_high_conf if d.should_recompute)
    n_low = sum(1 for d in devs_low_conf if d.should_recompute)
    assert n_low >= n_high


def test_position_factor_qwen():
    from semblend_core.bathtub import position_factor

    # Qwen: late layers more sensitive
    early = position_factor(0, 28, "qwen2.5-7b")
    late = position_factor(27, 28, "qwen2.5-7b")
    assert late > early
    assert early == pytest.approx(0.5)
    assert late == pytest.approx(1.0)


def test_position_factor_llama():
    from semblend_core.bathtub import position_factor

    # LLaMA: early layers more sensitive
    early = position_factor(0, 32, "llama-3.1-8b")
    late = position_factor(31, 32, "llama-3.1-8b")
    assert early > late
    assert early == pytest.approx(1.0)


def test_position_factor_default():
    from semblend_core.bathtub import position_factor

    assert position_factor(10, 32) == pytest.approx(0.75)


def test_recompute_config_force_layers():
    from semblend_core.bathtub import RecomputeConfig, compute_layer_deviations

    cfg = RecomputeConfig(force_recompute_layers=(10, 11, 12))
    devs = compute_layer_deviations(28, recompute_config=cfg)
    recompute = [d.layer_idx for d in devs if d.should_recompute]
    assert 10 in recompute
    assert 11 in recompute
    assert 12 in recompute


def test_recompute_config_skip_layers():
    from semblend_core.bathtub import RecomputeConfig, compute_layer_deviations

    # Force all layers to recompute via low threshold, then skip some
    cfg = RecomputeConfig(threshold=0.0, skip_recompute_layers=(0, 1, 2))
    devs = compute_layer_deviations(28, recompute_config=cfg)
    recompute = [d.layer_idx for d in devs if d.should_recompute]
    assert 0 not in recompute
    assert 1 not in recompute
    assert 2 not in recompute


def test_max_recompute_fraction_cap():
    from semblend_core.bathtub import RecomputeConfig, compute_layer_deviations

    cfg = RecomputeConfig(threshold=0.0, max_recompute_fraction=0.1)
    devs = compute_layer_deviations(28, recompute_config=cfg)
    n_recompute = sum(1 for d in devs if d.should_recompute)
    max_allowed = int(28 * 0.1)
    assert n_recompute <= max_allowed


def test_recompute_config_from_env(monkeypatch):
    from semblend_core.bathtub import RecomputeConfig

    monkeypatch.setenv("SEMBLEND_RECOMPUTE_THRESHOLD", "0.5")
    monkeypatch.setenv("SEMBLEND_FORCE_RECOMPUTE_LAYERS", "0,27")
    monkeypatch.setenv("SEMBLEND_MAX_RECOMPUTE_FRACTION", "0.15")
    cfg = RecomputeConfig.from_env()
    assert cfg.threshold == 0.5
    assert cfg.force_recompute_layers == (0, 27)
    assert cfg.max_recompute_fraction == 0.15


def test_force_verify_edge_layers_on_fuzzy():
    """With low fuzzy confidence, edge layers should be force-verified."""
    from semblend_core.bathtub import compute_layer_deviations

    devs = compute_layer_deviations(
        28,
        mismatch_fraction=0.05,
        model_name="qwen2.5-7b",
        similarity=0.9,  # high threshold -> few layers normally
        fuzzy_fraction=0.3,
        mean_fuzzy_confidence=0.85,
    )
    recompute = {d.layer_idx for d in devs if d.should_recompute}
    # Should include edge layers (0-4 and 24-27 for Qwen tau_e=3, tau_l=3)
    assert 0 in recompute or 27 in recompute


def test_bathtub_preset_has_new_fields():
    from semblend_core.bathtub import PRESETS

    for name, preset in PRESETS.items():
        assert hasattr(preset, "position_tau")
        assert hasattr(preset, "fuzzy_alpha")
        assert preset.position_tau > 0

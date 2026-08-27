"""Layer mask defaults off: measured on the reference ablation, per layer
zeroing caused most of the warm output drift (hit pairs scored ROUGE 0.38
with the mask on and 0.86 with every layer copied), so the safe default
is all copy. The mask stays available as an explicit opt in."""

from semblend.integration.sglang.config import SemBlendProviderConfig
from semblend.integration.sglang.provider import _layer_mask_enabled


def test_default_config_disables_the_layer_mask():
    assert SemBlendProviderConfig().enable_bathtub is False


def test_env_opt_in_reenables(monkeypatch):
    monkeypatch.delenv("SEMBLEND_LAYER_MASK", raising=False)
    assert _layer_mask_enabled(SemBlendProviderConfig()) is False

    monkeypatch.setenv("SEMBLEND_LAYER_MASK", "1")
    assert _layer_mask_enabled(SemBlendProviderConfig()) is True


def test_explicit_config_still_wins_without_env(monkeypatch):
    monkeypatch.delenv("SEMBLEND_LAYER_MASK", raising=False)
    assert _layer_mask_enabled(SemBlendProviderConfig(enable_bathtub=True)) is True

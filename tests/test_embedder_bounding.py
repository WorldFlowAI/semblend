"""SEMBLEND_EMBED_MAX_WINDOWS head-bounding (H8 registration-cost fix)."""

from semblend_core.embedder import MiniLMEmbedder, _bound_embed_text


def test_unset_is_identity(monkeypatch):
    monkeypatch.delenv("SEMBLEND_EMBED_MAX_WINDOWS", raising=False)
    assert _bound_embed_text("x" * 100_000) == "x" * 100_000


def test_bounds_to_window_chars(monkeypatch):
    monkeypatch.setenv("SEMBLEND_EMBED_MAX_WINDOWS", "2")
    out = _bound_embed_text("x" * 100_000)
    assert len(out) == 2 * 512 * 4


def test_invalid_and_nonpositive_are_identity(monkeypatch):
    for bad in ("abc", "0", "-3"):
        monkeypatch.setenv("SEMBLEND_EMBED_MAX_WINDOWS", bad)
        assert len(_bound_embed_text("x" * 50_000)) == 50_000


def test_bounded_embed_equals_embed_of_head(monkeypatch):
    embedder = MiniLMEmbedder()
    embedder._ensure_loaded()
    if not embedder.available:
        import pytest

        pytest.skip("MiniLM unavailable")
    long_text = " ".join(f"tok{i}" for i in range(6000))
    monkeypatch.setenv("SEMBLEND_EMBED_MAX_WINDOWS", "1")
    bounded = embedder.embed(long_text)
    monkeypatch.delenv("SEMBLEND_EMBED_MAX_WINDOWS")
    head_only = embedder.embed(long_text[: 512 * 4])
    import numpy as np

    assert np.allclose(bounded, head_only, atol=1e-6)

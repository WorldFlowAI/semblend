"""Verified paraphrase serve: high-similarity/low-coverage candidates
pass through the lexical fact gate; accepted spans serve donor KV as a
contiguous result, rejected or ungated candidates keep the previous
reject behavior."""

from types import SimpleNamespace

import pytest

from semblend.integration.sglang.provider import SemBlendProviderAdapter


def _adapter(monkeypatch, gate_on=True):
    if gate_on:
        monkeypatch.setenv("SEMBLEND_PARAPHRASE_SERVE", "1")
    else:
        monkeypatch.delenv("SEMBLEND_PARAPHRASE_SERVE", raising=False)
    a = SemBlendProviderAdapter.__new__(SemBlendProviderAdapter)
    a._config = SimpleNamespace(
        min_match_length=8,
        paraphrase_min_similarity=0.80,
        tail_reserve_frac=0.0,
    )
    a._tail_reserve_tokens = lambda n: n - 2
    return a


def _handle(text, n_kv=64):
    return SimpleNamespace(
        prompt_text=text,
        kv_indices=list(range(1000, 1000 + n_kv)),
        start_pos=0,
        end_pos=n_kv,
        last_node_id=7,
        extra_key=None,
        token_ids=list(range(n_kv)),
    )


def test_accepted_paraphrase_builds_contiguous_result(monkeypatch):
    a = _adapter(monkeypatch)
    h = _handle("The tally recorded exactly 47 approvals in Geneva.")
    remaining = list(range(500, 540))
    res = a._paraphrase_result(
        donor_id="d", handle=h, remaining=remaining, similarity=0.97
    )
    assert res is not None
    assert res.cached_token_count == 38  # 40 remaining - 2 tail reserve
    assert list(res.kv_cache_indices) == list(range(1000, 1038))
    assert res.cached_start_pos == 0
    assert res.donor_last_node_id == 7
    assert res.quality_signals.confidence_tier == "paraphrase_verified"
    assert res.segments is None  # contiguous contract


def test_short_window_returns_none(monkeypatch):
    a = _adapter(monkeypatch)
    h = _handle("text", n_kv=64)
    assert a._paraphrase_result(
        donor_id="d", handle=h, remaining=list(range(6)), similarity=0.97
    ) is None

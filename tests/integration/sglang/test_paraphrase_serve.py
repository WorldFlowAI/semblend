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
    a._nli_gate = None
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
    # One verified segment: the radix backend realizes segments into fresh
    # slots and drops a segment-less span that starts at the exact prefix.
    assert res.segments is not None and len(res.segments) == 1
    assert res.segments[0].length == res.cached_token_count
    assert res.segments[0].donor_positions[0] == 0


def test_short_window_returns_none(monkeypatch):
    a = _adapter(monkeypatch)
    h = _handle("text", n_kv=64)
    assert a._paraphrase_result(
        donor_id="d", handle=h, remaining=list(range(6)), similarity=0.97
    ) is None


class _FakeNliGate:
    def __init__(self, verdict):
        self.verdict = verdict
        self.calls = 0

    def spans_meaning_consistent(self, d, t):
        self.calls += 1
        return self.verdict


def test_nli_appeal_recovers_lexical_reject(monkeypatch):
    monkeypatch.setenv("SEMBLEND_NLI_APPEAL", "1")
    a = _adapter(monkeypatch)
    a._nli_gate = _FakeNliGate(True)
    assert a._nli_appeal("donor text", "target text") is True
    assert a._nli_gate.calls == 1


def test_nli_appeal_rejects_stay_rejected(monkeypatch):
    monkeypatch.setenv("SEMBLEND_NLI_APPEAL", "1")
    a = _adapter(monkeypatch)
    a._nli_gate = _FakeNliGate(False)
    assert a._nli_appeal("donor text", "target text") is False


def test_nli_appeal_fails_closed_on_gate_error(monkeypatch):
    monkeypatch.setenv("SEMBLEND_NLI_APPEAL", "1")
    a = _adapter(monkeypatch)

    class _Boom:
        def spans_meaning_consistent(self, d, t):
            raise RuntimeError("model unavailable")

    a._nli_gate = _Boom()
    assert a._nli_appeal("donor text", "target text") is False


def test_nli_gate_default_floors_pinned():
    """Operating point validated by sweep: strict NLI at these floors
    holds 100 percent divergence recall and full paraphrase accept-rate
    through the appeal composition, robust across the surveyed grid.
    The paraphrase-detection alternative was strictly worse (dropped
    recall at low floors). Changing a default means re-running the
    sweep, not editing this test."""
    from semblend_core.nli_gate import SentenceAlignedNliGate

    g = SentenceAlignedNliGate()
    assert g._entail_floor == 0.5
    assert g._align_floor == 0.2
    assert g._coverage_floor == 0.7
    assert "nli" in g._model_name

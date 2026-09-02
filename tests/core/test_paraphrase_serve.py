"""Engine-agnostic paraphrase-serve arbiter.

The arbiter composes the lexical fact gate with the sentence-aligned
NLI appeal as OR (never AND): the appeal rescues surface rewordings the
lexical gate cannot represent, and rescues zero fact-divergent spans.
Verdicts are memoized by span content so repeated donor/target pairs
never re-run the gates.
"""

from __future__ import annotations

import pytest

from semblend_core.paraphrase_serve import (
    ParaphraseArbiter,
    nli_appeal_enabled,
    paraphrase_serve_enabled,
)


class _CountingGate:
    def __init__(self, verdict: bool) -> None:
        self.verdict = verdict
        self.calls = 0

    def spans_meaning_consistent(self, donor_text: str, target_text: str) -> bool:
        self.calls += 1
        return self.verdict


class _RaisingGate:
    def spans_meaning_consistent(self, donor_text: str, target_text: str) -> bool:
        raise RuntimeError("model unavailable")


def test_env_gates_default_off(monkeypatch) -> None:
    monkeypatch.delenv("SEMBLEND_PARAPHRASE_SERVE", raising=False)
    monkeypatch.delenv("SEMBLEND_NLI_APPEAL", raising=False)
    assert paraphrase_serve_enabled() is False
    assert nli_appeal_enabled() is False
    monkeypatch.setenv("SEMBLEND_PARAPHRASE_SERVE", "1")
    monkeypatch.setenv("SEMBLEND_NLI_APPEAL", "1")
    assert paraphrase_serve_enabled() is True
    assert nli_appeal_enabled() is True


def test_fact_gate_acceptance_short_circuits_appeal() -> None:
    gate = _CountingGate(verdict=False)
    arbiter = ParaphraseArbiter(nli_gate=gate, nli_enabled=True)
    donor = "the quarterly report shows revenue grew across all regions"
    target = "revenue grew across all regions per the quarterly report"
    assert arbiter.verdict(donor, target) is True
    assert gate.calls == 0


def test_appeal_rescues_lexical_rejection() -> None:
    gate = _CountingGate(verdict=True)
    arbiter = ParaphraseArbiter(nli_gate=gate, nli_enabled=True)
    donor = "revenue reached 12 million in the quarter"
    target = "revenue reached twelve million in the quarter"
    assert arbiter.verdict(donor, target) is True
    assert gate.calls == 1


def test_appeal_denial_is_final() -> None:
    gate = _CountingGate(verdict=False)
    arbiter = ParaphraseArbiter(nli_gate=gate, nli_enabled=True)
    donor = "revenue reached 12 million in the quarter"
    target = "revenue reached 15 million in the quarter"
    assert arbiter.verdict(donor, target) is False
    assert gate.calls == 1


def test_fail_closed_without_appeal() -> None:
    arbiter = ParaphraseArbiter(nli_gate=None, nli_enabled=False)
    donor = "revenue reached 12 million in the quarter"
    target = "revenue reached twelve million in the quarter"
    assert arbiter.verdict(donor, target) is False


def test_fail_closed_when_appeal_raises() -> None:
    arbiter = ParaphraseArbiter(nli_gate=_RaisingGate(), nli_enabled=True)
    donor = "revenue reached 12 million in the quarter"
    target = "revenue reached twelve million in the quarter"
    assert arbiter.verdict(donor, target) is False


def test_verdict_memoized_by_span_content() -> None:
    gate = _CountingGate(verdict=True)
    arbiter = ParaphraseArbiter(nli_gate=gate, nli_enabled=True)
    donor = "revenue reached 12 million in the quarter"
    target = "revenue reached twelve million in the quarter"
    other = "revenue reached twelve million in the half"

    assert arbiter.verdict(donor, target) is True
    assert arbiter.verdict(donor, target) is True
    assert gate.calls == 1
    assert arbiter.verdict(donor, other) is True
    assert gate.calls == 2


def test_memo_capacity_bounded() -> None:
    gate = _CountingGate(verdict=True)
    arbiter = ParaphraseArbiter(nli_gate=gate, nli_enabled=True, memo_capacity=2)
    donor = "revenue reached 12 million in the quarter"
    targets = [f"revenue reached twelve million in period {i}" for i in range(3)]
    for t in targets:
        arbiter.verdict(donor, t)
    assert gate.calls == 3
    # Oldest entry evicted: re-asking it re-runs the gate.
    arbiter.verdict(donor, targets[0])
    assert gate.calls == 4
    # Newest entry still memoized.
    arbiter.verdict(donor, targets[2])
    assert gate.calls == 4


def test_empty_spans_rejected() -> None:
    arbiter = ParaphraseArbiter(nli_gate=_CountingGate(True), nli_enabled=True)
    assert arbiter.verdict("", "anything") is False
    assert arbiter.verdict("anything", "") is False


if __name__ == "__main__":
    pytest.main([__file__, "-q"])

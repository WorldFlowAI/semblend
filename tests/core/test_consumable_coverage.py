"""Consumable-coverage gate on alignment matches.

Token-level alignment (chunk_size 1, as the SGLang integration runs) can
report high reuse on a reordered paraphrase: most tokens appear on both
sides, but in runs too short to serve. Such a match must not be returned
as reuse. It must fall through to the paraphrase probe when that tier is
enabled, and to a miss otherwise.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from semblend_core.donor_store import DonorStore
from semblend_core.pipeline import PositionMapping, SemBlendPipeline, contiguous_coverage

DIM = 8
DONOR_TEXT = (
    "the migration plan moves every customer workload to the new region "
    "before the maintenance window closes and keeps replication running"
)
TARGET_TEXT = (
    "before the maintenance window closes the plan migrates all customer "
    "workloads to the new region with replication kept running"
)


def _identity(n, start=0):
    return PositionMapping(list(range(start, start + n)), list(range(start, start + n)))


def test_coverage_counts_only_runs_at_least_min_run():
    pmap = PositionMapping(
        donor_positions=[0, 1, 2, 3, 10, 12, 14, 16, 20, 21],
        target_positions=[0, 1, 2, 3, 10, 12, 14, 16, 20, 21],
    )
    # One run of 4 and one of 2 qualify at min_run=2; only nothing at min_run=8.
    assert contiguous_coverage(pmap, prompt_tokens=100, min_run=2) == 6 / 100
    assert contiguous_coverage(pmap, prompt_tokens=100, min_run=8) == 0.0


def test_coverage_of_identity_span_is_full():
    assert contiguous_coverage(_identity(64), prompt_tokens=64, min_run=16) == 1.0


def test_coverage_breaks_runs_on_either_side():
    pmap = PositionMapping(donor_positions=list(range(20)), target_positions=list(range(10)) + list(range(50, 60)))
    assert contiguous_coverage(pmap, prompt_tokens=100, min_run=8) == 20 / 100


class _StubEmbedder:
    dimension = DIM

    def embed(self, text):
        return np.ones(DIM, dtype=np.float32) / np.sqrt(DIM)


def _scattered_match(donor_tokens):
    """A store match claiming 80% token reuse in isolated single tokens."""
    n = len(donor_tokens)
    actions = [
        SimpleNamespace(action=SimpleNamespace(value="copy_from_donor"), target_pos=i, donor_pos=i)
        for i in range(0, n, 2)
    ]
    alignment = SimpleNamespace(
        reuse_ratio=0.8,
        slot_actions=actions,
        fuzzy_chunks=0,
        exact_chunks=len(actions),
        mean_fuzzy_confidence=1.0,
        chunk_confidences=(),
    )
    donor = SimpleNamespace(request_id="d1", token_ids=donor_tokens, prompt_text=DONOR_TEXT)
    return SimpleNamespace(donor=donor, similarity=0.93, alignment=alignment)


def _pipeline(monkeypatch):
    monkeypatch.setenv("SEMBLEND_CHUNK_FAST_PATH", "0")
    store = DonorStore(max_entries=8, embedding_dim=DIM, min_similarity=0.6, chunk_size=1)
    p = SemBlendPipeline(embedder_type="jaccard", donor_store=store, chunk_size=1, enable_pq_segments=False)
    p._embedder = _StubEmbedder()  # noqa: SLF001
    return p


def test_scattered_match_falls_through_to_paraphrase(monkeypatch):
    monkeypatch.setenv("SEMBLEND_PARAPHRASE_SERVE", "1")
    p = _pipeline(monkeypatch)
    donor_tokens = list(range(1000, 1200))
    p.register_donor("d1", donor_tokens, prompt_text=DONOR_TEXT)
    monkeypatch.setattr(p._donor_store, "find_donor", lambda **kw: _scattered_match(donor_tokens))  # noqa: SLF001

    result = p.find_donor(list(range(5000, 5200)), prompt_text=TARGET_TEXT)

    assert result.found is True
    assert result.confidence_tier == "paraphrase_verified"
    assert result.position_map.num_pairs == 192  # 200 - 8 tail reserve


def test_scattered_match_is_a_miss_without_paraphrase(monkeypatch):
    monkeypatch.delenv("SEMBLEND_PARAPHRASE_SERVE", raising=False)
    p = _pipeline(monkeypatch)
    donor_tokens = list(range(1000, 1200))
    p.register_donor("d1", donor_tokens, prompt_text=DONOR_TEXT)
    monkeypatch.setattr(p._donor_store, "find_donor", lambda **kw: _scattered_match(donor_tokens))  # noqa: SLF001

    result = p.find_donor(list(range(5000, 5200)), prompt_text=TARGET_TEXT)

    assert result.found is False
    assert result.rejection_reason == "low_consumable_coverage"


def test_contiguous_match_still_served_as_reuse(monkeypatch):
    monkeypatch.setenv("SEMBLEND_PARAPHRASE_SERVE", "1")
    p = _pipeline(monkeypatch)
    donor_tokens = list(range(1000, 1200))
    p.register_donor("d1", donor_tokens, prompt_text=DONOR_TEXT)
    n = len(donor_tokens)
    actions = [
        SimpleNamespace(action=SimpleNamespace(value="copy_from_donor"), target_pos=i, donor_pos=i)
        for i in range(n)
    ]
    match = _scattered_match(donor_tokens)
    match.alignment.slot_actions = actions
    match.alignment.reuse_ratio = 1.0
    monkeypatch.setattr(p._donor_store, "find_donor", lambda **kw: match)  # noqa: SLF001

    result = p.find_donor(list(donor_tokens), prompt_text=DONOR_TEXT)

    assert result.found is True
    assert result.confidence_tier == "exact"
    assert result.position_map.num_pairs == n

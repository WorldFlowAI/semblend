"""Identity-relaxed diagonal adoption (SEMBLEND_SPAN_MISMATCH_TOLERANCE).

A chunk whose hash misses (edited tokens) but whose neighbors match on a
consistent diagonal may be adopted from the same donor diagonal when its
mismatch fraction is within tolerance. Tolerance 0 keeps today's behavior
exactly; adopted mismatches are surfaced for analysis and leak auditing.
"""

import os

import pytest

from semblend_core.chunk_index import ChunkIndex
from semblend_core.multi_donor_alignment import compute_cdc_alignment


@pytest.fixture(autouse=True)
def _cdc_env(monkeypatch):
    monkeypatch.setenv("SEMBLEND_CDC_CHUNKS", "1")


def _pair(n=600, edit_at=300):
    base = [2000 + (i * 41) % 900 for i in range(n)]
    donor = list(base)
    target = list(base)
    target[edit_at] = 7  # one-token edit voids its chunk's hash
    return donor, target


def _align(donor, target, tolerance):
    os.environ["SEMBLEND_SPAN_MISMATCH_TOLERANCE"] = str(tolerance)
    try:
        idx = ChunkIndex()
        idx.add_donor_chunks("d1", donor)
        return compute_cdc_alignment(target, idx, {"d1": donor})
    finally:
        os.environ.pop("SEMBLEND_SPAN_MISMATCH_TOLERANCE", None)


def test_tolerance_zero_keeps_recompute_hole():
    donor, target = _pair()
    r = _align(donor, target, 0.0)
    assert r is not None
    assert r.recompute_chunks >= 1
    assert getattr(r, "tolerated_mismatch_tokens", 0) == 0


def test_tolerance_adopts_edited_chunk_on_diagonal():
    donor, target = _pair()
    r0 = _align(donor, target, 0.0)
    r = _align(donor, target, 0.05)
    assert r is not None
    assert r.recompute_chunks < r0.recompute_chunks  # hole was adopted
    assert r.tolerated_mismatch_tokens == 1
    assert r.mismatched_positions == ((307, 300),) or r.mismatched_positions == ((300 + 7, 300),) or len(r.mismatched_positions) == 1


def test_tolerance_below_density_still_recomputes():
    donor, target = _pair()
    # densify: 5 edits inside one region
    for k in range(5):
        target[300 + k] = 7 + k
    r = _align(donor, target, 0.01)  # 5 edits in a <=192-token chunk > 1%
    assert r is not None
    assert r.tolerated_mismatch_tokens == 0

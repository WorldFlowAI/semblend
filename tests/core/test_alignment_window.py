"""Fuzzy chunk matching is confined to a positional window; exact hash
matching still finds chunks that moved anywhere. The unbounded scan made
lookups quadratic (1.3 s per 8K-token lookup at chunk 16)."""

from __future__ import annotations

import random

from semblend_core import alignment as al


def _tokens(n, seed):
    rng = random.Random(seed)
    return [rng.randint(5, 30000) for _ in range(n)]


def test_shift_inside_window_still_fuzzy_matches():
    donor = _tokens(640, 1)
    target = donor[:37] + [1] + donor[37:]  # 1-token insertion shifts every later chunk by one
    res = al.compute_fuzzy_chunk_alignment(donor_tokens=donor, target_tokens=target, chunk_size=16)
    assert res.reuse_ratio > 0.5
    assert res.fuzzy_chunks > 0


def test_block_moved_far_is_found_by_exact_hash(monkeypatch):
    monkeypatch.setattr(al, "_FUZZY_CHUNK_WINDOW", 2)
    donor = _tokens(1600, 2)
    block = donor[64:320]  # 16 whole chunks
    target = _tokens(1024, 3) + block  # same block 60 chunks later
    res = al.compute_fuzzy_chunk_alignment(donor_tokens=donor, target_tokens=target, chunk_size=16)
    assert res.exact_chunks >= 16


def test_window_bounds_the_scan(monkeypatch):
    calls = []
    original = al._fuzzy_match_chunk

    def spy(*args, **kwargs):
        calls.append(kwargs.get("candidate_range"))
        return original(*args, **kwargs)

    monkeypatch.setattr(al, "_fuzzy_match_chunk", spy)
    donor = _tokens(320, 4)
    target = _tokens(320, 5)
    al.compute_fuzzy_chunk_alignment(donor_tokens=donor, target_tokens=target, chunk_size=16)
    assert calls and all(r is not None and r[1] - r[0] <= 2 * al._FUZZY_CHUNK_WINDOW + 1 for r in calls)

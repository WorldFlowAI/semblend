"""Tokenization-invariant alignment via canonical text space.

Matching happens on whitespace-collapsed text (reformat-invariant); matched
canonical spans are projected onto BOTH token sequences via offset maps,
and only resynced subruns survive: maximal stretches where donor and target
token ids agree exactly. Those subruns feed the existing token-identical
join machinery (re-RoPE joins, in-prefill recompute for the bubbles), so
consumption-side quality guarantees carry over unchanged.
"""

from __future__ import annotations

import bisect
from typing import List, Tuple

from semblend_core.chunk_index import cdc_boundaries

MIN_CANON_MATCH = 64  # canonical chars; below this, matches are noise


def canonicalize(text: str) -> Tuple[str, List[int]]:
    """Whitespace-collapse ``text``; return (canonical, canon->orig map)."""
    out: List[str] = []
    cmap: List[int] = []
    prev_space = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if not prev_space:
                out.append(" ")
                cmap.append(i)
            prev_space = True
        else:
            out.append(ch)
            cmap.append(i)
            prev_space = False
    # trailing collapsed separator is harmless; leading handled naturally
    return "".join(out), cmap


def _canonical_matches(donor_canon: str, target_canon: str):
    """Greedy anchored matching over canonical bytes.

    CDC boundaries over canonical bytes give reformat-invariant anchors;
    each anchor found in the donor is extended maximally in both
    directions (greedy diagonal, first-occurrence donor anchor).
    Returns non-overlapping (donor_start, target_start, length) in target
    order.
    """
    bounds = cdc_boundaries([ord(c) & 0xFFFF for c in target_canon])
    matches = []
    cursor = 0  # target canonical cursor
    for b0, b1 in bounds:
        if b1 - b0 < 16 or b0 < cursor:
            continue
        anchor = target_canon[b0:b1]
        d = donor_canon.find(anchor)
        if d < 0:
            continue
        # extend left
        left = 0
        while (
            d - left - 1 >= 0
            and b0 - left - 1 >= cursor
            and donor_canon[d - left - 1] == target_canon[b0 - left - 1]
        ):
            left += 1
        # extend right
        right = 0
        while (
            d + (b1 - b0) + right < len(donor_canon)
            and b1 + right < len(target_canon)
            and donor_canon[d + (b1 - b0) + right] == target_canon[b1 + right]
        ):
            right += 1
        start_d, start_t = d - left, b0 - left
        length = (b1 - b0) + left + right
        if length >= MIN_CANON_MATCH:
            matches.append((start_d, start_t, length))
            cursor = start_t + length
    return matches


def resynced_token_runs(
    donor_text: str,
    target_text: str,
    donor_ids_offsets,
    target_ids_offsets,
    min_run_tokens: int = 16,
) -> List[dict]:
    """Project canonical matches to token space; keep id-verified runs.

    ``*_ids_offsets`` are (token_ids, [(char_start, char_end), ...]) as
    produced by offset-mapping tokenizers. Returns runs of
    {donor_token_start, target_token_start, length, verified} where donor
    and target token ids agree exactly for the whole run.
    """
    d_ids, d_offs = donor_ids_offsets
    t_ids, t_offs = target_ids_offsets
    d_canon, d_map = canonicalize(donor_text)
    t_canon, t_map = canonicalize(target_text)

    def to_canon(cmap, orig_pos):
        return bisect.bisect_left(cmap, orig_pos)

    d_start_by_canon = {}
    for k, (o0, _o1) in enumerate(d_offs):
        d_start_by_canon.setdefault(to_canon(d_map, o0), k)

    runs: List[dict] = []
    for d0, t0, length in _canonical_matches(d_canon, t_canon):
        run_start = None
        run_len = 0
        k = None
        for tk, (o0, _o1) in enumerate(t_offs):
            c = to_canon(t_map, o0)
            if not (t0 <= c < t0 + length):
                continue
            dk = d_start_by_canon.get(c - t0 + d0)
            aligned = dk is not None and d_ids[dk] == t_ids[tk]
            contiguous = (
                run_start is not None
                and k is not None
                and tk == k + 1
                and aligned
                and dk == run_start["donor_token_start"] + run_len
            )
            if contiguous:
                run_len += 1
            else:
                if run_start is not None and run_len >= min_run_tokens:
                    runs.append({**run_start, "length": run_len, "verified": True})
                if aligned:
                    run_start = {
                        "donor_token_start": dk,
                        "target_token_start": tk,
                    }
                    run_len = 1
                else:
                    run_start = None
                    run_len = 0
            k = tk
        if run_start is not None and run_len >= min_run_tokens:
            runs.append({**run_start, "length": run_len, "verified": True})
    return runs

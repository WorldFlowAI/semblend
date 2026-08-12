"""Extract paired-KV training positions from near-duplicate pairs.

Aligns donor/target token sequences with the CDC + diagonal machinery and
emits divergence records at 1:1 substitution positions (equal-length gaps
between matched runs): the positions where a learned imputation model would
predict the target's K/V from the donor's. Unequal gaps (insertions or
deletions) are emitted as excluded_gap records and skipped by training.
"""

from __future__ import annotations

from typing import List

from semblend_core.chunk_index import ChunkIndex
from semblend_core.multi_donor_alignment import compute_cdc_alignment

MIN_FLANK_RUN = 8  # a substitution needs matched context on both sides


def _matched_runs(donor_positions, target_positions):
    """Group aligned (donor, target) pairs into +1/+1 runs."""
    runs = []
    if not donor_positions:
        return runs
    start = 0
    for i in range(1, len(donor_positions) + 1):
        if (
            i == len(donor_positions)
            or donor_positions[i] - donor_positions[i - 1] != 1
            or target_positions[i] - target_positions[i - 1] != 1
        ):
            runs.append(
                (
                    donor_positions[start],
                    target_positions[start],
                    i - start,
                )
            )
            start = i
    return runs


def extract_divergence_records(
    donor_tokens: List[int],
    target_tokens: List[int],
    donor_id: str,
) -> List[dict]:
    """Align one pair and emit substitution / excluded_gap records."""
    import os

    prev = os.environ.get("SEMBLEND_CDC_CHUNKS")
    os.environ["SEMBLEND_CDC_CHUNKS"] = "1"
    try:
        index = ChunkIndex()
        index.add_donor_chunks(donor_id, donor_tokens)
        result = compute_cdc_alignment(
            target_tokens, index, {donor_id: donor_tokens}
        )
    finally:
        if prev is None:
            os.environ.pop("SEMBLEND_CDC_CHUNKS", None)
        else:
            os.environ["SEMBLEND_CDC_CHUNKS"] = prev
    if result is None:
        return []

    pmap = result.composite_plan.position_map
    runs = _matched_runs(
        list(pmap.donor_positions), list(pmap.target_positions)
    )
    records: List[dict] = []
    for idx in range(len(runs) - 1):
        d0, t0, len0 = runs[idx]
        d1, t1, _len1 = runs[idx + 1]
        d_gap = d1 - (d0 + len0)
        t_gap = t1 - (t0 + len0)
        if d_gap <= 0 or t_gap <= 0:
            continue
        if d_gap != t_gap or len0 < MIN_FLANK_RUN:
            records.append(
                {
                    "kind": "excluded_gap",
                    "donor_id": donor_id,
                    "donor_pos": d0 + len0,
                    "target_pos": t0 + len0,
                    "donor_gap": d_gap,
                    "target_gap": t_gap,
                }
            )
            continue
        # The aligner is chunk-granular: one edited token voids its whole
        # chunk. Refine the equal-length gap token-by-token and emit only
        # positions whose tokens actually differ.
        for k in range(d_gap):
            d_tok = donor_tokens[d0 + len0 + k]
            t_tok = target_tokens[t0 + len0 + k]
            if d_tok == t_tok:
                continue
            records.append(
                {
                    "kind": "substitution",
                    "donor_id": donor_id,
                    "donor_pos": d0 + len0 + k,
                    "target_pos": t0 + len0 + k,
                    "donor_tok": d_tok,
                    "target_tok": t_tok,
                    "run_before": len0,
                    "run_after": runs[idx + 1][2],
                }
            )
    return records

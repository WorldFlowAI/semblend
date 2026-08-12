"""Sparse-plan construction: order gated segments into novel/donor spans.

The plan is the contract between alignment and the engine's plan-steered
chunked prefill: donor spans are consumed as boundary-anchored contiguous
joins between chunks (prefill skipped), novel spans are computed in-prefill
attending all earlier KV — which is what makes corrections propagate
(in-prefill recompute), unlike post-hoc slot swapping (measured).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence

from semblend.integration.sglang.types import FuzzyMatchSegment


def sparse_plan_enabled() -> bool:
    """Feature gate: sparse-plan emission is opt-in (SEMBLEND_SPARSE_PLAN=1)."""
    return os.environ.get("SEMBLEND_SPARSE_PLAN", "") == "1"


@dataclass(frozen=True)
class SparsePlanSpan:
    """One span of the remaining-prompt window (tail-relative positions)."""

    kind: str  # "novel" | "donor"
    target_start: int  # inclusive
    target_end: int  # exclusive
    donor_start: Optional[int] = None  # donor-side start (donor spans only)
    segment_index: Optional[int] = None  # index into the segments list


def _first_int(seq) -> int:
    return int(seq[0]) if len(seq) else 0


def build_sparse_plan(
    segments: Sequence[FuzzyMatchSegment],
    remaining_len: int,
    min_donor_span: int = 16,
    edge_shave: int = 0,
    gap_period: int = 0,
    gap_size: int = 64,
) -> Optional[List[SparsePlanSpan]]:
    """Order segments into an alternating novel/donor cover of the window.

    Segments are sorted by target start; a segment overlapping an earlier
    one is dropped (gates should prevent this — dropping is the safe
    resolution). Donor spans shorter than ``min_donor_span`` fold into the
    surrounding novel span (a chunk split costs a dispatch; tiny spans are
    not worth it). Spans reaching past the window are clamped.

    ``edge_shave`` moves each donor span's first/last N tokens into the
    surrounding novel spans: span-edge KV carries the strongest donor-side
    context contamination, and shaved tokens are recomputed in-prefill
    attending the joined interior (a quality lever).

    Returns None when no donor span survives — callers fall back to the
    non-sparse contract.
    """
    if remaining_len <= 0:
        return None

    indexed = sorted(
        ((idx, seg) for idx, seg in enumerate(segments)),
        key=lambda pair: _first_int(pair[1].target_positions),
    )

    donor_spans: List[SparsePlanSpan] = []
    cursor = 0
    for idx, seg in indexed:
        targets = seg.target_positions
        length = seg.length if seg.length is not None else len(targets)
        start = _first_int(targets) + edge_shave
        end = min(_first_int(targets) + int(length) - edge_shave, remaining_len)
        if start < cursor:  # overlap with an earlier donor span
            continue
        if end - start < min_donor_span:
            continue
        if start >= remaining_len:
            continue
        donor_spans.append(
            SparsePlanSpan(
                kind="donor",
                target_start=start,
                target_end=end,
                donor_start=_first_int(seg.donor_positions) + edge_shave,
                segment_index=idx,
            )
        )
        cursor = end

    if not donor_spans:
        return None

    if gap_period > 0:
        # H20: interleave small in-prefill recompute gaps inside long donor
        # spans. The gap tokens fall into the novel cover below and are
        # computed attending the joined KV on both sides, refreshing the
        # span-wide context contamination that edge-only recomputation
        # (refuted) cannot reach.
        split: List[SparsePlanSpan] = []
        for span in donor_spans:
            pos = span.target_start
            while span.target_end - pos > gap_period + gap_size:
                split.append(
                    SparsePlanSpan(
                        kind="donor",
                        target_start=pos,
                        target_end=pos + gap_period,
                        donor_start=span.donor_start + (pos - span.target_start),
                        segment_index=span.segment_index,
                    )
                )
                pos += gap_period + gap_size
            if span.target_end - pos >= min_donor_span:
                split.append(
                    SparsePlanSpan(
                        kind="donor",
                        target_start=pos,
                        target_end=span.target_end,
                        donor_start=span.donor_start + (pos - span.target_start),
                        segment_index=span.segment_index,
                    )
                )
        donor_spans = split
        if not donor_spans:
            return None

    plan: List[SparsePlanSpan] = []
    pos = 0
    for donor in donor_spans:
        if donor.target_start > pos:
            plan.append(
                SparsePlanSpan(
                    kind="novel", target_start=pos, target_end=donor.target_start
                )
            )
        plan.append(donor)
        pos = donor.target_end
    if pos < remaining_len:
        plan.append(
            SparsePlanSpan(kind="novel", target_start=pos, target_end=remaining_len)
        )
    return plan

"""Sparse-plan construction: order gated segments into novel/donor spans.

The plan is the H18 contract between alignment and the engine's plan-steered
chunked prefill: donor spans are consumed as boundary-anchored contiguous
joins between chunks (prefill skipped), novel spans are computed in-prefill
attending all earlier KV — which is what makes corrections propagate
(in-prefill recompute); post-hoc cache substitution lacks this property.
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
    attending the joined interior (H19 quality lever).

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

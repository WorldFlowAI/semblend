"""KV rearrangement plan: map donor KV positions to target positions.

Converts TokenMatchResult into scatter instructions + RoPE delta corrections
that the vLLM connector uses to load and rearrange donor KV from LMCache.
"""
from __future__ import annotations

from dataclasses import dataclass

from semblend_core.semshare.token_matcher import TokenMatchResult


@dataclass(frozen=True)
class ScatterInstruction:
    """Single scatter: copy donor KV at donor_pos to target_pos."""

    target_pos: int
    donor_pos: int
    rope_delta: int  # target_pos - donor_pos (for RoPE correction)
    similarity: float


@dataclass(frozen=True)
class KVRearrangePlan:
    """Complete plan for rearranging donor KV into target order."""

    # Scatter instructions sorted by target position
    scatter: tuple[ScatterInstruction, ...]

    # Positions that must be fully recomputed (no donor match)
    recompute_positions: tuple[int, ...]

    # Donor ID to load KV from
    donor_id: str

    # Target sequence length
    target_seq_len: int

    # Donor sequence length
    donor_seq_len: int

    # Fraction of target covered by donor KV
    coverage: float

    # Whether RoPE correction is needed (any rope_delta != 0)
    needs_rope_correction: bool

    @property
    def num_scatter(self) -> int:
        return len(self.scatter)

    @property
    def num_recompute(self) -> int:
        return len(self.recompute_positions)


def build_rearrange_plan(
    match_result: TokenMatchResult,
    target_seq_len: int,
    donor_seq_len: int,
) -> KVRearrangePlan:
    """Build a KV rearrangement plan from token match results.

    Args:
        match_result: result from token_matcher.match_tokens()
        target_seq_len: length of target sequence
        donor_seq_len: length of donor sequence

    Returns:
        KVRearrangePlan with scatter instructions and recompute positions
    """
    instructions: list[ScatterInstruction] = []
    has_rope_correction = False

    for target_pos, donor_pos, similarity in match_result.matched_pairs:
        rope_delta = target_pos - donor_pos
        if rope_delta != 0:
            has_rope_correction = True

        instructions.append(
            ScatterInstruction(
                target_pos=target_pos,
                donor_pos=donor_pos,
                rope_delta=rope_delta,
                similarity=similarity,
            )
        )

    # Sort by target position for sequential memory access
    instructions.sort(key=lambda x: x.target_pos)

    coverage = len(instructions) / target_seq_len if target_seq_len > 0 else 0.0

    return KVRearrangePlan(
        scatter=tuple(instructions),
        recompute_positions=match_result.unmatched_target,
        donor_id=match_result.donor_id,
        target_seq_len=target_seq_len,
        donor_seq_len=donor_seq_len,
        coverage=coverage,
        needs_rope_correction=has_rope_correction,
    )

"""Multi-donor composite KV injection data structures.

Frozen immutable types for multi-donor alignment, scatter-gather KV
assembly, and donor-aware RoPE position correction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class MatchType(Enum):
    """How a target chunk was matched to a donor."""

    EXACT = "exact"  # Hash-matched (O(1) ChunkIndex lookup)
    FUZZY = "fuzzy"  # Token-overlap matched (PQ ADC fallback)
    RECOMPUTE = "recompute"  # No match — fresh prefill needed


@dataclass(frozen=True)
class ChunkAssignment:
    """Maps a target chunk to its best donor source.

    Produced by multi_donor_alignment for each target chunk.
    """

    target_chunk_idx: int
    donor_id: str | None = None
    donor_chunk_idx: int | None = None
    match_type: MatchType = MatchType.RECOMPUTE
    confidence: float = 0.0


@dataclass(frozen=True)
class MultiDonorSlotAction:
    """SlotAction extended with donor_id for multi-donor scatter.

    Unlike the single-donor SlotAction, this carries the donor_id
    so the connector knows WHICH donor's KV to load for each position.
    """

    action: str  # "copy_from_donor" | "recompute"
    target_pos: int
    donor_pos: int | None = None
    donor_id: str | None = None


@dataclass(frozen=True)
class MultiDonorPositionMapping:
    """Donor-aware RoPE correction mapping for composite KV injection.

    Each entry maps a target position to its donor source position
    and the donor that owns the KV. RoPE correction applies
    delta = target_pos - donor_pos per position.
    """

    donor_ids: tuple[str, ...] = ()
    donor_positions: tuple[int, ...] = ()
    target_positions: tuple[int, ...] = ()

    @property
    def num_pairs(self) -> int:
        return len(self.target_positions)

    @property
    def needs_correction(self) -> bool:
        return any(d != t for d, t in zip(self.donor_positions, self.target_positions))

    def for_donor(self, donor_id: str) -> "MultiDonorPositionMapping":
        """Filter to positions from a specific donor."""
        indices = [i for i, did in enumerate(self.donor_ids) if did == donor_id]
        return MultiDonorPositionMapping(
            donor_ids=tuple(self.donor_ids[i] for i in indices),
            donor_positions=tuple(self.donor_positions[i] for i in indices),
            target_positions=tuple(self.target_positions[i] for i in indices),
        )


@dataclass(frozen=True)
class CompositeKVPlan:
    """Scatter-gather plan for multi-donor KV assembly.

    Tells the connector exactly which donors to load KV from,
    which positions to scatter them to, and which positions need
    fresh computation.

    Attributes:
        donor_ids: Unique donor IDs involved (ordered by contribution).
        chunk_assignments: Per-chunk donor mapping.
        slot_actions: Per-position donor-aware actions.
        position_map: Donor-aware RoPE correction.
        total_reuse_ratio: Combined reuse ratio across all donors.
        donors_per_composite: Number of distinct donors contributing.
    """

    donor_ids: tuple[str, ...] = ()
    chunk_assignments: tuple[ChunkAssignment, ...] = ()
    slot_actions: tuple[MultiDonorSlotAction, ...] = ()
    position_map: MultiDonorPositionMapping = field(
        default_factory=MultiDonorPositionMapping,
    )
    total_reuse_ratio: float = 0.0
    donors_per_composite: int = 0

    def actions_for_donor(self, donor_id: str) -> list[MultiDonorSlotAction]:
        """Get all slot actions for a specific donor."""
        return [
            sa
            for sa in self.slot_actions
            if sa.donor_id == donor_id and sa.action == "copy_from_donor"
        ]

    def recompute_positions(self) -> list[int]:
        """Get all positions needing fresh computation."""
        return [sa.target_pos for sa in self.slot_actions if sa.action == "recompute"]


@dataclass(frozen=True)
class MultiDonorAlignmentResult:
    """Result of multi-donor chunk alignment.

    Extends single-donor AlignmentResult with per-chunk donor
    assignments and composite KV plan.
    """

    reuse_ratio: float
    chunk_assignments: tuple[ChunkAssignment, ...]
    composite_plan: CompositeKVPlan
    donor_ids: tuple[str, ...]
    exact_chunks: int = 0
    fuzzy_chunks: int = 0
    recompute_chunks: int = 0
    chunk_index_hits: int = 0  # Chunks found via ChunkIndex (skipped embedding)

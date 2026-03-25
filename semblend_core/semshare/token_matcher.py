"""Token-level matching: convert LSH candidates into a rearrangement plan.

Implements greedy bipartite matching — each donor token used at most once.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from semblend_core.semshare.config import SemShareConfig
from semblend_core.semshare.lsh_index import LSHIndex, LSHMatch


@dataclass(frozen=True)
class TokenMatchResult:
    """Result of token-level matching between target and donor."""

    # Each triple: (target_pos, donor_pos, similarity)
    matched_pairs: tuple[tuple[int, int, float], ...]

    # Target positions with no donor match (must be recomputed)
    unmatched_target: tuple[int, ...]

    # Best donor ID (most matches)
    donor_id: str

    # Fraction of target tokens matched
    match_ratio: float

    # Per-token similarity scores (0.0 for unmatched)
    similarities: tuple[float, ...]


def match_tokens(
    target_embeddings: np.ndarray,
    lsh_index: LSHIndex,
    config: SemShareConfig,
    candidate_donor_ids: list[str] | None = None,
) -> TokenMatchResult | None:
    """Match target tokens to donor tokens via LSH, then greedy assign.

    Args:
        target_embeddings: shape [seq_len, embed_dim]
        lsh_index: populated LSH index with donors
        config: SemShareKV configuration
        candidate_donor_ids: optional filter

    Returns:
        TokenMatchResult if match_ratio >= config.min_match_ratio, else None
    """
    if lsh_index.num_donors == 0:
        return None

    # Query LSH for candidates per target token
    all_matches = lsh_index.query(target_embeddings, candidate_donor_ids)

    # Count matches per donor to find the best donor
    donor_votes: dict[str, int] = {}
    for token_matches in all_matches:
        seen_donors: set[str] = set()
        for m in token_matches:
            if m.donor_id not in seen_donors:
                donor_votes[m.donor_id] = donor_votes.get(m.donor_id, 0) + 1
                seen_donors.add(m.donor_id)

    if not donor_votes:
        return None

    best_donor_id = max(donor_votes, key=lambda d: donor_votes[d])

    # Greedy bipartite matching against best donor
    # Each donor position used at most once
    used_donor_positions: set[int] = set()
    matched_pairs: list[tuple[int, int, float]] = []
    unmatched: list[int] = []
    similarities: list[float] = []

    for target_pos, token_matches in enumerate(all_matches):
        best_match: LSHMatch | None = None
        for m in token_matches:
            if m.donor_id != best_donor_id:
                continue
            if m.donor_pos in used_donor_positions:
                continue
            best_match = m
            break

        if best_match is not None:
            used_donor_positions.add(best_match.donor_pos)
            matched_pairs.append(
                (target_pos, best_match.donor_pos, best_match.similarity)
            )
            similarities.append(best_match.similarity)
        else:
            unmatched.append(target_pos)
            similarities.append(0.0)

    seq_len = target_embeddings.shape[0]
    match_ratio = len(matched_pairs) / seq_len if seq_len > 0 else 0.0

    if match_ratio < config.min_match_ratio:
        return None

    return TokenMatchResult(
        matched_pairs=tuple(matched_pairs),
        unmatched_target=tuple(unmatched),
        donor_id=best_donor_id,
        match_ratio=match_ratio,
        similarities=tuple(similarities),
    )

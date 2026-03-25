"""Multi-donor composite alignment for cross-donor KV assembly.

For each target chunk:
  1. O(1) hash lookup in ChunkIndex → exact match from any donor
  2. If no exact: PQ ADC top-k → fuzzy match against best candidate
  3. If no match: RECOMPUTE

Context gate: adjacent chunk must also match (from any donor) to
prevent semantic staleness from isolated cross-donor matches.

Feature flag: SEMBLEND_MULTI_DONOR=1 to enable.
"""
from __future__ import annotations

import logging
import os
from collections import Counter

from semblend_core.alignment import (
    DEFAULT_CHUNK_SIZE,
    _fuzzy_match_chunk,
)
from semblend_core.chunk_index import ChunkIndex
from semblend_core.multi_donor_types import (
    ChunkAssignment,
    CompositeKVPlan,
    MatchType,
    MultiDonorAlignmentResult,
    MultiDonorPositionMapping,
    MultiDonorSlotAction,
)

logger = logging.getLogger(__name__)

_CONTEXT_GATE_ENABLED = os.environ.get("SEMBLEND_CONTEXT_GATE", "1") != "0"


def _get_chunk_embeddings(
    target_text: str,
    unmatched_indices: list[int],
    chunk_size: int,
    embedder: object | None,
) -> object | None:
    """Get per-chunk embeddings for PQ semantic matching.

    Splits the target text into ~chunk-sized text segments and embeds each
    one with MiniLM. Returns [N, dim] L2-normalized embeddings for the
    unmatched chunk positions.

    The text segmentation is approximate (by character count) since we
    don't have a tokenizer here. Each chunk ≈ chunk_size tokens ≈
    chunk_size * 4 characters.
    """
    if not target_text or embedder is None:
        return None

    try:
        import numpy as np

        chars_per_chunk = chunk_size * 4  # ~4 chars per token
        embeddings = []

        for t_idx in unmatched_indices:
            char_start = t_idx * chars_per_chunk
            char_end = char_start + chars_per_chunk
            chunk_text = target_text[char_start:char_end]

            if not chunk_text.strip():
                dim = getattr(embedder, "dimension", 384)
                embeddings.append(np.zeros(dim, dtype=np.float32))
                continue

            raw = embedder.embed(chunk_text)
            if raw is not None:
                vec = np.asarray(raw, dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                embeddings.append(vec)
            else:
                dim = getattr(embedder, "dimension", 384)
                embeddings.append(np.zeros(dim, dtype=np.float32))

        if not embeddings:
            return None

        return np.stack(embeddings, axis=0)

    except Exception as e:
        logger.debug("Failed to get chunk embeddings: %s", e)
        return None


def compute_multi_donor_alignment(
    target_tokens: list[int],
    chunk_index: ChunkIndex,
    donor_token_store: dict[str, list[int]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    context_gate: bool | None = None,
    min_fuzzy_overlap: float = 0.90,
    pq_store: object | None = None,
    target_text: str = "",
    embedder: object | None = None,
    token_index: object | None = None,
) -> MultiDonorAlignmentResult | None:
    """Compute multi-donor chunk alignment using ChunkIndex.

    For each target chunk, finds the best donor via:
      1. ChunkIndex exact hash match (any donor)
      2. Fuzzy token-overlap match if no exact (against all donors)
      3. RECOMPUTE if neither

    Args:
        target_tokens: Target token sequence.
        chunk_index: Global ChunkIndex with all donors indexed.
        donor_token_store: donor_id → token_ids mapping for fuzzy fallback.
        chunk_size: Engine chunk size.
        context_gate: Override for context gate. None uses env var.
        min_fuzzy_overlap: Minimum token overlap for fuzzy match.
        pq_store: Optional PQSegmentStore for fuzzy verification.

    Returns:
        MultiDonorAlignmentResult or None if no useful matches found.
    """
    use_context_gate = (
        context_gate if context_gate is not None else _CONTEXT_GATE_ENABLED
    )

    # Split target into chunks
    target_chunks: list[list[int]] = []
    for i in range(0, len(target_tokens), chunk_size):
        target_chunks.append(target_tokens[i:i + chunk_size])

    num_target_chunks = len(target_chunks)

    # Phase 1: ChunkIndex exact hash matches (O(1) per chunk)
    assignments: dict[int, ChunkAssignment] = {}
    used_donor_chunks: dict[str, set[int]] = {}  # donor_id → set of used chunk_idx
    chunk_index_hits = 0

    for t_idx, t_chunk in enumerate(target_chunks):
        if len(t_chunk) != chunk_size:
            continue

        locations = chunk_index.lookup_chunk(t_chunk)
        if not locations:
            continue

        # Pick first available location (prefer donors with more matches)
        for loc in locations:
            used = used_donor_chunks.get(loc.donor_id, set())
            if loc.chunk_idx not in used:
                assignments[t_idx] = ChunkAssignment(
                    target_chunk_idx=t_idx,
                    donor_id=loc.donor_id,
                    donor_chunk_idx=loc.chunk_idx,
                    match_type=MatchType.EXACT,
                    confidence=1.0,
                )
                used_donor_chunks.setdefault(loc.donor_id, set()).add(
                    loc.chunk_idx
                )
                chunk_index_hits += 1
                break

    # Phase 1.5: PQ semantic chunk matching for unmatched chunks
    # Uses per-chunk embeddings to find semantically similar chunks across
    # ALL donors, not just exact hash or token-overlap matches.
    semantic_assignments: dict[int, ChunkAssignment] = {}
    semantic_chunk_hits = 0

    unmatched_indices = [
        t_idx for t_idx in range(num_target_chunks)
        if t_idx not in assignments and len(target_chunks[t_idx]) == chunk_size
    ]

    if unmatched_indices and pq_store is not None:
        # Get query segment embeddings for unmatched chunks
        # We need the embedder to produce per-chunk embeddings
        try:
            from semblend_core.pq_segment_store import PQSegmentStore
            if isinstance(pq_store, PQSegmentStore) and pq_store.size > 0:
                # Try to get pre-computed segment embeddings for target chunks
                # If not available, we need to embed them on the fly
                if hasattr(pq_store, 'find_best_donor_per_chunk'):
                    # Build query segment embeddings from target text chunks
                    # We need the embedder — check if one is available via pipeline
                    query_seg_embeddings = _get_chunk_embeddings(
                        target_text, unmatched_indices, chunk_size, embedder,
                    )
                    if query_seg_embeddings is not None:
                        matches = pq_store.find_best_donor_per_chunk(
                            query_seg_embeddings,
                            min_similarity=0.85,
                        )
                        for i, match in enumerate(matches):
                            if match is not None:
                                donor_id, donor_chunk_idx, sim = match
                                t_idx = unmatched_indices[i]
                                semantic_assignments[t_idx] = ChunkAssignment(
                                    target_chunk_idx=t_idx,
                                    donor_id=donor_id,
                                    donor_chunk_idx=donor_chunk_idx,
                                    match_type=MatchType.FUZZY,
                                    confidence=sim,
                                )
                                used_donor_chunks.setdefault(
                                    donor_id, set()
                                ).add(donor_chunk_idx)
                                semantic_chunk_hits += 1

                        if semantic_chunk_hits > 0:
                            logger.info(
                                "PQ semantic matching: %d/%d chunks matched "
                                "across donors",
                                semantic_chunk_hits, len(unmatched_indices),
                            )
        except Exception as e:
            logger.debug("PQ semantic matching failed: %s", e)

    # Update unmatched indices after semantic matching
    unmatched_indices = [
        t_idx for t_idx in range(num_target_chunks)
        if t_idx not in assignments
        and t_idx not in semantic_assignments
        and len(target_chunks[t_idx]) == chunk_size
    ]

    # Phase 2: Fuzzy token-overlap matching for remaining unmatched chunks
    fuzzy_assignments: dict[int, ChunkAssignment] = {}

    if unmatched_indices:
        # Collect candidate donors for fuzzy matching
        candidate_donor_ids = set(used_donor_chunks.keys())

        # Include donors from neighbor matches
        for t_idx in unmatched_indices:
            for neighbor in (t_idx - 1, t_idx + 1):
                if neighbor in assignments or neighbor in semantic_assignments:
                    asgn = assignments.get(neighbor) or semantic_assignments.get(neighbor)
                    if asgn and asgn.donor_id:
                        candidate_donor_ids.add(asgn.donor_id)

        # CRITICAL: when no exact/semantic matches found, use TokenIndex
        # to find donor chunks with high token overlap (O(chunk_size) per chunk).
        if not candidate_donor_ids and token_index is not None:
            from semblend_core.token_index import TokenIndex
            if isinstance(token_index, TokenIndex):
                # For each target chunk, find candidate donor chunks via TokenIndex
                for t_idx in unmatched_indices:
                    t_chunk = target_chunks[t_idx]
                    candidates = token_index.find_fuzzy_candidates(t_chunk)
                    for ref, shared_count in candidates[:5]:  # Top 5 per chunk
                        candidate_donor_ids.add(ref.donor_id)

                if candidate_donor_ids:
                    logger.info(
                        "multi_donor fuzzy: TokenIndex found %d candidate donors "
                        "for %d unmatched chunks",
                        len(candidate_donor_ids), len(unmatched_indices),
                    )

        # Fallback: brute-force Jaccard if TokenIndex unavailable
        if not candidate_donor_ids and donor_token_store:
            target_set = set(target_tokens)
            scored = []
            for did, d_tokens in donor_token_store.items():
                donor_set = set(d_tokens)
                inter = len(target_set & donor_set)
                union = len(target_set | donor_set)
                jacc = inter / union if union > 0 else 0
                if jacc > 0.15:
                    scored.append((jacc, did))
            scored.sort(reverse=True)
            candidate_donor_ids = {did for _, did in scored[:20]}

        donor_chunk_cache: dict[str, tuple[list[list[int]], list[int]]] = {}
        for donor_id in candidate_donor_ids:
            donor_tokens = donor_token_store.get(donor_id)
            if donor_tokens is None:
                continue
            d_chunks: list[list[int]] = []
            d_starts: list[int] = []
            for j in range(0, len(donor_tokens), chunk_size):
                d_chunks.append(donor_tokens[j:j + chunk_size])
                d_starts.append(j)
            donor_chunk_cache[donor_id] = (d_chunks, d_starts)

        # For multi-donor, use a lower fuzzy threshold (0.70 vs 0.90)
        # because shifted chunk boundaries reduce per-chunk overlap even
        # when the underlying content is identical.
        multi_donor_fuzzy_overlap = max(0.70, min_fuzzy_overlap - 0.20)
        logger.info(
            "multi_donor fuzzy: checking %d unmatched chunks against %d donors "
            "(threshold=%.2f)",
            len(unmatched_indices), len(donor_chunk_cache), multi_donor_fuzzy_overlap,
        )

        for t_idx in unmatched_indices:
            t_chunk = target_chunks[t_idx]
            best_match: ChunkAssignment | None = None
            best_overlap = 0.0

            for donor_id, (donor_chunks, donor_chunk_starts) in donor_chunk_cache.items():
                used = used_donor_chunks.get(donor_id, set())
                result = _fuzzy_match_chunk(
                    t_chunk, donor_chunks, donor_chunk_starts,
                    used, multi_donor_fuzzy_overlap,
                )
                if result is not None:
                    d_idx, pairs = result
                    t_counts = Counter(t_chunk)
                    d_counts = Counter(donor_chunks[d_idx])
                    overlap_count = sum(
                        min(t_counts[tok], d_counts[tok])
                        for tok in t_counts if tok in d_counts
                    )
                    overlap = overlap_count / max(len(t_chunk), 1)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_match = ChunkAssignment(
                            target_chunk_idx=t_idx,
                            donor_id=donor_id,
                            donor_chunk_idx=d_idx,
                            match_type=MatchType.FUZZY,
                            confidence=overlap,
                        )

            if best_match is not None and best_match.donor_id is not None:
                fuzzy_assignments[t_idx] = best_match
                used_donor_chunks.setdefault(
                    best_match.donor_id, set()
                ).add(best_match.donor_chunk_idx)

    # Merge assignments (exact > semantic > fuzzy priority)
    all_assignments = {**fuzzy_assignments, **semantic_assignments, **assignments}

    # Phase 3: Context gate — reject isolated matches
    if use_context_gate and all_assignments:
        matched_set = set(all_assignments.keys())
        validated: dict[int, ChunkAssignment] = {}
        rejected = 0

        for t_idx, asgn in all_assignments.items():
            has_neighbor = (
                (t_idx - 1) in matched_set
                or (t_idx + 1) in matched_set
            )
            if has_neighbor:
                validated[t_idx] = asgn
            else:
                rejected += 1
                logger.debug(
                    "multi_donor context_gate: rejected isolated chunk %d "
                    "from donor %s",
                    t_idx, asgn.donor_id,
                )

        if rejected > 0:
            logger.info(
                "multi_donor context_gate: rejected %d/%d isolated matches",
                rejected, len(all_assignments),
            )
        all_assignments = validated

    if not all_assignments:
        logger.info(
            "multi_donor_alignment: no assignments after all phases "
            "(exact=%d, semantic=%d, fuzzy=%d, total_chunks=%d)",
            len(assignments), len(semantic_assignments), len(fuzzy_assignments),
            num_target_chunks,
        )
        return None

    # Phase 4: Build composite plan
    return _build_composite_result(
        target_tokens=target_tokens,
        target_chunks=target_chunks,
        assignments=all_assignments,
        donor_token_store=donor_token_store,
        chunk_size=chunk_size,
        chunk_index_hits=chunk_index_hits,
    )


def _build_composite_result(
    target_tokens: list[int],
    target_chunks: list[list[int]],
    assignments: dict[int, ChunkAssignment],
    donor_token_store: dict[str, list[int]],
    chunk_size: int,
    chunk_index_hits: int,
) -> MultiDonorAlignmentResult:
    """Build the composite plan from chunk assignments."""
    # Collect unique donor IDs
    donor_ids_set: set[str] = set()
    for asgn in assignments.values():
        if asgn.donor_id:
            donor_ids_set.add(asgn.donor_id)
    donor_ids = tuple(sorted(donor_ids_set))

    # Build per-position slot actions and position mapping
    slot_actions: list[MultiDonorSlotAction] = []
    map_donor_ids: list[str] = []
    map_donor_positions: list[int] = []
    map_target_positions: list[int] = []

    num_reused = 0
    exact_chunks = 0
    fuzzy_chunks = 0
    recompute_chunks = 0

    # Fill in chunk assignments for all chunks (including RECOMPUTE)
    full_assignments: list[ChunkAssignment] = []

    for t_idx, t_chunk in enumerate(target_chunks):
        t_start = t_idx * chunk_size
        asgn = assignments.get(t_idx)

        if asgn is not None and asgn.donor_id is not None:
            d_start = asgn.donor_chunk_idx * chunk_size
            full_assignments.append(asgn)

            if asgn.match_type == MatchType.EXACT:
                exact_chunks += 1
                for i in range(len(t_chunk)):
                    if t_start + i < len(target_tokens):
                        slot_actions.append(MultiDonorSlotAction(
                            action="copy_from_donor",
                            target_pos=t_start + i,
                            donor_pos=d_start + i,
                            donor_id=asgn.donor_id,
                        ))
                        map_donor_ids.append(asgn.donor_id)
                        map_donor_positions.append(d_start + i)
                        map_target_positions.append(t_start + i)
                        num_reused += 1

            elif asgn.match_type == MatchType.FUZZY:
                fuzzy_chunks += 1
                # For fuzzy matches, use position-aligned copy
                # (simplified: copy all tokens at aligned positions)
                donor_tokens = donor_token_store.get(asgn.donor_id, [])
                d_chunk = donor_tokens[d_start:d_start + chunk_size]

                for i in range(len(t_chunk)):
                    if t_start + i < len(target_tokens):
                        if i < len(d_chunk) and t_chunk[i] == d_chunk[i]:
                            slot_actions.append(MultiDonorSlotAction(
                                action="copy_from_donor",
                                target_pos=t_start + i,
                                donor_pos=d_start + i,
                                donor_id=asgn.donor_id,
                            ))
                            map_donor_ids.append(asgn.donor_id)
                            map_donor_positions.append(d_start + i)
                            map_target_positions.append(t_start + i)
                            num_reused += 1
                        else:
                            slot_actions.append(MultiDonorSlotAction(
                                action="recompute",
                                target_pos=t_start + i,
                            ))
        else:
            recompute_chunks += 1
            full_assignments.append(ChunkAssignment(
                target_chunk_idx=t_idx,
                match_type=MatchType.RECOMPUTE,
            ))
            for i in range(len(t_chunk)):
                if t_start + i < len(target_tokens):
                    slot_actions.append(MultiDonorSlotAction(
                        action="recompute",
                        target_pos=t_start + i,
                    ))

    target_len = len(target_tokens)
    reuse_ratio = num_reused / max(target_len, 1)

    position_map = MultiDonorPositionMapping(
        donor_ids=tuple(map_donor_ids),
        donor_positions=tuple(map_donor_positions),
        target_positions=tuple(map_target_positions),
    )

    composite_plan = CompositeKVPlan(
        donor_ids=donor_ids,
        chunk_assignments=tuple(full_assignments),
        slot_actions=tuple(slot_actions),
        position_map=position_map,
        total_reuse_ratio=reuse_ratio,
        donors_per_composite=len(donor_ids),
    )

    logger.info(
        "multi_donor_alignment: %d exact + %d fuzzy + %d recompute chunks, "
        "reuse=%.3f, donors=%d, chunk_index_hits=%d",
        exact_chunks, fuzzy_chunks, recompute_chunks,
        reuse_ratio, len(donor_ids), chunk_index_hits,
    )

    return MultiDonorAlignmentResult(
        reuse_ratio=reuse_ratio,
        chunk_assignments=tuple(full_assignments),
        composite_plan=composite_plan,
        donor_ids=donor_ids,
        exact_chunks=exact_chunks,
        fuzzy_chunks=fuzzy_chunks,
        recompute_chunks=recompute_chunks,
        chunk_index_hits=chunk_index_hits,
    )

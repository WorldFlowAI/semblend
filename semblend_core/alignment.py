"""In-process token alignment for SemBlend donor KV reuse.

Three alignment strategies:
  1. Chunk-level alignment (primary): Splits sequences into engine-sized chunks
     and matches by content hash. Handles REORDER scenarios where identical
     chunks appear in different positions. O(n) time.
  2. Fuzzy chunk alignment: When exact chunk hash fails, checks token overlap
     within chunks. If overlap >= threshold (default 0.90), treats the chunk
     as a fuzzy match with per-token donor_pos mapping. Produces Δ≠0 for
     shifted tokens, activating RoPE correction. O(n*c) time where c=chunk_size.
  3. Levenshtein alignment (fallback): Uses rapidfuzz edit distance for
     fine-grained token-level alignment when chunk matching is insufficient.

The chunk size is parameterized to support different backends:
  - LMCache (vLLM): 256 tokens per chunk
  - TRT-LLM: 128 tokens per chunk (configurable power-of-2)

Context gate (enabled by default): rejects isolated chunk matches where
no adjacent chunk also matches. Prevents semantic staleness from token-
identical chunks at wrong semantic positions.
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# Default chunk size — overridden by backend's get_kv_block_size()
DEFAULT_CHUNK_SIZE = 256
# Backward-compatible alias used by semblend_connector.py
LMCACHE_CHUNK_SIZE = DEFAULT_CHUNK_SIZE

# Context gate: reject isolated chunk matches where no adjacent chunk also
# matches. Prevents semantic staleness (PPL=1.27 outlier) from token-identical
# chunks appearing at wrong semantic positions in the document.
# Disable with SEMBLEND_CONTEXT_GATE=0 for backward compatibility.
_CONTEXT_GATE_ENABLED = os.environ.get("SEMBLEND_CONTEXT_GATE", "1") != "0"

# Fuzzy chunk matching: when exact hash fails, check token overlap.
# Enable with SEMBLEND_FUZZY_CHUNKS=1. Default threshold 0.90 (90% overlap).
_FUZZY_CHUNKS_ENABLED = os.environ.get("SEMBLEND_FUZZY_CHUNKS", "1") == "1"
_FUZZY_CHUNK_MIN_OVERLAP = float(
    os.environ.get("SEMBLEND_FUZZY_CHUNK_OVERLAP", "0.90")
)


@dataclass(frozen=True)
class FuzzyMatchConfig:
    """Fully configurable fuzzy matching parameters for any model/engine."""
    min_overlap: float = float(os.environ.get("SEMBLEND_FUZZY_CHUNK_OVERLAP", "0.90"))
    decay_function: str = os.environ.get("SEMBLEND_FUZZY_DECAY_FN", "exponential")
    position_tau: float = float(os.environ.get("SEMBLEND_FUZZY_POSITION_TAU", "128"))
    confidence_high: float = float(os.environ.get("SEMBLEND_FUZZY_CONFIDENCE_HIGH", "0.92"))
    confidence_low: float = float(os.environ.get("SEMBLEND_FUZZY_CONFIDENCE_LOW", "0.80"))
    bag_cosine_min: float = float(os.environ.get("SEMBLEND_FUZZY_BAG_COSINE_MIN", "0.94"))
    bag_cosine_adaptive: bool = os.environ.get("SEMBLEND_FUZZY_BAG_COSINE_ADAPTIVE", "1") == "1"
    segment_verify: bool = os.environ.get("SEMBLEND_FUZZY_SEGMENT_VERIFY", "1") == "1"
    segment_similarity_min: float = float(os.environ.get("SEMBLEND_FUZZY_SEGMENT_SIM_MIN", "0.85"))


@dataclass(frozen=True)
class ChunkConfidence:
    """Per-chunk confidence metadata for fuzzy matches."""
    chunk_idx: int
    overlap_ratio: float
    positional_coherence: float
    mean_abs_delta: float
    bag_cosine: float
    segment_similarity: float
    confidence: float
    tier: str  # "fast_reuse" | "verified_reuse" | "recompute"


# Default fuzzy config (auto-populates from env vars)
_DEFAULT_FUZZY_CONFIG = FuzzyMatchConfig()


try:
    from rapidfuzz.distance import Opcodes  # noqa: F401
    from rapidfuzz.distance.Levenshtein import opcodes as lev_opcodes
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False
    logger.warning("rapidfuzz not available - alignment disabled")


class SlotActionType(Enum):
    COPY_FROM_DONOR = "copy_from_donor"
    RECOMPUTE = "recompute"


@dataclass(frozen=True)
class SlotAction:
    action: SlotActionType
    target_pos: int
    donor_pos: int | None = None


@dataclass(frozen=True)
class AlignmentResult:
    reuse_ratio: float
    slot_actions: list[SlotAction]
    edit_distance: int
    donor_len: int
    target_len: int
    fuzzy_chunks: int = 0
    exact_chunks: int = 0
    chunk_confidences: tuple[ChunkConfidence, ...] = ()
    mean_fuzzy_confidence: float = 1.0
    fuzzy_recompute_chunks: int = 0


def _chunk_hash(tokens: list[int]) -> str:
    """Hash a chunk of token IDs for matching.

    Uses full 4-byte representation of each token ID to avoid collisions
    with large vocabularies (e.g., Qwen 152K vocab).
    """
    import struct
    data = struct.pack(f"<{len(tokens)}I", *tokens)
    return hashlib.md5(data).hexdigest()


def compute_chunk_alignment(
    donor_tokens: list[int],
    target_tokens: list[int],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    context_gate: bool | None = None,
) -> AlignmentResult:
    """Chunk-level alignment for KV reuse.

    Splits both sequences into fixed-size chunks (matching the engine's KV
    storage granularity) and matches by content hash. Handles REORDER scenarios
    where identical chunks appear at different positions.

    Two-phase approach:
      Phase 1: Identify all hash matches between donor and target chunks.
      Phase 2 (context gate): Reject isolated matches — a chunk match is
        accepted only if at least one adjacent target chunk also matches.
        This prevents semantic staleness where a token-identical chunk from
        a different document position gets incorrectly reused.

    Args:
        donor_tokens: Donor token sequence.
        target_tokens: Target token sequence.
        chunk_size: Chunk size in tokens (must match engine block size).
        context_gate: Override for context gate. None uses env var default.

    Returns:
        AlignmentResult with chunk-aligned slot actions.
    """
    use_context_gate = (
        context_gate if context_gate is not None else _CONTEXT_GATE_ENABLED
    )

    # Split into chunks
    donor_chunks = []
    for i in range(0, len(donor_tokens), chunk_size):
        donor_chunks.append(donor_tokens[i:i + chunk_size])

    target_chunks = []
    for i in range(0, len(target_tokens), chunk_size):
        target_chunks.append(target_tokens[i:i + chunk_size])

    # Build donor chunk hash → list of chunk indices
    # Only full-size chunks can match (partial trailing chunks can't)
    donor_hash_map: dict[str, list[int]] = {}
    for idx, chunk in enumerate(donor_chunks):
        if len(chunk) == chunk_size:
            h = _chunk_hash(chunk)
            donor_hash_map.setdefault(h, []).append(idx)

    # Phase 1: identify all hash matches
    chunk_matches: dict[int, int] = {}  # target_idx -> donor_idx
    used_donor_chunks: set[int] = set()

    for t_idx, t_chunk in enumerate(target_chunks):
        if len(t_chunk) == chunk_size:
            h = _chunk_hash(t_chunk)
            candidates = donor_hash_map.get(h, [])
            for d_idx in candidates:
                if d_idx not in used_donor_chunks:
                    chunk_matches[t_idx] = d_idx
                    used_donor_chunks.add(d_idx)
                    break

    # Phase 2: context gate — reject isolated matches
    if use_context_gate and chunk_matches:
        matched_set = set(chunk_matches.keys())
        validated: dict[int, int] = {}
        rejected_count = 0
        for t_idx, d_idx in chunk_matches.items():
            has_neighbor = (
                (t_idx - 1) in matched_set
                or (t_idx + 1) in matched_set
            )
            if has_neighbor:
                validated[t_idx] = d_idx
            else:
                rejected_count += 1
                logger.info(
                    "context_gate: rejected isolated chunk match "
                    "target[%d] -> donor[%d]",
                    t_idx, d_idx,
                )
        if rejected_count > 0:
            logger.info(
                "context_gate: rejected %d/%d isolated chunk matches",
                rejected_count, len(chunk_matches),
            )
        chunk_matches = validated

    # Phase 3: build slot actions from validated matches
    slot_actions: list[SlotAction] = []
    num_reused = 0

    for t_idx, t_chunk in enumerate(target_chunks):
        t_start = t_idx * chunk_size
        d_idx = chunk_matches.get(t_idx)

        if d_idx is not None:
            d_start = d_idx * chunk_size
            for i in range(chunk_size):
                slot_actions.append(SlotAction(
                    action=SlotActionType.COPY_FROM_DONOR,
                    target_pos=t_start + i,
                    donor_pos=d_start + i,
                ))
                num_reused += 1
        else:
            for i in range(len(t_chunk)):
                slot_actions.append(SlotAction(
                    action=SlotActionType.RECOMPUTE,
                    target_pos=t_start + i,
                ))

    target_len = len(target_tokens)
    reuse_ratio = num_reused / max(target_len, 1)
    edit_dist = target_len - num_reused

    matched_chunks = len(chunk_matches)
    total_target_chunks = sum(1 for c in target_chunks if len(c) == chunk_size)
    logger.info(
        "chunk_alignment: %d/%d chunks matched (reuse=%.2f), "
        "donor_chunks=%d, target_chunks=%d",
        matched_chunks, total_target_chunks, reuse_ratio,
        len(donor_chunks), len(target_chunks),
    )

    return AlignmentResult(
        reuse_ratio=reuse_ratio,
        slot_actions=slot_actions,
        edit_distance=edit_dist,
        donor_len=len(donor_tokens),
        target_len=target_len,
        exact_chunks=matched_chunks,
    )


def chunk_bag_cosine(donor_chunk: list[int], target_chunk: list[int]) -> float:
    """Token-frequency weighted cosine similarity between two chunks.

    Uses Counter-based token frequency vectors. Fast (~0.01ms per chunk)
    and captures distribution shape beyond simple overlap ratio.
    """
    from collections import Counter
    d_counts = Counter(donor_chunk)
    t_counts = Counter(target_chunk)
    vocab = set(d_counts.keys()) | set(t_counts.keys())
    dot = sum(d_counts.get(v, 0) * t_counts.get(v, 0) for v in vocab)
    d_norm = sum(c * c for c in d_counts.values()) ** 0.5
    t_norm = sum(c * c for c in t_counts.values()) ** 0.5
    return dot / max(d_norm * t_norm, 1e-12)


def _compute_position_decay(mean_abs_delta: float, tau: float, decay_fn: str) -> float:
    """Compute position decay for confidence scoring."""
    if decay_fn == "exponential":
        return math.exp(-mean_abs_delta / max(tau, 1e-6))
    elif decay_fn == "linear":
        return max(0.0, 1.0 - mean_abs_delta / (2.0 * max(tau, 1e-6)))
    elif decay_fn == "step":
        return 1.0 if mean_abs_delta < tau else 0.0
    return math.exp(-mean_abs_delta / max(tau, 1e-6))


def _compute_chunk_confidence(
    pairs: list[tuple[int, int]],
    overlap_ratio: float,
    donor_chunk: list[int],
    target_chunk: list[int],
    chunk_idx: int,
    config: FuzzyMatchConfig,
    global_similarity: float = 0.0,
) -> ChunkConfidence:
    """Compute confidence score and tier for a fuzzy-matched chunk."""
    from collections import Counter

    # Positional coherence: fraction of pairs sharing the mode delta
    if pairs:
        deltas = [t_off - d_off for (t_off, d_off) in pairs]
        delta_counts = Counter(deltas)
        mode_delta_count = delta_counts.most_common(1)[0][1]
        positional_coherence = mode_delta_count / len(pairs)
        abs_deltas = [abs(d) for d in deltas]
        mean_abs_delta = sum(abs_deltas) / len(abs_deltas)
    else:
        positional_coherence = 0.0
        mean_abs_delta = 0.0

    # Position decay
    decay = _compute_position_decay(mean_abs_delta, config.position_tau, config.decay_function)

    # Bag-cosine verification
    bag_cos = chunk_bag_cosine(donor_chunk, target_chunk)

    # Composite confidence
    confidence = overlap_ratio * positional_coherence * decay

    # Bag-cosine threshold (adaptive if enabled)
    bag_threshold = config.bag_cosine_min
    if config.bag_cosine_adaptive and global_similarity > 0:
        bag_threshold = config.bag_cosine_min + 0.04 * global_similarity

    # Tier classification
    if bag_cos < bag_threshold:
        tier = "recompute"
        confidence = 0.0  # Force recompute if bag-cosine fails
    elif confidence >= config.confidence_high:
        tier = "fast_reuse"
    elif confidence >= config.confidence_low:
        tier = "verified_reuse"
    else:
        tier = "recompute"

    return ChunkConfidence(
        chunk_idx=chunk_idx,
        overlap_ratio=overlap_ratio,
        positional_coherence=positional_coherence,
        mean_abs_delta=mean_abs_delta,
        bag_cosine=bag_cos,
        segment_similarity=0.0,  # Set later by PQ store if available
        confidence=confidence,
        tier=tier,
    )


def _fuzzy_match_chunk(
    target_chunk: list[int],
    donor_chunks: list[list[int]],
    donor_chunk_starts: list[int],
    used_donor_chunks: set[int],
    min_overlap: float,
) -> tuple[int, list[tuple[int, int]]] | None:
    """Find the best fuzzy-matching donor chunk for a target chunk.

    Computes token-set overlap between the target chunk and each unused
    donor chunk. Returns the best match if overlap >= min_overlap.

    Args:
        target_chunk: Token IDs in the target chunk.
        donor_chunks: All donor chunks (list of token ID lists).
        donor_chunk_starts: Start position of each donor chunk.
        used_donor_chunks: Set of already-used donor chunk indices.
        min_overlap: Minimum fraction of target tokens that must match.

    Returns:
        (donor_chunk_idx, [(target_pos, donor_pos), ...]) for matched
        tokens, or None if no chunk meets the overlap threshold.
    """
    target_len = len(target_chunk)
    if target_len == 0:
        return None

    # Build target token multiset
    from collections import Counter
    target_counts = Counter(target_chunk)

    best_idx: int | None = None
    best_overlap = 0

    for d_idx, d_chunk in enumerate(donor_chunks):
        if d_idx in used_donor_chunks:
            continue
        if len(d_chunk) != target_len:
            continue

        # Fast overlap check: count matching tokens (multiset intersection)
        donor_counts = Counter(d_chunk)
        overlap_count = sum(
            min(target_counts[tok], donor_counts[tok])
            for tok in target_counts
            if tok in donor_counts
        )
        overlap_ratio = overlap_count / target_len

        if overlap_ratio >= min_overlap and overlap_ratio > best_overlap:
            best_idx = d_idx
            best_overlap = overlap_ratio

    if best_idx is None:
        return None

    # Build per-token alignment for the best-matching donor chunk.
    # For the common shifted-prefix case, tokens are identical but offset.
    # Use greedy positional matching: prefer same-position matches first.
    d_chunk = donor_chunks[best_idx]
    donor_chunk_starts[best_idx]
    pairs: list[tuple[int, int]] = []

    # Phase 1: match tokens at same relative offset (zero-delta preferred)
    matched_target: set[int] = set()
    matched_donor: set[int] = set()
    for i in range(target_len):
        if target_chunk[i] == d_chunk[i]:
            pairs.append((i, i))
            matched_target.add(i)
            matched_donor.add(i)

    # Phase 2: match remaining tokens greedily by token ID
    from collections import defaultdict
    remaining_donor: dict[int, list[int]] = defaultdict(list)
    for i in range(target_len):
        if i not in matched_donor:
            remaining_donor[d_chunk[i]].append(i)

    for i in range(target_len):
        if i in matched_target:
            continue
        tok = target_chunk[i]
        candidates = remaining_donor.get(tok)
        if candidates:
            d_offset = candidates.pop(0)
            pairs.append((i, d_offset))
            matched_target.add(i)

    return (best_idx, pairs)


def compute_fuzzy_chunk_alignment(
    donor_tokens: list[int],
    target_tokens: list[int],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    context_gate: bool | None = None,
    min_overlap: float | None = None,
    fuzzy_config: FuzzyMatchConfig | None = None,
    global_similarity: float = 0.0,
) -> AlignmentResult:
    """Chunk-level alignment with fuzzy matching for non-exact chunks.

    Extends compute_chunk_alignment with a fuzzy matching phase: when a
    target chunk has no exact hash match, checks token-set overlap against
    all unused donor chunks. If overlap >= min_overlap, the chunk is
    treated as a fuzzy match with per-token (donor_pos, target_pos)
    mapping that may produce Δ≠0, activating RoPE correction.

    This recovers reuse in shifted-prefix scenarios where instruction
    length differences shift all 256-token chunk boundaries, causing
    0% exact chunk reuse despite 99%+ token-level overlap.

    Args:
        donor_tokens: Donor token sequence.
        target_tokens: Target token sequence.
        chunk_size: Chunk size in tokens (must match engine block size).
        context_gate: Override for context gate. None uses env var default.
        min_overlap: Minimum token overlap fraction for fuzzy match.
            None uses SEMBLEND_FUZZY_CHUNK_OVERLAP env var (default 0.90).

    Returns:
        AlignmentResult with exact + fuzzy chunk-aligned slot actions.
    """
    use_context_gate = (
        context_gate if context_gate is not None else _CONTEXT_GATE_ENABLED
    )
    overlap_threshold = (
        min_overlap if min_overlap is not None else _FUZZY_CHUNK_MIN_OVERLAP
    )

    # Split into chunks
    donor_chunks: list[list[int]] = []
    donor_chunk_starts: list[int] = []
    for i in range(0, len(donor_tokens), chunk_size):
        donor_chunks.append(donor_tokens[i:i + chunk_size])
        donor_chunk_starts.append(i)

    target_chunks: list[list[int]] = []
    for i in range(0, len(target_tokens), chunk_size):
        target_chunks.append(target_tokens[i:i + chunk_size])

    # Build donor chunk hash map (exact matching)
    donor_hash_map: dict[str, list[int]] = {}
    for idx, chunk in enumerate(donor_chunks):
        if len(chunk) == chunk_size:
            h = _chunk_hash(chunk)
            donor_hash_map.setdefault(h, []).append(idx)

    # Phase 1: exact hash matches
    exact_matches: dict[int, int] = {}  # target_idx -> donor_idx
    used_donor_chunks: set[int] = set()

    for t_idx, t_chunk in enumerate(target_chunks):
        if len(t_chunk) == chunk_size:
            h = _chunk_hash(t_chunk)
            candidates = donor_hash_map.get(h, [])
            for d_idx in candidates:
                if d_idx not in used_donor_chunks:
                    exact_matches[t_idx] = d_idx
                    used_donor_chunks.add(d_idx)
                    break

    # Phase 2: fuzzy matches for unmatched target chunks
    fuzzy_matches: dict[int, list[tuple[int, int]]] = {}
    fuzzy_donor_idx: dict[int, int] = {}

    for t_idx, t_chunk in enumerate(target_chunks):
        if t_idx in exact_matches:
            continue
        if len(t_chunk) != chunk_size:
            continue

        result = _fuzzy_match_chunk(
            t_chunk, donor_chunks, donor_chunk_starts,
            used_donor_chunks, overlap_threshold,
        )
        if result is not None:
            d_idx, pairs = result
            fuzzy_matches[t_idx] = pairs
            fuzzy_donor_idx[t_idx] = d_idx
            used_donor_chunks.add(d_idx)

    # Merge exact + fuzzy matches for context gate
    all_matched = set(exact_matches.keys()) | set(fuzzy_matches.keys())

    # Phase 3: context gate — reject isolated matches
    if use_context_gate and all_matched:
        validated_exact: dict[int, int] = {}
        validated_fuzzy: dict[int, list[tuple[int, int]]] = {}
        rejected_count = 0

        for t_idx in all_matched:
            has_neighbor = (
                (t_idx - 1) in all_matched
                or (t_idx + 1) in all_matched
            )
            if has_neighbor:
                if t_idx in exact_matches:
                    validated_exact[t_idx] = exact_matches[t_idx]
                if t_idx in fuzzy_matches:
                    validated_fuzzy[t_idx] = fuzzy_matches[t_idx]
            else:
                rejected_count += 1

        if rejected_count > 0:
            logger.info(
                "context_gate: rejected %d/%d isolated chunk matches",
                rejected_count, len(all_matched),
            )
        exact_matches = validated_exact
        fuzzy_matches = validated_fuzzy

    # Phase 3.5: confidence gating for fuzzy matches
    config = fuzzy_config or _DEFAULT_FUZZY_CONFIG
    chunk_confidences: list[ChunkConfidence] = []
    downgraded_fuzzy: set[int] = set()

    for t_idx in list(fuzzy_matches.keys()):
        pairs = fuzzy_matches[t_idx]
        d_idx = fuzzy_donor_idx[t_idx]
        t_chunk = target_chunks[t_idx]
        d_chunk = donor_chunks[d_idx]

        # Compute overlap ratio for this specific chunk
        from collections import Counter
        t_counts = Counter(t_chunk)
        d_counts = Counter(d_chunk)
        overlap_count = sum(
            min(t_counts[tok], d_counts[tok])
            for tok in t_counts if tok in d_counts
        )
        overlap = overlap_count / max(len(t_chunk), 1)

        conf = _compute_chunk_confidence(
            pairs=pairs,
            overlap_ratio=overlap,
            donor_chunk=d_chunk,
            target_chunk=t_chunk,
            chunk_idx=t_idx,
            config=config,
            global_similarity=global_similarity,
        )
        chunk_confidences.append(conf)

        if conf.tier == "recompute":
            downgraded_fuzzy.add(t_idx)

    # Remove downgraded chunks from fuzzy matches
    for t_idx in downgraded_fuzzy:
        del fuzzy_matches[t_idx]
        del fuzzy_donor_idx[t_idx]

    num_downgraded = len(downgraded_fuzzy)
    if num_downgraded > 0:
        logger.info(
            "confidence_gate: downgraded %d fuzzy chunks to recompute",
            num_downgraded,
        )

    # Phase 4: build slot actions
    slot_actions: list[SlotAction] = []
    num_reused = 0
    num_exact_chunks = len(exact_matches)
    num_fuzzy_chunks = len(fuzzy_matches)

    for t_idx, t_chunk in enumerate(target_chunks):
        t_start = t_idx * chunk_size

        if t_idx in exact_matches:
            # Exact match: all tokens copy from donor at same relative offset
            d_idx = exact_matches[t_idx]
            d_start = d_idx * chunk_size
            for i in range(chunk_size):
                slot_actions.append(SlotAction(
                    action=SlotActionType.COPY_FROM_DONOR,
                    target_pos=t_start + i,
                    donor_pos=d_start + i,
                ))
                num_reused += 1

        elif t_idx in fuzzy_matches:
            # Fuzzy match: per-token alignment within the chunk
            d_idx = fuzzy_donor_idx[t_idx]
            d_start = d_idx * chunk_size
            pairs = fuzzy_matches[t_idx]
            matched_offsets = {p[0] for p in pairs}

            for i in range(len(t_chunk)):
                if i in matched_offsets:
                    # Find the donor offset for this target offset
                    donor_offset = next(
                        p[1] for p in pairs if p[0] == i
                    )
                    slot_actions.append(SlotAction(
                        action=SlotActionType.COPY_FROM_DONOR,
                        target_pos=t_start + i,
                        donor_pos=d_start + donor_offset,
                    ))
                    num_reused += 1
                else:
                    slot_actions.append(SlotAction(
                        action=SlotActionType.RECOMPUTE,
                        target_pos=t_start + i,
                    ))
        else:
            # No match: recompute all tokens in this chunk
            for i in range(len(t_chunk)):
                slot_actions.append(SlotAction(
                    action=SlotActionType.RECOMPUTE,
                    target_pos=t_start + i,
                ))

    target_len = len(target_tokens)
    reuse_ratio = num_reused / max(target_len, 1)
    edit_dist = target_len - num_reused

    total_target_chunks = sum(1 for c in target_chunks if len(c) == chunk_size)
    logger.info(
        "fuzzy_chunk_alignment: %d exact + %d fuzzy / %d chunks "
        "(reuse=%.3f), donor_chunks=%d, target_chunks=%d",
        num_exact_chunks, num_fuzzy_chunks, total_target_chunks,
        reuse_ratio, len(donor_chunks), len(target_chunks),
    )

    # Compute mean fuzzy confidence (only for non-downgraded chunks)
    active_confidences = [c for c in chunk_confidences if c.tier != "recompute"]
    mean_conf = (
        sum(c.confidence for c in active_confidences) / len(active_confidences)
        if active_confidences else 1.0
    )

    return AlignmentResult(
        reuse_ratio=reuse_ratio,
        slot_actions=slot_actions,
        edit_distance=edit_dist,
        donor_len=len(donor_tokens),
        target_len=target_len,
        fuzzy_chunks=num_fuzzy_chunks,
        exact_chunks=num_exact_chunks,
        chunk_confidences=tuple(chunk_confidences),
        mean_fuzzy_confidence=mean_conf,
        fuzzy_recompute_chunks=num_downgraded,
    )


def estimate_reuse_ratio(
    donor_tokens: list[int],
    target_tokens: list[int],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> float:
    """Fast O(n) reuse ratio estimation without building slot_actions.

    Used by donor_store candidate scoring to avoid creating thousands of
    SlotAction objects per candidate. Only the winning candidate needs
    the full alignment with slot_actions.

    Strategy: chunk alignment count, then token-set count as fallback.
    """
    import struct

    # Try chunk-level matching first
    donor_hashes: dict[str, int] = {}
    for i in range(0, len(donor_tokens), chunk_size):
        chunk = donor_tokens[i:i + chunk_size]
        if len(chunk) == chunk_size:
            data = struct.pack(f"<{len(chunk)}I", *chunk)
            h = hashlib.md5(data).hexdigest()
            donor_hashes[h] = donor_hashes.get(h, 0) + 1

    matched_tokens = 0
    for i in range(0, len(target_tokens), chunk_size):
        chunk = target_tokens[i:i + chunk_size]
        if len(chunk) == chunk_size:
            data = struct.pack(f"<{len(chunk)}I", *chunk)
            h = hashlib.md5(data).hexdigest()
            if donor_hashes.get(h, 0) > 0:
                donor_hashes[h] -= 1
                matched_tokens += chunk_size

    if matched_tokens > 0:
        return matched_tokens / max(len(target_tokens), 1)

    # Fallback: token-set overlap count (O(n), no SlotAction creation)
    from collections import Counter
    donor_counts = Counter(donor_tokens)
    for tok in target_tokens:
        if donor_counts.get(tok, 0) > 0:
            donor_counts[tok] -= 1
            matched_tokens += 1

    return matched_tokens / max(len(target_tokens), 1)


def compute_alignment(
    donor_tokens: list[int],
    target_tokens: list[int],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    fuzzy: bool | None = None,
) -> AlignmentResult:
    """Compute alignment between donor and target sequences.

    Uses chunk-level alignment (primary) which correctly handles REORDER
    scenarios. When fuzzy matching is enabled, non-exact chunks are matched
    by token overlap (recovering shifted-prefix reuse). Falls back to
    Levenshtein for fine-grained alignment when no chunks match.

    Args:
        donor_tokens: Token IDs of the cached donor sequence.
        target_tokens: Token IDs of the new target sequence.
        chunk_size: Chunk size in tokens (engine block size).
        fuzzy: Enable fuzzy chunk matching. None uses env var default.

    Returns:
        AlignmentResult with slot actions and reuse ratio.
    """
    use_fuzzy = fuzzy if fuzzy is not None else _FUZZY_CHUNKS_ENABLED

    if use_fuzzy:
        # Fuzzy alignment: exact + fuzzy chunk matching in one pass
        fuzzy_result = compute_fuzzy_chunk_alignment(
            donor_tokens, target_tokens, chunk_size=chunk_size,
        )
        if fuzzy_result.reuse_ratio > 0:
            return fuzzy_result

    # Primary: exact chunk-level alignment (handles REORDER correctly)
    chunk_result = compute_chunk_alignment(
        donor_tokens, target_tokens, chunk_size=chunk_size,
    )

    # If chunk alignment found reusable chunks, use it
    if chunk_result.reuse_ratio > 0:
        return chunk_result

    # Fallback: Levenshtein for fine-grained alignment (PARTIAL scenarios
    # where changes are within chunks, not between them)
    return _levenshtein_alignment(donor_tokens, target_tokens)


def _levenshtein_alignment(
    donor_tokens: list[int],
    target_tokens: list[int],
) -> AlignmentResult:
    """Fine-grained Levenshtein alignment using rapidfuzz.

    Used when chunk-level matching finds no reusable chunks (e.g., PARTIAL
    overlap where changes are distributed within chunks).

    For sequences > 4096 tokens, falls back to O(n) token-set alignment
    to avoid O(n*m) blowup (~3s at 16K tokens).
    """
    if not HAS_RAPIDFUZZ:
        return _token_set_alignment(donor_tokens, target_tokens)

    # Cap Levenshtein to avoid O(n*m) blowup on long sequences.
    # At 16K tokens, lev_opcodes takes ~3 seconds — unacceptable for the
    # hot path. Use token-set alignment (O(n)) for long sequences.
    MAX_LEVENSHTEIN_TOKENS = 4096
    if (
        len(donor_tokens) > MAX_LEVENSHTEIN_TOKENS
        or len(target_tokens) > MAX_LEVENSHTEIN_TOKENS
    ):
        return _token_set_alignment(donor_tokens, target_tokens)

    ops = lev_opcodes(donor_tokens, target_tokens)

    slot_actions: list[SlotAction] = []
    num_reused = 0

    for op in ops:
        tag = op.tag
        src_start, _src_end = op.src_start, op.src_end
        dest_start, dest_end = op.dest_start, op.dest_end

        if tag == "equal":
            for i in range(dest_end - dest_start):
                slot_actions.append(SlotAction(
                    action=SlotActionType.COPY_FROM_DONOR,
                    target_pos=dest_start + i,
                    donor_pos=src_start + i,
                ))
                num_reused += 1
        elif tag == "replace":
            for i in range(dest_end - dest_start):
                slot_actions.append(SlotAction(
                    action=SlotActionType.RECOMPUTE,
                    target_pos=dest_start + i,
                ))
        elif tag == "insert":
            for i in range(dest_end - dest_start):
                slot_actions.append(SlotAction(
                    action=SlotActionType.RECOMPUTE,
                    target_pos=dest_start + i,
                ))

    target_len = len(target_tokens)
    reuse_ratio = num_reused / max(target_len, 1)
    edit_dist = sum(
        1 for sa in slot_actions if sa.action == SlotActionType.RECOMPUTE
    )

    return AlignmentResult(
        reuse_ratio=reuse_ratio,
        slot_actions=slot_actions,
        edit_distance=edit_dist,
        donor_len=len(donor_tokens),
        target_len=target_len,
    )


def compute_batch_alignment(
    candidates: list[tuple[str, list[int]]],
    target_tokens: list[int],
    min_reuse_ratio: float = 0.5,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> tuple[str, AlignmentResult] | None:
    """Run alignment on multiple candidates, return best.

    Args:
        candidates: List of (donor_id, donor_tokens) tuples.
        target_tokens: Target token sequence.
        min_reuse_ratio: Minimum acceptable reuse ratio.
        chunk_size: Chunk size in tokens (engine block size).

    Returns:
        (donor_id, AlignmentResult) for best candidate, or None.
    """
    best: tuple[str, AlignmentResult] | None = None
    best_ratio = min_reuse_ratio

    for donor_id, donor_tokens in candidates:
        result = compute_alignment(
            donor_tokens, target_tokens, chunk_size=chunk_size,
        )
        if result.reuse_ratio >= best_ratio:
            best_ratio = result.reuse_ratio
            best = (donor_id, result)

    return best


def _token_set_alignment(
    donor_tokens: list[int],
    target_tokens: list[int],
) -> AlignmentResult:
    """O(n) token-set alignment for long sequences.

    For sequences too long for Levenshtein (>4096 tokens), matches each
    target token to a donor token with the same ID using a hash map.

    Handles REORDER naturally: same tokens at different positions produce
    COPY_FROM_DONOR with donor_pos != target_pos, feeding into RoPE
    correction. For PARTIAL overlap, correctly identifies which target
    tokens have matching donor tokens and which need recomputation.

    Less precise than Levenshtein (ignores order/context) but O(n)
    vs O(n^2) and sufficient for the KV reuse decision gate.
    """
    from collections import defaultdict

    # Build donor token → list of available positions (FIFO order)
    donor_positions: dict[int, list[int]] = defaultdict(list)
    for pos, tok in enumerate(donor_tokens):
        donor_positions[tok].append(pos)

    # Index into each token's position list (avoids set removal overhead)
    donor_pos_idx: dict[int, int] = defaultdict(int)

    slot_actions: list[SlotAction] = []
    num_reused = 0

    for t_pos, t_tok in enumerate(target_tokens):
        candidates = donor_positions.get(t_tok)
        if candidates is not None:
            idx = donor_pos_idx[t_tok]
            if idx < len(candidates):
                d_pos = candidates[idx]
                donor_pos_idx[t_tok] = idx + 1
                slot_actions.append(SlotAction(
                    action=SlotActionType.COPY_FROM_DONOR,
                    target_pos=t_pos,
                    donor_pos=d_pos,
                ))
                num_reused += 1
                continue

        slot_actions.append(SlotAction(
            action=SlotActionType.RECOMPUTE,
            target_pos=t_pos,
        ))

    target_len = len(target_tokens)
    reuse_ratio = num_reused / max(target_len, 1)
    edit_dist = target_len - num_reused

    logger.info(
        "token_set_alignment: reuse=%.2f (%d/%d tokens), "
        "donor_len=%d, target_len=%d",
        reuse_ratio, num_reused, target_len,
        len(donor_tokens), target_len,
    )

    return AlignmentResult(
        reuse_ratio=reuse_ratio,
        slot_actions=slot_actions,
        edit_distance=edit_dist,
        donor_len=len(donor_tokens),
        target_len=target_len,
    )


def _fallback_prefix_alignment(
    donor_tokens: list[int],
    target_tokens: list[int],
) -> AlignmentResult:
    """Fallback: simple prefix matching when rapidfuzz is unavailable."""
    prefix_len = 0
    for i in range(min(len(donor_tokens), len(target_tokens))):
        if donor_tokens[i] == target_tokens[i]:
            prefix_len += 1
        else:
            break

    slot_actions: list[SlotAction] = []
    for i in range(len(target_tokens)):
        if i < prefix_len:
            slot_actions.append(SlotAction(
                action=SlotActionType.COPY_FROM_DONOR,
                target_pos=i,
                donor_pos=i,
            ))
        else:
            slot_actions.append(SlotAction(
                action=SlotActionType.RECOMPUTE,
                target_pos=i,
            ))

    reuse_ratio = prefix_len / max(len(target_tokens), 1)
    edit_dist = len(target_tokens) - prefix_len

    return AlignmentResult(
        reuse_ratio=reuse_ratio,
        slot_actions=slot_actions,
        edit_distance=edit_dist,
        donor_len=len(donor_tokens),
        target_len=len(target_tokens),
    )

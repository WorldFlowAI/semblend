"""Local mirror of SGLang's FuzzyMatchResult schema plus SemBlend extensions.

These dataclasses let the SemBlend provider adapter be built and tested
without requiring SGLang to be importable. The SGLang-side thin wrapper
(`SemanticEmbeddingProvider`) copies fields from our dataclass to the
actual `sglang.srt.mem_cache.fuzzy_match.provider.FuzzyMatchResult`.

Field names mirror the proposal in docs/sglang_semantic_provider_design.md
section 4. When SGLang's upstream schema is finalized, these fields must
stay name-compatible (or the wrapper must translate).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass(frozen=True)
class QualitySignals:
    """Provider-visible quality signals attached to a match.

    The SGLang scheduler logs / exports these for observability.
    """

    cosine_similarity: float
    reuse_ratio: float
    confidence_tier: str  # "exact" | "fuzzy" | "recompute" | "verified_reuse" | "fast_reuse"
    passed_quality_gate: bool
    rejection_reason: Optional[str] = None


@dataclass
class FuzzyMatchSegment:
    """One contiguous span of matched tokens in an N:M alignment plan.

    When a FuzzyMatchResult carries `segments`, model_runner must iterate
    per-segment rather than using the single contiguous `cached_start_pos`
    path. Multiple segments may reference different donors (multi-donor
    scatter) — identified by `donor_req_id`.

    Two addressing modes:

    * **NodeRef** (preferred for new providers, e.g. SemBlend): the segment
      points at a TreeNode in the radix tree's ``_node_registry`` via
      ``donor_node_id`` + ``donor_offset`` + ``length``. The model_runner
      resolves to pool indices at consume time via
      ``radix_tree._node_registry[donor_node_id].value[offset:offset+length]``.
      This preserves Chenxin's "no double-counting" principle (the radix
      tree is the single owner of pool indices) and makes donor lifetime
      structural — paired with ``FuzzyMatchResult.donor_last_node_id``
      ``inc_lock_ref`` protection.

    * **Legacy pool-indices** (``donor_kv_indices``): a raw tensor of pool
      indices. Used by ``TokenBlockMatchProvider``'s contiguous-span path
      where donor lifetime isn't a concern. New providers populate
      ``donor_node_id`` instead and leave ``donor_kv_indices=None``.
    """

    target_positions: Any  # torch.Tensor of absolute target-prompt positions
    donor_positions: Any  # torch.Tensor of source positions (for RoPE delta)

    # NodeRef-based addressing (preferred for new providers).
    donor_node_id: Optional[int] = None
    donor_offset: Optional[int] = None
    length: Optional[int] = None

    # Legacy: raw pool-indices tensor. Optional so new providers can omit it.
    donor_kv_indices: Any = None

    donor_req_id: Optional[str] = None
    layer_recompute_mask: Optional[List[bool]] = None


@dataclass
class FuzzyMatchResult:
    """Semantic-provider match result.

    Mirrors ibifrost/sglang's draft `FuzzyMatchResult` (Chenxin Wu,
    2026-04-22) with three additive optional fields:
      - segments: N:M alignment / multi-donor scatter
      - layer_recompute_mask: bathtub-curve per-layer decisions
      - quality_signals: cosine, reuse ratio, confidence tier

    When `segments is None` the result degrades to Chenxin's single-span
    contract: model_runner uses `kv_cache_indices` and `cached_start_pos`
    exactly as in TokenBlockMatchProvider's path.
    """

    cached_token_count: int
    cached_token_ids: List[int]
    prompt_token_count: int
    kv_cache_indices: Any  # torch.Tensor (possibly empty when segments is multi-span)
    position_offset: int
    cached_start_pos: int = 0
    _match_entry: Any = None

    # Semantic extensions (all optional; TokenBlockMatch ignores these).
    segments: Optional[List[FuzzyMatchSegment]] = None
    layer_recompute_mask: Optional[List[bool]] = None
    quality_signals: Optional[QualitySignals] = None
    # ID of the donor's TreeNode in radix_tree._node_registry. When set,
    # SGLang's RadixCache.match_prefix inc_lock_ref's the donor node so its
    # KV slots can't be LRU-evicted while the recipient is consuming them.
    # Populated by SemBlendProviderAdapter.match() from the donor's
    # _DonorKVHandle.last_node_id (set in on_donor_inserted).
    donor_last_node_id: Optional[int] = None

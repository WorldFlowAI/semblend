"""SemBlend pipeline orchestrator — in-process semantic KV donor discovery.

Coordinates the full SemBlend Local pipeline with zero network hops:
  1. MiniLM ONNX GPU embedding (~3ms) — 384-dim order-invariant vector
  2. Donor store lookup (~2ms) — numpy cosine similarity + Jaccard pre-filter
  3. Chunk alignment (<1ms) — O(n) hash matching; Levenshtein capped at 4K tokens
  4. Bathtub curve layer deviations (<0.1ms) — predict recomputation needs
  5. Build PartialAttention plan — consumed by Triton kernels or engine adapter

Total overhead: ~8ms (excluding KV load from CPU offload).

Backend-agnostic: accepts an optional SemBlendBackend instance to
parameterize chunk_size and other engine-specific settings.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

import numpy as np

from semblend_core.alignment import DEFAULT_CHUNK_SIZE
from semblend_core.backend import SemBlendBackend

logger = logging.getLogger(__name__)


def _order_invariant_text(text: str, max_chars: int = 200_000) -> str:
    """Prepare text for embedding with length truncation.

    With full-document segmented embedding (overlapping windows + mean
    pooling), the embedding is inherently near-order-invariant (0.996
    cosine similarity for reordered documents). Sentence sorting was
    removed because:
      - The segmented mean pool already provides order invariance
      - Sorting on ". " broke on code, markdown, non-English text
      - Sorting slightly hurt cross-instruction similarity (-0.002)
      - The 0.004 gap from not sorting is invisible at any practical
        matching threshold (default 0.60)

    The max_chars limit is set high (200K) to accommodate full-document
    embedding; the embedder handles segmentation internally.
    """
    return text[:max_chars]


@dataclass
class PipelineTimings:
    """Per-stage latency breakdown in milliseconds."""
    embed_ms: float = 0.0
    lookup_ms: float = 0.0
    align_ms: float = 0.0
    bathtub_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class PositionMapping:
    """Maps donor positions to target positions for RoPE correction.

    Each pair (donor_pos[i], target_pos[i]) means: the KV at donor_pos[i]
    should be placed at target_pos[i] in the target sequence, with K
    corrected by RoPE(target_pos[i] - donor_pos[i]).
    """
    donor_positions: list[int] = field(default_factory=list)
    target_positions: list[int] = field(default_factory=list)

    @property
    def num_pairs(self) -> int:
        return len(self.donor_positions)

    @property
    def needs_correction(self) -> bool:
        """True if any position pair has a non-zero delta."""
        return any(
            d != t
            for d, t in zip(self.donor_positions, self.target_positions)
        )


@dataclass
class PipelineResult:
    """Result of the SemBlend pipeline for a single request."""
    found: bool
    donor_id: str | None = None
    similarity: float = 0.0
    reuse_ratio: float = 0.0
    donor_tokens: list[int] = field(default_factory=list)
    slot_actions: list[dict] = field(default_factory=list)
    layer_deviations: list[dict] = field(default_factory=list)
    position_map: PositionMapping = field(default_factory=PositionMapping)
    timings: PipelineTimings = field(default_factory=PipelineTimings)
    rejection_reason: str | None = None
    fuzzy_confidence: float = 1.0
    force_verify_layers: list[int] = field(default_factory=list)
    confidence_tier: str = "exact"
    # Multi-donor fields
    donor_ids: list[str] = field(default_factory=list)
    composite_plan: object | None = None  # CompositeKVPlan when multi-donor
    multi_donor_position_map: object | None = None  # MultiDonorPositionMapping
    chunk_fast_path_used: bool = False
    # SemShareKV fields (mode="semshare")
    mode: str = "chunk"  # "chunk" or "semshare"
    semshare_schedule: object | None = None  # LayerSchedule
    kv_rearrange_plan: object | None = None  # KVRearrangePlan
    hd_detection: object | None = None  # HDDetectionResult
    token_match_ratio: float = 0.0


class SemBlendPipeline:
    """In-process SemBlend pipeline with zero network hops on the hot path.

    All components are loaded at init and run in-process:
      - MiniLM ONNX GPU embedder (~3ms, 384-dim, order-invariant)
      - Numpy donor store (vectorized cosine search, ~2ms at 10K)
      - rapidfuzz alignment (compiled C++, <1ms per candidate)
      - Calibrated bathtub curve (pure math, <0.1ms)

    Args:
        max_donors: Maximum entries in the donor store.
        min_similarity: Minimum cosine similarity for donor candidates.
        min_reuse_ratio: Minimum alignment reuse ratio.
        embedder_type: "minilm", "jina", or "jaccard".
        model_name: Model name for bathtub curve preset lookup.
        backend: Optional SemBlendBackend for engine-specific settings.
        chunk_size: KV block size override (default from backend or 256).
        donor_store: Optional pre-configured donor store instance. If None,
            creates a default numpy DonorStore.
    """

    def __init__(
        self,
        max_donors: int = 10_000,
        min_similarity: float = 0.60,
        min_reuse_ratio: float = 0.50,
        embedder_type: str | None = None,
        model_name: str | None = None,
        backend: SemBlendBackend | None = None,
        chunk_size: int | None = None,
        donor_store: object | None = None,
        fuzzy_config: object | None = None,
        recompute_config: object | None = None,
        enable_pq_segments: bool = True,
    ) -> None:
        self._min_similarity = min_similarity
        self._min_reuse_ratio = min_reuse_ratio
        self._model_name = model_name
        self._backend = backend
        self._fuzzy_config = fuzzy_config
        self._recompute_config = recompute_config
        self._pq_store = None

        # FM1 ablation: set SEMBLEND_USE_ALIGNMENT=0 to disable RoPE delta correction
        self._use_alignment = os.environ.get(
            "SEMBLEND_USE_ALIGNMENT", "1"
        ).strip() not in ("0", "false", "False")
        if not self._use_alignment:
            logger.warning(
                "[SemBlend] SEMBLEND_USE_ALIGNMENT=0: "
                "RoPE delta correction DISABLED (FM1 ablation mode)"
            )

        # Chunk fast path: skip MiniLM when ChunkIndex finds >=3 matching chunks
        self._chunk_fast_path = os.environ.get(
            "SEMBLEND_CHUNK_FAST_PATH", "1"
        ).strip() not in ("0", "false", "False")

        # Multi-donor composite KV injection
        self._multi_donor = os.environ.get(
            "SEMBLEND_MULTI_DONOR", "0"
        ).strip() in ("1", "true", "True")

        # SemShareKV mode: token-level LSH matching with per-layer selective recompute
        self._mode = os.environ.get("SEMBLEND_MODE", "chunk").strip().lower()
        self._semshare_config = None
        self._lsh_index = None
        if self._mode == "semshare":
            from semblend_core.semshare.config import SemShareConfig
            from semblend_core.semshare.lsh_index import LSHIndex
            self._semshare_config = SemShareConfig(max_donors=max_donors)
            self._lsh_index = LSHIndex(self._semshare_config)
            logger.info("SemBlend mode=semshare: token-level LSH matching enabled")

        # Minimum ChunkIndex hits to trigger fast path (skip embedding)
        try:
            self._fast_path_min_hits = int(
                os.environ.get("SEMBLEND_FAST_PATH_MIN_HITS", "3")
            )
        except ValueError:
            logger.warning("Invalid SEMBLEND_FAST_PATH_MIN_HITS, defaulting to 3")
            self._fast_path_min_hits = 3

        # Determine chunk size: explicit > backend > default
        if chunk_size is not None:
            self._chunk_size = chunk_size
        elif backend is not None:
            self._chunk_size = backend.get_kv_block_size()
        else:
            self._chunk_size = DEFAULT_CHUNK_SIZE

        # Lazy imports to avoid hard dependency at module level
        from semblend_core.embedder import create_embedder

        self._embedder = create_embedder(embedder_type)

        # Use provided donor_store or create default
        if donor_store is not None:
            self._donor_store = donor_store
        else:
            from semblend_core.donor_store import DonorStore
            self._donor_store = DonorStore(
                max_entries=max_donors,
                embedding_dim=self._embedder.dimension,
                min_similarity=min_similarity,
                chunk_size=self._chunk_size,
            )

        # Initialize PQ segment store for fuzzy matching verification
        if enable_pq_segments and os.environ.get("SEMBLEND_SEGMENT_EMBEDDINGS", "1") == "1":
            try:
                from semblend_core.pq_segment_store import PQSegmentStore
                pq_train = int(os.environ.get("SEMBLEND_PQ_TRAIN_THRESHOLD", "500"))
                self._pq_store = PQSegmentStore(
                    max_entries=max_donors,
                    train_threshold=pq_train,
                )
            except Exception:
                logger.warning("PQ segment store init failed", exc_info=True)

        logger.info(
            "SemBlend pipeline initialized: mode=%s, embedder=%s (dim=%d), "
            "store=%s, max_donors=%d, min_sim=%.2f, min_reuse=%.2f, "
            "chunk_size=%d, pq_segments=%s, chunk_fast_path=%s, "
            "multi_donor=%s",
            self._mode,
            type(self._embedder).__name__,
            self._embedder.dimension,
            type(self._donor_store).__name__,
            max_donors,
            min_similarity,
            min_reuse_ratio,
            self._chunk_size,
            self._pq_store is not None,
            self._chunk_fast_path,
            self._multi_donor,
        )

    @property
    def donor_count(self) -> int:
        return self._donor_store.size

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def find_donor(
        self,
        token_ids: list[int],
        prompt_text: str = "",
        top_k: int = 5,
    ) -> PipelineResult:
        """Run the full SemBlend donor discovery pipeline.

        Graceful degradation: any internal error returns a miss result
        rather than propagating the exception to the caller. This ensures
        SemBlend never blocks inference — worst case is cold prefill.

        Args:
            token_ids: Token IDs of the new request.
            prompt_text: Decoded prompt text (for embedding).
            top_k: Number of top candidates for alignment.

        Returns:
            PipelineResult with donor info and timings.
        """
        timings = PipelineTimings()
        t_start = time.monotonic()

        try:
            return self._find_donor_inner(
                token_ids, prompt_text, top_k, timings, t_start,
            )
        except Exception as e:
            logger.error(
                "SemBlend pipeline error (graceful degradation → cold prefill): %s",
                e, exc_info=True,
            )
            timings.total_ms = (time.monotonic() - t_start) * 1000
            try:
                from semblend_core.metrics import METRICS
                METRICS.record_pipeline_error(stage="pipeline")
            except Exception:
                pass
            return PipelineResult(
                found=False,
                timings=timings,
                rejection_reason=f"pipeline_error: {e}",
            )

    def _find_donor_inner(
        self,
        token_ids: list[int],
        prompt_text: str,
        top_k: int,
        timings: PipelineTimings,
        t_start: float,
    ) -> PipelineResult:
        """Inner pipeline logic (may raise exceptions)."""
        # Stage 0: ChunkIndex fast path — skip embedding if >=N chunks match
        if self._chunk_fast_path and hasattr(self._donor_store, 'chunk_index'):
            chunk_index = self._donor_store.chunk_index
            if chunk_index.num_donors > 0:
                chunk_matches = chunk_index.find_matching_chunks(
                    token_ids, min_matches=1,
                )
                if len(chunk_matches) >= self._fast_path_min_hits:
                    # Fast path: try multi-donor alignment directly
                    fast_result = self._try_chunk_fast_path(
                        token_ids, chunk_matches, timings, t_start,
                    )
                    if fast_result is not None:
                        return fast_result
                    # Attempted fast path but fell through to full pipeline

        # Stage 1: Embedding (order-invariant via sentence sorting)
        t0 = time.monotonic()
        query_embedding = None
        if prompt_text:
            raw = self._embedder.embed(_order_invariant_text(prompt_text))
            if raw is not None:
                query_embedding = np.asarray(raw, dtype=np.float32)
        timings.embed_ms = (time.monotonic() - t0) * 1000
        try:
            from semblend_core.metrics import METRICS
            METRICS.record_embedding_latency(timings.embed_ms)
        except Exception:
            pass

        # Stage 1.5: Multi-donor path (if enabled)
        if self._multi_donor:
            try:
                multi_result = self._try_multi_donor(
                    token_ids, timings, t_start, prompt_text=prompt_text,
                )
                if multi_result is not None:
                    return multi_result
            except Exception as e:
                logger.warning("multi-donor path failed: %s", e, exc_info=True)

        # Stage 2: Donor lookup (cosine similarity + alignment)
        t0 = time.monotonic()
        match = self._donor_store.find_donor(
            query_embedding=query_embedding,
            query_tokens=token_ids,
            top_k=top_k,
            min_reuse_ratio=self._min_reuse_ratio,
        )
        timings.lookup_ms = (time.monotonic() - t0) * 1000

        if match is None:
            timings.total_ms = (time.monotonic() - t_start) * 1000
            try:
                from semblend_core.metrics import METRICS
                METRICS.record_pipeline_result(hit=False)
            except Exception:
                pass
            return PipelineResult(
                found=False,
                timings=timings,
                rejection_reason="no_donor_match",
            )

        # Guard: reject fuzzy-only matches with low reuse (false positive prevention)
        alignment = match.alignment
        fuzzy_chunks = getattr(alignment, "fuzzy_chunks", 0)
        exact_chunks = getattr(alignment, "exact_chunks", 0)
        if fuzzy_chunks > 0 and exact_chunks == 0 and alignment.reuse_ratio < 0.30:
            timings.total_ms = (time.monotonic() - t_start) * 1000
            return PipelineResult(
                found=False,
                timings=timings,
                rejection_reason="fuzzy_low_reuse",
            )

        # Stage 4: Bathtub curve layer deviations (fuzzy-aware)
        t0 = time.monotonic()
        from semblend_core.bathtub import compute_layer_deviations

        num_layers = self._detect_num_layers()
        mismatch = 1.0 - alignment.reuse_ratio

        # Extract fuzzy metadata from alignment
        alignment = match.alignment
        fuzzy_chunks = getattr(alignment, "fuzzy_chunks", 0)
        exact_chunks = getattr(alignment, "exact_chunks", 0)
        total_chunks = fuzzy_chunks + exact_chunks
        fuzzy_fraction = fuzzy_chunks / max(total_chunks, 1) if total_chunks > 0 else 0.0
        mean_fuzzy_conf = getattr(alignment, "mean_fuzzy_confidence", 1.0)

        layer_devs = compute_layer_deviations(
            num_layers=num_layers,
            mismatch_fraction=mismatch,
            model_name=self._model_name,
            similarity=match.similarity,
            fuzzy_fraction=fuzzy_fraction,
            mean_fuzzy_confidence=mean_fuzzy_conf,
            recompute_config=self._recompute_config,
        )
        timings.bathtub_ms = (time.monotonic() - t0) * 1000

        # Determine confidence tier and force-verify layers
        force_verify = [d.layer_idx for d in layer_devs if d.should_recompute]
        if fuzzy_chunks > 0:
            confidences = getattr(alignment, "chunk_confidences", ())
            tiers = [c.tier for c in confidences] if confidences else []
            if any(t == "verified_reuse" for t in tiers):
                confidence_tier = "verified_reuse"
            elif any(t == "fast_reuse" for t in tiers):
                confidence_tier = "fast_reuse"
            else:
                confidence_tier = "exact"
        else:
            confidence_tier = "exact"

        timings.total_ms = (time.monotonic() - t_start) * 1000

        # Build slot actions in the format expected by partial_attention
        slot_actions = []
        for sa in match.alignment.slot_actions:
            slot_actions.append({
                "action": sa.action.value,
                "targetPos": sa.target_pos,
                "donorPos": sa.donor_pos,
            })

        layer_dev_dicts = [
            {
                "layerIdx": ld.layer_idx,
                "deviationScore": ld.deviation_score,
                "shouldRecompute": ld.should_recompute,
            }
            for ld in layer_devs
        ]

        # Build position map for RoPE correction
        # FM1 ablation: when SEMBLEND_USE_ALIGNMENT=0, use identity mapping
        position_map = PositionMapping()
        if self._use_alignment:
            for sa in match.alignment.slot_actions:
                if sa.action.value == "copy_from_donor" and sa.donor_pos is not None:
                    position_map.donor_positions.append(sa.donor_pos)
                    position_map.target_positions.append(sa.target_pos)
        else:
            n = sum(1 for sa in match.alignment.slot_actions
                    if sa.action.value == "copy_from_donor")
            position_map.donor_positions = list(range(n))
            position_map.target_positions = list(range(n))

        try:
            from semblend_core.metrics import METRICS
            METRICS.record_pipeline_result(
                hit=True,
                similarity=match.similarity,
                reuse_ratio=match.alignment.reuse_ratio,
                fuzzy_chunks=getattr(match.alignment, "fuzzy_chunks", 0),
            )
            METRICS.record_donor_store_size(
                len(self._donor_store._entries)
                if hasattr(self._donor_store, "_entries")
                else 0
            )
        except Exception:
            pass

        return PipelineResult(
            found=True,
            donor_id=match.donor.request_id,
            similarity=match.similarity,
            reuse_ratio=match.alignment.reuse_ratio,
            donor_tokens=match.donor.token_ids,
            slot_actions=slot_actions,
            layer_deviations=layer_dev_dicts,
            position_map=position_map,
            timings=timings,
            fuzzy_confidence=mean_fuzzy_conf,
            force_verify_layers=force_verify,
            confidence_tier=confidence_tier,
        )

    def find_donor_candidates(
        self,
        token_ids: list[int],
        prompt_text: str = "",
        top_k: int = 5,
    ) -> list[PipelineResult]:
        """Find multiple donor candidates, ranked by score with recency bias.

        Used by the connector to try each candidate's KV in the engine
        cache until one is found.
        """
        timings = PipelineTimings()
        t_start = time.monotonic()

        # Stage 1: Embedding
        t0 = time.monotonic()
        query_embedding = None
        if prompt_text:
            raw = self._embedder.embed(_order_invariant_text(prompt_text))
            if raw is not None:
                query_embedding = np.asarray(raw, dtype=np.float32)
        timings.embed_ms = (time.monotonic() - t0) * 1000

        # Stage 2: Multi-candidate donor lookup
        t0 = time.monotonic()
        matches = self._donor_store.find_donors(
            query_embedding=query_embedding,
            query_tokens=token_ids,
            top_k=top_k,
            min_reuse_ratio=self._min_reuse_ratio,
        )
        timings.lookup_ms = (time.monotonic() - t0) * 1000

        if not matches:
            timings.total_ms = (time.monotonic() - t_start) * 1000
            return [PipelineResult(
                found=False,
                timings=timings,
                rejection_reason="no_donor_match",
            )]

        # Build PipelineResult for each candidate
        results = []
        for match in matches:
            from semblend_core.bathtub import compute_layer_deviations

            num_layers = self._detect_num_layers()
            mismatch = 1.0 - match.alignment.reuse_ratio
            layer_devs = compute_layer_deviations(
                num_layers=num_layers,
                mismatch_fraction=mismatch,
                model_name=self._model_name,
            )

            slot_actions = []
            for sa in match.alignment.slot_actions:
                slot_actions.append({
                    "action": sa.action.value,
                    "targetPos": sa.target_pos,
                    "donorPos": sa.donor_pos,
                })

            layer_dev_dicts = [
                {
                    "layerIdx": ld.layer_idx,
                    "deviationScore": ld.deviation_score,
                    "shouldRecompute": ld.should_recompute,
                }
                for ld in layer_devs
            ]

            position_map = PositionMapping()
            if self._use_alignment:
                for sa in match.alignment.slot_actions:
                    if sa.action.value == "copy_from_donor" and sa.donor_pos is not None:
                        position_map.donor_positions.append(sa.donor_pos)
                        position_map.target_positions.append(sa.target_pos)
            else:
                n = sum(1 for sa in match.alignment.slot_actions
                        if sa.action.value == "copy_from_donor")
                position_map.donor_positions = list(range(n))
                position_map.target_positions = list(range(n))

            result_timings = PipelineTimings(
                embed_ms=timings.embed_ms,
                lookup_ms=timings.lookup_ms,
                total_ms=(time.monotonic() - t_start) * 1000,
            )

            results.append(PipelineResult(
                found=True,
                donor_id=match.donor.request_id,
                similarity=match.similarity,
                reuse_ratio=match.alignment.reuse_ratio,
                donor_tokens=match.donor.token_ids,
                slot_actions=slot_actions,
                layer_deviations=layer_dev_dicts,
                position_map=position_map,
                timings=result_timings,
            ))

        return results

    # ------------------------------------------------------------------
    # Donor registration
    # ------------------------------------------------------------------

    def register_donor(
        self,
        request_id: str,
        token_ids: list[int],
        prompt_text: str = "",
    ) -> None:
        """Register a completed request as a potential donor.

        Uses embed_with_segments when PQ segment store is available,
        falling back to standard embed() otherwise.

        Args:
            request_id: Unique request identifier.
            token_ids: Token IDs of the completed request.
            prompt_text: Decoded prompt text for embedding.
        """
        from semblend_core.donor_store import DonorNode

        embedding = None
        segment_embeddings = None

        if prompt_text:
            text = _order_invariant_text(prompt_text)

            # Try embed_with_segments for PQ store integration
            if self._pq_store and hasattr(self._embedder, "embed_with_segments"):
                result = self._embedder.embed_with_segments(
                    text, chunk_size=self._chunk_size,
                )
                if result is not None:
                    embedding = np.asarray(result.pooled, dtype=np.float32)
                    segment_embeddings = result.segments

            # Fallback to standard embed
            if embedding is None:
                raw = self._embedder.embed(text)
                if raw is not None:
                    embedding = np.asarray(raw, dtype=np.float32)

        node = DonorNode(
            request_id=request_id,
            token_ids=token_ids,
            embedding=embedding,
            timestamp=time.monotonic(),
            prompt_text=prompt_text[:200],
        )
        self._donor_store.add_donor(node)

        # Store segment embeddings in PQ store
        if self._pq_store and segment_embeddings is not None:
            self._pq_store.add_segments(request_id, segment_embeddings.matrix)

    # ------------------------------------------------------------------
    # PartialAttention plan building
    # ------------------------------------------------------------------

    def build_partial_attention_plan(
        self,
        pipeline_result: PipelineResult,
    ) -> object | None:
        """Build a PartialAttentionPlan from a pipeline result.

        Converts the pipeline's slot_actions and layer_deviations into
        a structured plan consumed by the engine adapter and Triton kernels.
        """
        if not pipeline_result.found or not pipeline_result.slot_actions:
            return None

        try:
            from semblend_core.partial_attention import build_attention_plan

            copy_positions = [
                sa["targetPos"]
                for sa in pipeline_result.slot_actions
                if sa["action"] == "copy_from_donor"
            ]
            placeholder_positions = [
                sa["targetPos"]
                for sa in pipeline_result.slot_actions
                if sa["action"] == "recompute"
            ]

            layer_hints = None
            if pipeline_result.layer_deviations:
                layer_hints = [
                    {
                        "recompute_all": ld["shouldRecompute"],
                        "deviation_score": ld["deviationScore"],
                    }
                    for ld in pipeline_result.layer_deviations
                ]

            num_layers = self._detect_num_layers()

            # Infer target/donor lengths from slot actions
            target_len = max(
                (sa["targetPos"] for sa in pipeline_result.slot_actions),
                default=0,
            ) + 1
            donor_len = len(pipeline_result.donor_tokens)

            plan = build_attention_plan(
                donor_id=pipeline_result.donor_id or "",
                target_len=target_len,
                donor_len=donor_len,
                copy_positions=copy_positions,
                placeholder_positions=placeholder_positions,
                slot_actions=pipeline_result.slot_actions,
                layer_hints=layer_hints,
                num_layers=num_layers,
            )

            logger.info(
                "PartialAttention plan: reuse=%d, partial=%d, "
                "full_layers=%d/%d, comp_ratio=%.2f",
                plan.num_reuse_positions,
                plan.num_partial_positions,
                plan.num_full_layers,
                num_layers,
                plan.computation_ratio,
            )
            return plan

        except Exception:
            logger.warning(
                "Failed to build PartialAttention plan", exc_info=True
            )
            return None

    # ------------------------------------------------------------------
    # Multi-turn fast path + multi-donor
    # ------------------------------------------------------------------

    def _try_chunk_fast_path(
        self,
        token_ids: list[int],
        chunk_matches: dict[int, list],
        timings: PipelineTimings,
        t_start: float,
    ) -> PipelineResult | None:
        """Attempt fast path using ChunkIndex matches (skip embedding).

        When >=3 chunks match via ChunkIndex, we can go directly to
        multi-donor alignment, bypassing MiniLM embedding (~3ms savings).
        """
        t0 = time.monotonic()
        result = self._donor_store.find_multi_donor(
            query_tokens=token_ids,
            min_reuse_ratio=self._min_reuse_ratio,
            pq_store=self._pq_store,
        )
        timings.lookup_ms = (time.monotonic() - t0) * 1000

        if result is None or result.reuse_ratio < self._min_reuse_ratio:
            return None

        timings.embed_ms = 0.0  # Skipped embedding!
        return self._build_multi_donor_result(
            result, timings, t_start,
            similarity=1.0, chunk_fast_path_used=True,
        )

    def _try_multi_donor(
        self,
        token_ids: list[int],
        timings: PipelineTimings,
        t_start: float,
        prompt_text: str = "",
    ) -> PipelineResult | None:
        """Attempt multi-donor composite alignment with semantic chunk matching.

        Falls back to single-donor path if multi-donor finds nothing better.
        """
        t0 = time.monotonic()
        result = self._donor_store.find_multi_donor(
            query_tokens=token_ids,
            min_reuse_ratio=self._min_reuse_ratio,
            pq_store=self._pq_store,
            target_text=prompt_text,
            embedder=self._embedder,
        )
        lookup_ms = (time.monotonic() - t0) * 1000

        if result is None:
            return None
        if result.reuse_ratio < self._min_reuse_ratio:
            return None  # Below minimum reuse threshold

        timings.lookup_ms = lookup_ms

        fuzzy_fraction = result.fuzzy_chunks / max(
            result.exact_chunks + result.fuzzy_chunks, 1,
        )
        return self._build_multi_donor_result(
            result, timings, t_start,
            similarity=0.0, fuzzy_fraction=fuzzy_fraction,
        )

    def _build_multi_donor_result(
        self,
        result: object,
        timings: PipelineTimings,
        t_start: float,
        similarity: float = 0.0,
        fuzzy_fraction: float = 0.0,
        chunk_fast_path_used: bool = False,
    ) -> PipelineResult:
        """Build PipelineResult from a MultiDonorAlignmentResult."""
        from semblend_core.bathtub import compute_layer_deviations

        t0 = time.monotonic()
        num_layers = self._detect_num_layers()
        layer_devs = compute_layer_deviations(
            num_layers=num_layers,
            mismatch_fraction=1.0 - result.reuse_ratio,
            model_name=self._model_name,
            fuzzy_fraction=fuzzy_fraction,
            recompute_config=self._recompute_config,
        )
        timings.bathtub_ms = (time.monotonic() - t0) * 1000
        timings.total_ms = (time.monotonic() - t_start) * 1000

        force_verify = [d.layer_idx for d in layer_devs if d.should_recompute]
        layer_dev_dicts = [
            {
                "layerIdx": ld.layer_idx,
                "deviationScore": ld.deviation_score,
                "shouldRecompute": ld.should_recompute,
            }
            for ld in layer_devs
        ]

        composite = result.composite_plan
        slot_actions = [
            {
                "action": sa.action,
                "targetPos": sa.target_pos,
                "donorPos": sa.donor_pos,
                "donorId": sa.donor_id,
            }
            for sa in composite.slot_actions
        ]

        position_map = PositionMapping()
        if self._use_alignment and composite.position_map:
            for i in range(composite.position_map.num_pairs):
                position_map.donor_positions.append(
                    composite.position_map.donor_positions[i]
                )
                position_map.target_positions.append(
                    composite.position_map.target_positions[i]
                )

        try:
            from semblend_core.metrics import METRICS
            METRICS.record_pipeline_result(hit=True, reuse_ratio=result.reuse_ratio)
            if chunk_fast_path_used:
                METRICS.record_chunk_fast_path_hit()
            if result.donors_per_composite > 1:
                METRICS.record_multi_donor_hit(result.donors_per_composite)
            METRICS.record_chunk_index_size(
                self._donor_store.chunk_index.num_entries
                if hasattr(self._donor_store, 'chunk_index') else 0
            )
        except Exception:
            pass

        primary_donor_id = result.donor_ids[0] if result.donor_ids else None
        donor_tokens = []
        if primary_donor_id:
            dt = self._donor_store.get_donor_tokens(primary_donor_id)
            if dt is not None:
                donor_tokens = dt

        confidence_tier = "exact"
        if result.fuzzy_chunks > 0:
            confidence_tier = "verified_reuse"

        return PipelineResult(
            found=True,
            donor_id=primary_donor_id,
            similarity=similarity,
            reuse_ratio=result.reuse_ratio,
            donor_tokens=donor_tokens,
            slot_actions=slot_actions,
            layer_deviations=layer_dev_dicts,
            position_map=position_map,
            timings=timings,
            confidence_tier=confidence_tier,
            force_verify_layers=force_verify,
            donor_ids=list(result.donor_ids),
            composite_plan=composite,
            multi_donor_position_map=composite.position_map,
            chunk_fast_path_used=chunk_fast_path_used,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _detect_num_layers(self) -> int:
        """Detect number of transformer layers from model name or backend."""
        # Try backend config first
        if self._backend is not None:
            config = self._backend.get_model_config()
            if "num_layers" in config:
                return config["num_layers"]

        if self._model_name:
            name = self._model_name.lower()
            if "1.5b" in name:
                return 28
            if "7b" in name:
                return 28
            if "72b" in name:
                return 80
            if "14b" in name:
                return 40
        return 32  # default

    # ------------------------------------------------------------------
    # SemShareKV mode: token-level LSH matching
    # ------------------------------------------------------------------

    def find_donor_semshare(
        self,
        token_ids: list[int],
        token_embeddings: np.ndarray,
    ) -> PipelineResult:
        """SemShareKV donor discovery via token-level LSH matching.

        Called when SEMBLEND_MODE=semshare and token embeddings are available
        from the model's layer 0 output (captured by the vLLM hook).

        Args:
            token_ids: Token IDs of the target request.
            token_embeddings: shape [seq_len, embed_dim] from model layer 0.

        Returns:
            PipelineResult with semshare-specific fields.
        """
        timings = PipelineTimings()
        t_start = time.monotonic()

        try:
            return self._find_donor_semshare_inner(
                token_ids, token_embeddings, timings, t_start,
            )
        except Exception as e:
            logger.error(
                "SemShareKV pipeline error (graceful → cold prefill): %s",
                e, exc_info=True,
            )
            timings.total_ms = (time.monotonic() - t_start) * 1000
            return PipelineResult(
                found=False, mode="semshare", timings=timings,
                rejection_reason=f"semshare_error: {e}",
            )

    def _find_donor_semshare_inner(
        self,
        token_ids: list[int],
        token_embeddings: np.ndarray,
        timings: PipelineTimings,
        t_start: float,
    ) -> PipelineResult:
        """Inner semshare lookup — no exception handling."""
        from semblend_core.semshare.token_matcher import match_tokens
        from semblend_core.semshare.kv_rearrange import build_rearrange_plan

        if self._lsh_index is None or self._semshare_config is None:
            return PipelineResult(
                found=False, mode="semshare", timings=timings,
                rejection_reason="semshare_not_initialized",
            )

        # Stage 1: LSH token matching
        t0 = time.monotonic()
        match_result = match_tokens(
            token_embeddings, self._lsh_index, self._semshare_config,
        )
        timings.lookup_ms = (time.monotonic() - t0) * 1000

        if match_result is None:
            timings.total_ms = (time.monotonic() - t_start) * 1000
            return PipelineResult(
                found=False, mode="semshare", timings=timings,
                rejection_reason="semshare_no_match",
            )

        # Stage 2: Build KV rearrangement plan
        t1 = time.monotonic()
        donor_node = self._donor_store.get_donor(match_result.donor_id)
        donor_seq_len = len(donor_node.token_ids) if donor_node else 0

        rearrange_plan = build_rearrange_plan(
            match_result,
            target_seq_len=len(token_ids),
            donor_seq_len=donor_seq_len,
        )
        timings.align_ms = (time.monotonic() - t1) * 1000

        timings.total_ms = (time.monotonic() - t_start) * 1000

        logger.info(
            "SemShareKV match: donor=%s, match_ratio=%.2f, "
            "coverage=%.2f, rope_correction=%s, latency=%.1fms",
            match_result.donor_id,
            match_result.match_ratio,
            rearrange_plan.coverage,
            rearrange_plan.needs_rope_correction,
            timings.total_ms,
        )

        return PipelineResult(
            found=True,
            mode="semshare",
            donor_id=match_result.donor_id,
            similarity=match_result.match_ratio,
            reuse_ratio=rearrange_plan.coverage,
            donor_tokens=list(donor_node.token_ids) if donor_node else [],
            timings=timings,
            token_match_ratio=match_result.match_ratio,
            kv_rearrange_plan=rearrange_plan,
        )

    def register_donor_semshare(
        self,
        request_id: str,
        token_ids: list[int],
        token_embeddings: np.ndarray,
        prompt_text: str = "",
    ) -> None:
        """Register a donor with token embeddings for SemShareKV LSH indexing.

        Called after a request completes when SEMBLEND_MODE=semshare.
        The token_embeddings come from model layer 0 (captured by vLLM hook).

        Args:
            request_id: Unique request identifier.
            token_ids: Token IDs of the completed request.
            token_embeddings: shape [seq_len, embed_dim] from model layer 0.
            prompt_text: Decoded prompt text (for MiniLM embedding too).
        """
        # Register in LSH index for token-level matching
        if self._lsh_index is not None:
            self._lsh_index.register_donor(request_id, token_embeddings)

        # Also register in standard donor store (for chunk-level fallback)
        self.register_donor(request_id, token_ids, prompt_text)

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def lsh_index(self):
        return self._lsh_index

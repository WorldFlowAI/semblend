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


def _order_invariant_text(text: str, max_chars: int = 8000) -> str:
    """Produce order-invariant text representation for embedding.

    Splits text into sentences, sorts them, and joins. This makes the
    embedding invariant to document reordering — critical for SemBlend's
    REORDER scenario where the same chunks appear in different order.

    IMPORTANT: Sort ALL sentences first, then truncate. Truncating before
    sorting captures different paragraphs for different orderings, breaking
    order invariance at long context lengths.
    """
    parts = [s.strip() for s in text.replace('\n', '. ').split('. ') if s.strip()]
    parts.sort()
    result = '. '.join(parts)
    return result[:max_chars]


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
    ) -> None:
        self._min_similarity = min_similarity
        self._min_reuse_ratio = min_reuse_ratio
        self._model_name = model_name
        self._backend = backend

        # FM1 ablation: set SEMBLEND_USE_ALIGNMENT=0 to disable RoPE delta correction
        self._use_alignment = os.environ.get(
            "SEMBLEND_USE_ALIGNMENT", "1"
        ).strip() not in ("0", "false", "False")
        if not self._use_alignment:
            logger.warning(
                "[SemBlend] SEMBLEND_USE_ALIGNMENT=0: "
                "RoPE delta correction DISABLED (FM1 ablation mode)"
            )

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

        logger.info(
            "SemBlend pipeline initialized: embedder=%s (dim=%d), "
            "store=%s, max_donors=%d, min_sim=%.2f, min_reuse=%.2f, chunk_size=%d",
            type(self._embedder).__name__,
            self._embedder.dimension,
            type(self._donor_store).__name__,
            max_donors,
            min_similarity,
            min_reuse_ratio,
            self._chunk_size,
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

        # Stage 4: Bathtub curve layer deviations
        t0 = time.monotonic()
        from semblend_core.bathtub import compute_layer_deviations

        num_layers = self._detect_num_layers()
        mismatch = 1.0 - match.alignment.reuse_ratio
        layer_devs = compute_layer_deviations(
            num_layers=num_layers,
            mismatch_fraction=mismatch,
            model_name=self._model_name,
        )
        timings.bathtub_ms = (time.monotonic() - t0) * 1000

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

        Args:
            request_id: Unique request identifier.
            token_ids: Token IDs of the completed request.
            prompt_text: Decoded prompt text for embedding.
        """
        from semblend_core.donor_store import DonorNode

        embedding = None
        if prompt_text:
            raw = self._embedder.embed(_order_invariant_text(prompt_text))
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

"""SemBlend provider adapter for SGLang's FuzzyMatchProvider contract.

Wraps `semblend_core.SemBlendPipeline` to implement the two methods SGLang's
`FuzzyMatchProvider` ABC requires (see docs/sglang_semantic_provider_design.md):

    register_donor(req, token_ids, kv_cache, start, end, radix_tree) -> bool
    match(prompt_token_ids, already_matched_len, extra_key) -> FuzzyMatchResult | None

Design goals:

1. **Pure adapter** — no new SemBlend algorithmic code. Everything delegates to
   `semblend_core`. The adapter's job is shape-shifting: SGLang types in,
   SGLang types out.
2. **Importable without SGLang** — uses local `types.FuzzyMatchResult`. The
   SGLang-side thin wrapper converts to the real SGLang dataclass.
3. **Graceful degradation** — pipeline errors never raise to the caller;
   they become `None` (match miss), never blocking inference.
4. **Backend-agnostic KV handle** — the adapter stores donor KV as an
   opaque object (`Any`). For local mode it's typically a pool-indices
   torch.Tensor; for gateway mode it's a chunk-hash reference. Callers
   decide the representation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

from semblend.integration.sglang.config import SemBlendProviderConfig
from semblend.integration.sglang.types import (
    FuzzyMatchResult,
    FuzzyMatchSegment,
    QualitySignals,
)

logger = logging.getLogger(__name__)


@dataclass
class _DonorKVHandle:
    """Opaque reference to a donor's KV location.

    - `kv_indices`: pool-indices tensor (or list in tests). Returned to the
      engine as `FuzzyMatchResult.kv_cache_indices` / `FuzzyMatchSegment.donor_kv_indices`.
    - `start_pos` / `end_pos`: original source positions in the donor's prompt.
      Used to compute RoPE deltas at reuse time.
    """

    kv_indices: Any
    start_pos: int
    end_pos: int


class SemBlendProviderAdapter:
    """Semantic fuzzy-match provider backed by SemBlendPipeline.

    Contract (matches the Chenxin draft ABC):
      - `register_donor`: called from `cache_on_request_finished` on the SGLang
        side. Inserts the completed request's embedding + tokens + KV handle
        into the donor store (and optionally the gateway).
      - `match`: called from `match_on_prefix_miss` on the SGLang side.
        Runs the SemBlend pipeline, converts `PipelineResult` to
        `FuzzyMatchResult`.

    Not thread-safe for concurrent writes; reads are safe because
    SemBlendPipeline guards its own state. Callers should serialize
    `register_donor` (SGLang's scheduler does).
    """

    def __init__(
        self,
        config: SemBlendProviderConfig,
        *,
        pipeline: Any = None,
        gateway_client: Any = None,
    ) -> None:
        self._config = config
        self._pipeline = pipeline or self._build_pipeline(config)
        self._gateway = gateway_client or self._build_gateway(config)

        # Opaque KV handles keyed by donor request_id. Donor registration
        # inserts here; match lookups resolve donor_id -> handle to produce
        # pool-index tensors on return.
        self._donor_kv: dict[str, _DonorKVHandle] = {}

        self._stats = _Stats()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_pipeline(config: SemBlendProviderConfig) -> Any:
        """Lazy-import SemBlendPipeline so tests can inject a mock."""
        from semblend_core.bathtub import RecomputeConfig
        from semblend_core.pipeline import SemBlendPipeline

        return SemBlendPipeline(
            max_donors=config.max_entries,
            min_similarity=config.min_similarity,
            min_reuse_ratio=config.min_reuse_ratio,
            embedder_type=config.embedder_type,
            model_name=config.model_arch,
            chunk_size=config.block_size,
            recompute_config=RecomputeConfig.from_env(),
        )

    @staticmethod
    def _build_gateway(config: SemBlendProviderConfig) -> Any:
        if config.embedding_backend != "gateway":
            return None
        from semblend.integration.sglang.gateway_client import SynapseGatewayClient

        return SynapseGatewayClient(
            url=config.gateway_url or "",
            timeout_ms=config.gateway_timeout_ms,
        )

    # ------------------------------------------------------------------
    # Public API (invoked by the SGLang-side wrapper)
    # ------------------------------------------------------------------

    def register_donor(
        self,
        request_id: str,
        token_ids: List[int],
        kv_cache: Any,
        cache_start_pos: int,
        cache_end_pos: int,
        *,
        prompt_text: Optional[str] = None,
        extra_key: Optional[str] = None,
        radix_tree: Any = None,  # accepted for ABC compatibility; NodeRef path is v2
    ) -> bool:
        """Insert a completed request into the donor store.

        Args:
            request_id: Unique identifier for the donor (SGLang's ``req.rid``).
            token_ids: Full token sequence of the request.
            kv_cache: Opaque handle — pool-indices tensor or equivalent. Stored
                and returned verbatim in later match results.
            cache_start_pos: Source-position start (typically 0).
            cache_end_pos: Source-position end (exclusive).
            prompt_text: Optional pre-decoded prompt text. Required for
                embedding; the SGLang wrapper supplies this via its tokenizer.
            extra_key: Optional namespace tag (e.g., LoRA adapter ID).
            radix_tree: Reserved for the future NodeRef resolution path.

        Returns:
            True if the donor was registered, False otherwise.
        """
        segment_tokens = list(token_ids[cache_start_pos:cache_end_pos])

        if len(segment_tokens) < self._config.min_match_length:
            self._stats.register_rejected += 1
            return False

        embedding = self._embed(segment_tokens, prompt_text)
        if embedding is None:
            # No embedder available — gateway mode is still usable via token
            # indexing, but local mode requires an embedding. Treat as miss.
            self._stats.register_rejected += 1
            return False

        try:
            from semblend_core.donor_store import DonorNode

            node = DonorNode(
                request_id=request_id,
                token_ids=segment_tokens,
                embedding=embedding,
                timestamp=time.monotonic(),
                prompt_text=prompt_text or "",
            )
            # Pipeline exposes the donor store it owns; add via that path.
            self._pipeline._donor_store.add_donor(node)  # noqa: SLF001 — intentional
        except Exception as e:  # pragma: no cover — pipeline errors shouldn't block
            logger.warning("register_donor failed: %s", e, exc_info=True)
            self._stats.register_rejected += 1
            return False

        # Record the KV handle so match results can reference it.
        self._donor_kv[request_id] = _DonorKVHandle(
            kv_indices=kv_cache,
            start_pos=cache_start_pos,
            end_pos=cache_end_pos,
        )

        if self._gateway is not None:
            try:
                self._gateway.register(
                    donor_id=request_id,
                    embedding=embedding,
                    token_ids=segment_tokens,
                    extra_key=extra_key,
                )
            except Exception as e:  # pragma: no cover
                logger.warning("gateway register failed: %s", e, exc_info=True)

        self._stats.register_ok += 1
        return True

    def match(
        self,
        prompt_token_ids: List[int],
        already_matched_len: int,
        *,
        prompt_text: Optional[str] = None,
        extra_key: Optional[str] = None,
    ) -> Optional[FuzzyMatchResult]:
        """Look up a semantic match for the unmatched suffix of a prompt.

        Called from SGLang's `match_on_prefix_miss`. Returns None if no
        donor clears the cosine threshold or the reuse ratio is below the
        configured floor.
        """
        self._stats.match_calls += 1

        remaining = list(prompt_token_ids[already_matched_len:])
        if len(remaining) < self._config.min_match_length:
            return None

        try:
            result = self._pipeline.find_donor(
                token_ids=remaining,
                prompt_text=prompt_text or "",
                top_k=self._config.top_k,
            )
        except Exception as e:  # pragma: no cover
            logger.error("SemBlendPipeline.find_donor raised: %s", e, exc_info=True)
            self._stats.match_errors += 1
            return None

        if not getattr(result, "found", False):
            self._stats.match_misses += 1
            return None

        if result.similarity < self._config.min_similarity:
            self._stats.match_misses += 1
            return None

        if result.reuse_ratio < self._config.min_reuse_ratio:
            self._stats.match_rejected_low_reuse += 1
            return None

        converted = self._convert_result(
            pipeline_result=result,
            already_matched_len=already_matched_len,
            remaining=remaining,
        )
        if converted is None:
            self._stats.match_rejected_no_kv += 1
            return None

        # Discovery-only mode: SemBlend pipeline ran fully, found a real
        # donor, computed alignment + bathtub mask. We surface the hit
        # via QualitySignals (so telemetry sees it) but zero out the
        # KV-injection fields so SGLang's RadixCache treats it as a
        # miss for cache_protected_len / merged_value purposes. Avoids
        # the upstream _node_registry leak when running against an
        # unpatched RadixCache.
        if self._config.discovery_only:
            self._stats.match_hits_discovery_only += 1
            return FuzzyMatchResult(
                cached_token_count=0,
                cached_token_ids=[],
                prompt_token_count=0,
                kv_cache_indices=_empty_tensor_like(converted.kv_cache_indices),
                position_offset=already_matched_len,
                cached_start_pos=0,
                _match_entry=converted._match_entry,
                segments=None,
                layer_recompute_mask=None,
                quality_signals=converted.quality_signals,
            )

        self._stats.match_hits += 1
        return converted

    def stats(self) -> dict:
        """Return a plain-dict snapshot of adapter statistics (for logging/metrics)."""
        return self._stats.as_dict()

    def donor_count(self) -> int:
        return len(self._donor_kv)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _embed(
        self,
        token_ids: List[int],
        prompt_text: Optional[str],
    ) -> Optional[np.ndarray]:
        """Compute an embedding for the donor or the query.

        The pipeline's embedder handles GPU detection / CPU fallback.
        Returns None if no prompt_text was supplied and no tokens-to-text
        bridge is available.
        """
        if prompt_text:
            try:
                vec = self._pipeline._embedder.embed(prompt_text)  # noqa: SLF001
                if vec is None:
                    return None
                return np.asarray(vec, dtype=np.float32)
            except Exception as e:  # pragma: no cover
                logger.debug("embedder failed: %s", e)
                return None

        # No text supplied. The SGLang wrapper is responsible for decoding
        # tokens to text via its own tokenizer before calling register_donor.
        # We deliberately don't reach into SGLang here.
        return None

    def _convert_result(
        self,
        pipeline_result: Any,
        already_matched_len: int,
        remaining: List[int],
    ) -> Optional[FuzzyMatchResult]:
        """Translate `PipelineResult` → `FuzzyMatchResult`.

        Builds segments from `position_map` if N:M alignment is required;
        otherwise returns a contiguous-span result compatible with the
        non-semantic `TokenBlockMatchProvider` path.
        """
        donor_id = pipeline_result.donor_id
        if donor_id is None:
            return None
        handle = self._donor_kv.get(donor_id)
        if handle is None:
            # Donor's KV is gone (evicted, request failed). Ignore the match.
            return None

        # Build layer_recompute_mask from pipeline_result.layer_deviations.
        layer_mask = None
        if self._config.enable_bathtub and pipeline_result.layer_deviations:
            layer_mask = [
                bool(d.get("shouldRecompute", False))
                for d in pipeline_result.layer_deviations
            ]

        segments = self._build_segments(pipeline_result, handle, layer_mask)

        # cached_token_count: total target positions we're going to fill from KV.
        # When segments is None (single contiguous span) this equals
        # len(target_positions) in the single segment.
        if segments:
            total_covered = sum(
                len(_as_list(s.target_positions)) for s in segments
            )
        else:
            total_covered = 0

        # Quality signals
        quality = QualitySignals(
            cosine_similarity=float(pipeline_result.similarity),
            reuse_ratio=float(pipeline_result.reuse_ratio),
            confidence_tier=str(getattr(pipeline_result, "confidence_tier", "exact")),
            passed_quality_gate=True,
        )

        # For the single-segment case, also populate the legacy fields so a
        # model_runner on the old interface still gets enough to work with.
        single_segment = segments[0] if segments and len(segments) == 1 else None
        if single_segment is not None:
            kv_cache_indices = single_segment.donor_kv_indices
            cached_start_pos = int(_first(single_segment.donor_positions, 0))
            out_segments = None  # collapse to legacy contiguous path
        elif segments:
            kv_cache_indices = _empty_tensor_like(segments[0].donor_kv_indices)
            cached_start_pos = int(_first(segments[0].donor_positions, 0))
            out_segments = segments
        else:
            # No usable alignment: fall back to donor's full KV handle.
            kv_cache_indices = handle.kv_indices
            cached_start_pos = handle.start_pos
            out_segments = None
            total_covered = min(len(pipeline_result.donor_tokens), len(remaining))

        return FuzzyMatchResult(
            cached_token_count=total_covered,
            cached_token_ids=list(pipeline_result.donor_tokens[:total_covered]),
            prompt_token_count=total_covered,
            kv_cache_indices=kv_cache_indices,
            position_offset=already_matched_len,
            cached_start_pos=cached_start_pos,
            _match_entry=donor_id,
            segments=out_segments,
            layer_recompute_mask=layer_mask,
            quality_signals=quality,
        )

    def _build_segments(
        self,
        pipeline_result: Any,
        handle: _DonorKVHandle,
        layer_mask: Optional[List[bool]],
    ) -> List[FuzzyMatchSegment]:
        """Group `position_map` pairs into contiguous segments.

        Each segment is a run of (donor_pos, target_pos) pairs where both
        sides advance by +1 per step. Runs shorter than a single token are
        skipped (nothing to reuse).
        """
        pmap = getattr(pipeline_result, "position_map", None)
        if pmap is None:
            return []

        donor_positions = list(getattr(pmap, "donor_positions", []))
        target_positions = list(getattr(pmap, "target_positions", []))
        if not donor_positions or not target_positions:
            return []
        if len(donor_positions) != len(target_positions):
            logger.debug(
                "position_map mismatch: %d donor vs %d target positions",
                len(donor_positions),
                len(target_positions),
            )
            return []

        runs: List[tuple[int, int]] = []  # (start_idx, end_idx_exclusive)
        run_start = 0
        for i in range(1, len(donor_positions)):
            d_step = donor_positions[i] - donor_positions[i - 1]
            t_step = target_positions[i] - target_positions[i - 1]
            if d_step != 1 or t_step != 1:
                runs.append((run_start, i))
                run_start = i
        runs.append((run_start, len(donor_positions)))

        segments: List[FuzzyMatchSegment] = []
        for start, end in runs:
            if end - start < 1:
                continue
            seg_donor_positions = donor_positions[start:end]
            seg_target_positions = target_positions[start:end]
            kv_slice = _slice_indices(
                handle.kv_indices,
                offset=seg_donor_positions[0] - handle.start_pos,
                length=end - start,
            )
            segments.append(
                FuzzyMatchSegment(
                    donor_kv_indices=kv_slice,
                    target_positions=seg_target_positions,
                    donor_positions=seg_donor_positions,
                    donor_req_id=pipeline_result.donor_id,
                    layer_recompute_mask=layer_mask,
                )
            )
        return segments


@dataclass
class _Stats:
    register_ok: int = 0
    register_rejected: int = 0
    match_calls: int = 0
    match_hits: int = 0
    match_hits_discovery_only: int = 0
    match_misses: int = 0
    match_errors: int = 0
    match_rejected_low_reuse: int = 0
    match_rejected_no_kv: int = 0

    def as_dict(self) -> dict:
        return {
            "register_ok": self.register_ok,
            "register_rejected": self.register_rejected,
            "match_calls": self.match_calls,
            "match_hits": self.match_hits,
            "match_hits_discovery_only": self.match_hits_discovery_only,
            "match_misses": self.match_misses,
            "match_errors": self.match_errors,
            "match_rejected_low_reuse": self.match_rejected_low_reuse,
            "match_rejected_no_kv": self.match_rejected_no_kv,
        }


# ----------------------------------------------------------------------
# Tensor-agnostic helpers
# ----------------------------------------------------------------------


def _as_list(obj: Any) -> list:
    """Best-effort conversion to a Python list (supports torch/numpy/list/tuple)."""
    if obj is None:
        return []
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        return list(tolist())
    return list(obj)


def _first(obj: Any, default: Any) -> Any:
    seq = _as_list(obj)
    return seq[0] if seq else default


def _slice_indices(indices: Any, offset: int, length: int) -> Any:
    """Slice a sequence or tensor of KV indices.

    Works for lists, tuples, numpy arrays, and torch tensors. Callers may
    wrap the result with torch.as_tensor as needed on the engine side.
    """
    if indices is None:
        return None
    # torch.Tensor and numpy.ndarray both support slicing via __getitem__.
    try:
        return indices[offset : offset + length]
    except Exception:
        return list(indices)[offset : offset + length]


def _empty_tensor_like(template: Any) -> Any:
    """Return an empty slice of the same type as `template`.

    Used when `segments` collapses to the legacy path but we still want
    the right container type in `kv_cache_indices`.
    """
    if template is None:
        return []
    try:
        return template[0:0]
    except Exception:
        return []

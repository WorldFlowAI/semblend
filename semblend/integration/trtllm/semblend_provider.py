"""SemBlend reference implementation of SemanticCacheLookupProvider.

This is the concrete implementation that would be included in the
upstream PR to NVIDIA/TensorRT-LLM as the reference provider.

It bridges SemBlend's pipeline (embedding, donor lookup, alignment)
with the TRT-LLM SemanticCacheLookupProvider ABC, enabling semantic
KV cache reuse in TRT-LLM's prefix matching path.

Usage (after upstream PR is merged):
    from tensorrt_llm.kv_cache import KvCacheConfig
    from semblend.integration.trtllm.semblend_provider import SemBlendProvider

    config = KvCacheConfig(
        semantic_cache_provider=SemBlendProvider(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            min_similarity=0.60,
        ),
    )
    llm = LLM(model=model, kv_cache_config=config)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from semblend.integration.trtllm.namespace import namespace_key
from semblend.integration.trtllm.upstream_interface import (
    CacheNamespace,
    DonorEvicted,
    DonorLocation,
    DonorRegistered,
    DonorSegment,
    PostPrefixLoadHook,
    SemanticCacheLookupProvider,
    SemanticKvEngineAttentionMode,
    SemanticKvEngineExecution,
    SemanticKvLookupRequest,
    SemanticKvLookupResult,
    SemanticKvMaterializationKind,
    SemanticKvPlan,
    SemanticKvProvider,
    SemanticKvPublicationPolicy,
    SemanticKvSegment,
    SemanticMatchResult,
)

logger = logging.getLogger("semblend.trtllm.provider")


@dataclass
class _TensorRTDonorHandle:
    donor_id: str
    token_ids: list[int]
    namespace: CacheNamespace
    block_ids: tuple[int, ...]
    location: DonorLocation
    provider_generation: int
    block_hashes: tuple[int, ...] = ()


class SemBlendTensorRTProvider(SemanticKvProvider):
    """SemBlend semantic KV provider for TensorRT-LLM connectors."""

    def __init__(
        self,
        *,
        model_name: str = "",
        min_similarity: float = 0.60,
        min_reuse_ratio: float = 0.50,
        max_donors: int = 10_000,
        embedder_type: str = "minilm",
        chunk_size: int = 128,
        allow_segmented: bool | None = None,
        min_match_length: int = 128,
    ) -> None:
        self._model_name = model_name or os.environ.get("SEMBLEND_MODEL_NAME", "")
        self._min_similarity = min_similarity
        self._min_reuse_ratio = min_reuse_ratio
        self._max_donors = max_donors
        self._embedder_type = embedder_type
        self._chunk_size = chunk_size
        self._allow_segmented = (
            os.environ.get("SEMBLEND_TRTLLM_ENABLE_SEGMENTED", "0") == "1"
            if allow_segmented is None
            else allow_segmented
        )
        self._min_match_length = min_match_length
        self._pipeline = None
        self._tokenizer = None
        self._tokenizer_load_attempted = False
        self._generation = int(time.time() * 1000)
        self._events = 0
        self._donors: dict[str, _TensorRTDonorHandle] = {}
        self._stats = {
            "queries": 0,
            "hits": 0,
            "misses": 0,
            "registrations": 0,
            "evictions": 0,
            "materializations": 0,
            "rope_corrections": 0,
        }

    def lookup(self, request: SemanticKvLookupRequest) -> SemanticKvLookupResult:
        self._stats["queries"] += 1

        if len(request.token_ids) < self._min_match_length:
            self._stats["misses"] += 1
            return SemanticKvLookupResult(found=False, rejection_reason="short_request")

        fallback = (
            self._find_exact_prefix_donor(request) if _exact_prefix_fast_path_enabled() else None
        )
        if fallback is None and _token_prefix_fast_path_enabled():
            fallback = self._find_token_prefix_run_donor(request)
        if fallback is not None:
            plan = self._build_plan(request, fallback)
            if plan is not None:
                self._stats["hits"] += 1
                return SemanticKvLookupResult(
                    found=True,
                    plan=plan,
                    similarity=float(fallback.similarity),
                    reuse_ratio=float(fallback.reuse_ratio),
                    quality_signals={
                        "confidence_tier": str(fallback.confidence_tier),
                        "fuzzy_confidence": float(fallback.fuzzy_confidence),
                        "force_verify_layers": [],
                    },
                )

        pipeline = self._get_pipeline()
        if pipeline is None:
            self._stats["misses"] += 1
            return SemanticKvLookupResult(found=False, rejection_reason="provider_disabled")

        prompt_text = request.prompt_text or self._tokens_to_text(list(request.token_ids)) or ""
        result = pipeline.find_donor(
            token_ids=list(request.token_ids),
            prompt_text=prompt_text,
            extra_key=namespace_key(request.namespace),
        )

        if not getattr(result, "found", False):
            fallback = self._find_exact_prefix_donor(request)
            if fallback is None:
                self._stats["misses"] += 1
                return SemanticKvLookupResult(
                    found=False,
                    rejection_reason=getattr(result, "rejection_reason", "no_donor_match") or "",
                    timings_ms=_timings_dict(getattr(result, "timings", None)),
                )
            result = fallback

        if result.similarity < self._min_similarity:
            self._stats["misses"] += 1
            return SemanticKvLookupResult(found=False, rejection_reason="low_similarity")
        if result.reuse_ratio < self._min_reuse_ratio:
            self._stats["misses"] += 1
            return SemanticKvLookupResult(found=False, rejection_reason="low_reuse")

        plan = self._build_plan(request, result)
        if plan is None:
            self._stats["misses"] += 1
            return SemanticKvLookupResult(found=False, rejection_reason="donor_not_materializable")

        self._stats["hits"] += 1
        return SemanticKvLookupResult(
            found=True,
            plan=plan,
            similarity=float(result.similarity),
            reuse_ratio=float(result.reuse_ratio),
            quality_signals={
                "confidence_tier": str(getattr(result, "confidence_tier", "exact")),
                "fuzzy_confidence": float(getattr(result, "fuzzy_confidence", 1.0)),
                "force_verify_layers": list(getattr(result, "force_verify_layers", [])),
            },
            timings_ms=_timings_dict(getattr(result, "timings", None)),
        )

    def register_donor(self, event: DonorRegistered) -> None:
        self._donors[event.donor_id] = _TensorRTDonorHandle(
            donor_id=event.donor_id,
            token_ids=[],
            namespace=event.namespace,
            block_ids=tuple(event.block_ids),
            location=event.location,
            provider_generation=event.provider_generation,
            block_hashes=tuple(
                h for segment in event.segments for h in getattr(segment, "block_hashes", ())
            ),
        )

    def evict_donor(self, event: DonorEvicted) -> None:
        self._donors.pop(event.donor_id, None)
        self._stats["evictions"] += 1

    def clear(self) -> None:
        self._donors.clear()
        pipeline = self._pipeline
        clear = getattr(pipeline, "clear_donors", None)
        if callable(clear):
            clear()
        self._generation += 1

    def register_completed(
        self,
        *,
        request_id: str,
        token_ids: list[int],
        prompt_text: str,
        namespace: CacheNamespace,
        block_ids: list[int],
        block_hashes: list[int] | None = None,
        location: DonorLocation | None = None,
    ) -> DonorRegistered | None:
        if len(token_ids) < self._min_match_length:
            return None

        pipeline = self._get_pipeline()
        if pipeline is None:
            return None

        pipeline.register_donor(
            request_id=request_id,
            token_ids=token_ids,
            prompt_text=prompt_text,
            extra_key=namespace_key(namespace),
        )

        segment = DonorSegment(
            segment_id=0,
            token_range=(0, len(token_ids)),
            digest=_stable_token_digest(token_ids),
            block_hashes=tuple(block_hashes or ()),
        )
        event = DonorRegistered(
            donor_id=request_id,
            namespace=namespace,
            location=location or DonorLocation.worker(worker_id=0, dp_rank=0),
            token_count=len(token_ids),
            segments=(segment,),
            provider_generation=self._generation,
            block_ids=tuple(block_ids),
        )
        self._donors[request_id] = _TensorRTDonorHandle(
            donor_id=request_id,
            token_ids=list(token_ids),
            namespace=namespace,
            block_ids=tuple(block_ids),
            location=event.location,
            provider_generation=self._generation,
            block_hashes=tuple(block_hashes or ()),
        )
        self._events += 1
        self._stats["registrations"] += 1
        return event

    def materialize_plan(
        self,
        plan: SemanticKvPlan,
        kv_cache_tensor: Any,
        *,
        stream: Any = None,
    ) -> int:
        """Copy donor KV into target blocks and apply RoPE correction on K."""
        del stream
        if not plan.segments:
            return 0

        try:
            import torch
        except Exception:
            return 0

        materialized = 0
        with torch.no_grad():
            for segment in plan.segments:
                target_blocks = segment.target_block_ids
                donor_blocks = segment.donor_block_ids
                if not target_blocks or not donor_blocks:
                    continue
                mask = segment.layer_recompute_mask
                for donor_pos, target_pos, token_count in _position_runs(
                    segment.donor_positions,
                    segment.target_positions,
                    plan.namespace.block_size,
                ):
                    donor_block_idx = donor_pos // plan.namespace.block_size
                    target_block_idx = target_pos // plan.namespace.block_size
                    if donor_block_idx >= len(donor_blocks) or target_block_idx >= len(
                        target_blocks
                    ):
                        continue
                    donor_block = donor_blocks[donor_block_idx]
                    target_block = target_blocks[target_block_idx]
                    donor_offset = donor_pos % plan.namespace.block_size
                    target_offset = target_pos % plan.namespace.block_size
                    copied = _copy_token_range_kv(
                        kv_cache_tensor,
                        donor_block=donor_block,
                        donor_offset=donor_offset,
                        target_block=target_block,
                        target_offset=target_offset,
                        token_count=token_count,
                        block_size=plan.namespace.block_size,
                        layer_recompute_mask=mask,
                        rope_delta=target_pos - donor_pos,
                        rope_base=_rope_base(plan.namespace.rope_config),
                    )
                    materialized += copied

        self._stats["materializations"] += int(materialized > 0)
        if plan.requires_rope_correction:
            self._stats["rope_corrections"] += int(materialized > 0)
        return materialized

    def get_stats(self) -> dict:
        donor_count = len(self._donors)
        pipeline = self._pipeline
        if pipeline is not None:
            donor_count = max(donor_count, getattr(pipeline, "donor_count", donor_count))
        return {**self._stats, "donor_store_size": donor_count}

    def _find_exact_prefix_donor(self, request: SemanticKvLookupRequest) -> Any | None:
        best_donor = None
        best_len = 0
        request_tokens = list(request.token_ids)
        for donor in self._donors.values():
            if donor.namespace != request.namespace:
                continue
            common = _common_prefix_len(donor.token_ids, request_tokens)
            if common > best_len:
                best_donor = donor
                best_len = common

        if best_donor is None or best_len < self._min_match_length:
            return None

        reuse_ratio = best_len / max(len(request_tokens), 1)
        if reuse_ratio < self._min_reuse_ratio:
            return None

        return SimpleNamespace(
            found=True,
            donor_id=best_donor.donor_id,
            donor_tokens=best_donor.token_ids,
            similarity=1.0,
            reuse_ratio=reuse_ratio,
            position_map=SimpleNamespace(
                donor_positions=tuple(range(best_len)),
                target_positions=tuple(range(best_len)),
                num_pairs=best_len,
            ),
            slot_actions=[],
            layer_deviations=None,
            confidence_tier="exact_token_prefix",
            fuzzy_confidence=1.0,
            force_verify_layers=[],
            timings=None,
            chunk_fast_path_used=False,
        )

    def _find_token_prefix_run_donor(self, request: SemanticKvLookupRequest) -> Any | None:
        best_donor = None
        best_start = 0
        best_len = 0
        request_tokens = list(request.token_ids)
        for donor in self._donors.values():
            if donor.namespace != request.namespace:
                continue
            start, common = _longest_target_prefix_run(donor.token_ids, request_tokens)
            if common > best_len:
                best_donor = donor
                best_start = start
                best_len = common

        if best_donor is None or best_len < self._min_match_length:
            return None

        reuse_ratio = best_len / max(len(request_tokens), 1)
        if reuse_ratio < self._min_reuse_ratio:
            return None

        return SimpleNamespace(
            found=True,
            donor_id=best_donor.donor_id,
            donor_tokens=best_donor.token_ids,
            similarity=reuse_ratio,
            reuse_ratio=reuse_ratio,
            position_map=SimpleNamespace(
                donor_positions=tuple(range(best_start, best_start + best_len)),
                target_positions=tuple(range(best_len)),
                num_pairs=best_len,
            ),
            slot_actions=[],
            layer_deviations=None,
            confidence_tier="token_prefix_run",
            fuzzy_confidence=reuse_ratio,
            force_verify_layers=[],
            timings=None,
            chunk_fast_path_used=False,
        )

    def _build_plan(self, request: SemanticKvLookupRequest, result: Any) -> SemanticKvPlan | None:
        layer_mask = _layer_recompute_mask(getattr(result, "layer_deviations", None))
        layer_mask = _apply_forced_recompute_layers(layer_mask)
        segments = self._segments_from_result(request, result, layer_mask)
        if not segments:
            return None

        request_token_count = len(request.token_ids)
        covered = min(sum(segment.token_count for segment in segments), request_token_count)
        prefix = min(_prefix_token_count(segments), request_token_count)
        kind = (
            SemanticKvMaterializationKind.REQUEST_LOCAL_SEGMENTED
            if len(segments) > 1 or prefix < covered
            else SemanticKvMaterializationKind.REQUEST_LOCAL_PREFIX
        )
        if kind == SemanticKvMaterializationKind.REQUEST_LOCAL_SEGMENTED and not (
            self._allow_segmented and request.allow_segmented
        ):
            segments = (_longest_prefixable_segment(segments),)
            covered = min(segments[0].token_count, request_token_count)
            prefix = min(_prefix_token_count(segments), request_token_count)
            kind = SemanticKvMaterializationKind.REQUEST_LOCAL_PREFIX

        engine_execution = _engine_execution_policy(
            prefix_token_count=prefix,
            layer_mask=layer_mask,
        )

        return SemanticKvPlan(
            request_id=request.request_id,
            namespace=request.namespace,
            kind=kind,
            publication_policy=SemanticKvPublicationPolicy.REQUEST_LOCAL,
            segments=tuple(segments),
            donor_ids=tuple(dict.fromkeys(segment.donor_id for segment in segments)),
            covered_token_count=covered,
            prefix_token_count=prefix,
            computed_token_count=prefix if _trust_non_identical_kv() else 0,
            requires_rope_correction=any(
                d != t
                for segment in segments
                for d, t in zip(segment.donor_positions, segment.target_positions)
            ),
            engine_execution=engine_execution,
            diagnostics={
                "confidence_tier": str(getattr(result, "confidence_tier", "exact")),
                "chunk_fast_path": bool(getattr(result, "chunk_fast_path_used", False)),
                "engine_execution": (
                    engine_execution.to_dict() if engine_execution is not None else None
                ),
            },
        )

    def _segments_from_result(
        self,
        request: SemanticKvLookupRequest,
        result: Any,
        layer_mask: tuple[bool, ...] | None,
    ) -> list[SemanticKvSegment]:
        slot_actions = list(getattr(result, "slot_actions", []) or [])
        if not slot_actions:
            pmap = getattr(result, "position_map", None)
            donor_positions = tuple(getattr(pmap, "donor_positions", ()) or ())
            target_positions = tuple(getattr(pmap, "target_positions", ()) or ())
            if not donor_positions or not target_positions:
                n = min(len(getattr(result, "donor_tokens", ())), len(request.token_ids))
                donor_positions = tuple(range(n))
                target_positions = tuple(range(n))
            return self._group_positions(
                donor_id=str(result.donor_id),
                donor_positions=donor_positions,
                target_positions=target_positions,
                request_token_count=len(request.token_ids),
                layer_mask=layer_mask,
                namespace=request.namespace,
            )

        by_donor: dict[str, tuple[list[int], list[int]]] = {}
        default_donor = str(getattr(result, "donor_id", "") or "")
        for action in slot_actions:
            if action.get("action") != "copy_from_donor":
                continue
            donor_id = str(action.get("donorId") or default_donor)
            donor_pos = action.get("donorPos")
            target_pos = action.get("targetPos")
            if donor_pos is None or target_pos is None:
                continue
            donor_positions, target_positions = by_donor.setdefault(donor_id, ([], []))
            donor_positions.append(int(donor_pos))
            target_positions.append(int(target_pos))

        segments: list[SemanticKvSegment] = []
        for donor_id, (donor_positions, target_positions) in by_donor.items():
            segments.extend(
                self._group_positions(
                    donor_id=donor_id,
                    donor_positions=tuple(donor_positions),
                    target_positions=tuple(target_positions),
                    request_token_count=len(request.token_ids),
                    layer_mask=layer_mask,
                    namespace=request.namespace,
                )
            )
        return segments

    def _group_positions(
        self,
        *,
        donor_id: str,
        donor_positions: tuple[int, ...],
        target_positions: tuple[int, ...],
        request_token_count: int,
        layer_mask: tuple[bool, ...] | None,
        namespace: CacheNamespace,
    ) -> list[SemanticKvSegment]:
        handle = self._donors.get(donor_id)
        if handle is None or not handle.block_ids:
            return []
        if handle.namespace != namespace:
            return []

        pairs = sorted(
            (
                (donor_pos, target_pos)
                for donor_pos, target_pos in zip(donor_positions, target_positions)
                if donor_pos >= 0 and 0 <= target_pos < request_token_count
            ),
            key=lambda p: (p[1], p[0]),
        )
        runs: list[list[tuple[int, int]]] = []
        current: list[tuple[int, int]] = []
        for pair in pairs:
            if current:
                prev_d, prev_t = current[-1]
                if pair[0] != prev_d + 1 or pair[1] != prev_t + 1:
                    runs.append(current)
                    current = []
            current.append(pair)
        if current:
            runs.append(current)

        segments = []
        for idx, run in enumerate(runs):
            segments.append(
                SemanticKvSegment(
                    donor_id=donor_id,
                    donor_segment_id=idx,
                    donor_positions=tuple(d for d, _ in run),
                    target_positions=tuple(t for _, t in run),
                    donor_block_ids=handle.block_ids,
                    layer_recompute_mask=layer_mask,
                    donor_location=handle.location,
                )
            )
        return segments

    def _get_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        if os.environ.get("SEMBLEND_ENABLED", "1") != "1":
            return None

        from semblend_core.pipeline import SemBlendPipeline

        self._pipeline = SemBlendPipeline(
            max_donors=self._max_donors,
            min_similarity=self._min_similarity,
            min_reuse_ratio=self._min_reuse_ratio,
            embedder_type=self._embedder_type,
            model_name=self._model_name,
            chunk_size=self._chunk_size,
        )
        return self._pipeline

    def _get_tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer
        if self._tokenizer_load_attempted:
            return None
        self._tokenizer_load_attempted = True
        if not self._model_name:
            return None
        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_name,
                trust_remote_code=True,
            )
        except Exception:
            logger.debug("failed to load tokenizer for %s", self._model_name, exc_info=True)
        return self._tokenizer

    def _tokens_to_text(self, token_ids: list[int]) -> str | None:
        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            return None
        return tokenizer.decode(token_ids[:2000], skip_special_tokens=True)


def _timings_dict(timings: Any) -> dict[str, float]:
    if timings is None:
        return {}
    return {
        "embed_ms": float(getattr(timings, "embed_ms", 0.0)),
        "lookup_ms": float(getattr(timings, "lookup_ms", 0.0)),
        "align_ms": float(getattr(timings, "align_ms", 0.0)),
        "bathtub_ms": float(getattr(timings, "bathtub_ms", 0.0)),
        "total_ms": float(getattr(timings, "total_ms", 0.0)),
    }


def _stable_token_digest(token_ids: list[int]) -> int:
    import hashlib

    payload = ",".join(str(token) for token in token_ids).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def _common_prefix_len(left: list[int], right: list[int]) -> int:
    count = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        count += 1
    return count


def _longest_target_prefix_run(donor: list[int], target: list[int]) -> tuple[int, int]:
    if not donor or not target:
        return 0, 0
    best_start = 0
    best_len = 0
    first = target[0]
    for start, token in enumerate(donor):
        if token != first:
            continue
        common = 0
        max_count = min(len(donor) - start, len(target))
        while common < max_count and donor[start + common] == target[common]:
            common += 1
        if common > best_len:
            best_start = start
            best_len = common
    return best_start, best_len


def _position_runs(
    donor_positions: tuple[int, ...],
    target_positions: tuple[int, ...],
    block_size: int,
) -> list[tuple[int, int, int]]:
    pairs = sorted(zip(donor_positions, target_positions), key=lambda p: (p[1], p[0]))
    runs = []
    idx = 0
    while idx < len(pairs):
        donor_start, target_start = pairs[idx]
        count = 1
        donor_block_remaining = block_size - donor_start % block_size
        target_block_remaining = block_size - target_start % block_size
        max_count = max(1, min(donor_block_remaining, target_block_remaining))
        while idx + count < len(pairs) and count < max_count:
            donor_pos, target_pos = pairs[idx + count]
            if donor_pos != donor_start + count or target_pos != target_start + count:
                break
            count += 1
        runs.append((donor_start, target_start, count))
        idx += count
    return runs


def _layer_recompute_mask(layer_deviations: Any) -> tuple[bool, ...] | None:
    if not layer_deviations:
        return None
    return tuple(bool(d.get("shouldRecompute", False)) for d in layer_deviations)


def _apply_forced_recompute_layers(mask: tuple[bool, ...] | None) -> tuple[bool, ...] | None:
    raw = os.environ.get("SEMBLEND_TRTLLM_FORCE_RECOMPUTE_LAYERS", "")
    if not raw:
        raw = os.environ.get("SEMBLEND_FORCE_RECOMPUTE_LAYERS", "")
    if not raw:
        return mask
    forced = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            forced.append(int(part))
        except ValueError:
            continue
    if not forced:
        return mask
    size = max(max(forced) + 1, len(mask or ()))
    values = list(mask or (False,) * size)
    if len(values) < size:
        values.extend([False] * (size - len(values)))
    for layer_idx in forced:
        if layer_idx >= 0:
            values[layer_idx] = True
    return tuple(values)


def _forced_recompute_layers() -> tuple[int, ...]:
    raw = os.environ.get("SEMBLEND_TRTLLM_FORCE_RECOMPUTE_LAYERS", "")
    if not raw:
        raw = os.environ.get("SEMBLEND_FORCE_RECOMPUTE_LAYERS", "")
    layers = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            layer = int(part)
        except ValueError:
            continue
        if layer >= 0:
            layers.append(layer)
    return tuple(dict.fromkeys(layers))


def _recompute_boundary_layer(layer_mask: tuple[bool, ...] | None) -> int | None:
    raw = os.environ.get("SEMBLEND_TRTLLM_RECOMPUTE_LAYERS")
    if raw is None:
        raw = os.environ.get("SEMBLEND_TRTLLM_RECOMPUTE_BOUNDARY_LAYER")
    if raw is not None:
        try:
            boundary = int(raw)
        except ValueError:
            boundary = -1
        return boundary if boundary >= 0 else None
    if not layer_mask:
        return None
    boundary = 0
    for should_recompute in layer_mask:
        if not should_recompute:
            break
        boundary += 1
    return boundary or None


def _engine_execution_policy(
    *,
    prefix_token_count: int,
    layer_mask: tuple[bool, ...] | None,
) -> SemanticKvEngineExecution | None:
    if os.environ.get("SEMBLEND_TRTLLM_ENGINE_BLEND", "0") != "1":
        return None
    if prefix_token_count <= 0:
        return None
    boundary = _recompute_boundary_layer(layer_mask)
    if boundary is None:
        return None
    return SemanticKvEngineExecution(
        attention_mode=SemanticKvEngineAttentionMode.SUFFIX_ONLY_AFTER_PREFIX,
        materialized_prefix_token_count=prefix_token_count,
        suffix_start_position=prefix_token_count,
        recompute_boundary_layer=boundary,
        force_recompute_layers=_forced_recompute_layers(),
        require_materialization_barrier=True,
    )


def _prefix_token_count(segments: list[SemanticKvSegment] | tuple[SemanticKvSegment, ...]) -> int:
    covered = 0
    for segment in sorted(segments, key=lambda s: min(s.target_positions, default=10**12)):
        pairs = sorted(zip(segment.donor_positions, segment.target_positions), key=lambda p: p[1])
        for _, target_pos in pairs:
            if target_pos != covered:
                return covered
            covered += 1
    return covered


def _longest_prefixable_segment(segments: list[SemanticKvSegment]) -> SemanticKvSegment:
    return max(
        segments,
        key=lambda segment: sum(
            1 for i, target in enumerate(segment.target_positions) if target == i
        ),
    )


def _copy_token_kv(
    kv_cache_tensor: Any,
    *,
    donor_block: int,
    donor_offset: int,
    target_block: int,
    target_offset: int,
    block_size: int,
    layer_recompute_mask: tuple[bool, ...] | None,
    rope_delta: int,
    rope_base: float,
) -> int:
    """Copy one token's KV from donor to target for common TRT-LLM pool layouts."""

    if getattr(kv_cache_tensor, "ndim", 0) == 4:
        # TensorRT-LLM PyTorch primary pool:
        # [blocks, layers, kv, tokens_per_block * kv_heads * head_dim].
        # FlashInfer uses HND pages: [blocks, kv, heads, tokens, head_dim].
        blocks, layers, kv_factor, flat = kv_cache_tensor.shape
        del blocks
        if kv_factor != 2 or block_size <= 0 or flat % block_size != 0:
            return 0
        head_dim = _infer_head_dim(flat, block_size)
        if head_dim <= 0:
            return 0
        num_heads = flat // (block_size * head_dim)
        if num_heads <= 0:
            return 0
        layout = _trtllm_flat_layout()
        if layout == "nhd":
            view = kv_cache_tensor.view(
                kv_cache_tensor.shape[0],
                layers,
                kv_factor,
                block_size,
                num_heads,
                head_dim,
            )
        else:
            view = kv_cache_tensor.view(
                kv_cache_tensor.shape[0],
                layers,
                kv_factor,
                num_heads,
                block_size,
                head_dim,
            )
        copied = 0
        for layer_idx in range(layers):
            if layer_recompute_mask is not None and layer_idx < len(layer_recompute_mask):
                if layer_recompute_mask[layer_idx]:
                    continue
            if layout == "nhd":
                src = view[donor_block, layer_idx, :, donor_offset, :, :]
                dst = view[target_block, layer_idx, :, target_offset, :, :]
            else:
                src = view[donor_block, layer_idx, :, :, donor_offset, :]
                dst = view[target_block, layer_idx, :, :, target_offset, :]
            dst.copy_(src)
            if rope_delta:
                dst[0].copy_(_rope_correct_k(dst[0], rope_delta, rope_base))
            copied += 1
        return copied

    if getattr(kv_cache_tensor, "ndim", 0) == 6:
        # [blocks, layers, kv, tokens, heads, head_dim]
        layers = kv_cache_tensor.shape[1]
        copied = 0
        for layer_idx in range(layers):
            if layer_recompute_mask is not None and layer_idx < len(layer_recompute_mask):
                if layer_recompute_mask[layer_idx]:
                    continue
            src = kv_cache_tensor[donor_block, layer_idx, :, donor_offset, :, :]
            dst = kv_cache_tensor[target_block, layer_idx, :, target_offset, :, :]
            dst.copy_(src)
            if rope_delta:
                dst[0].copy_(_rope_correct_k(dst[0], rope_delta, rope_base))
            copied += 1
        return copied

    if getattr(kv_cache_tensor, "ndim", 0) == 5:
        # [blocks, kv, tokens, heads, head_dim], usually a per-layer view.
        src = kv_cache_tensor[donor_block, :, donor_offset, :, :]
        dst = kv_cache_tensor[target_block, :, target_offset, :, :]
        dst.copy_(src)
        if rope_delta:
            dst[0].copy_(_rope_correct_k(dst[0], rope_delta, rope_base))
        return 1

    if getattr(kv_cache_tensor, "ndim", 0) >= 1:
        kv_cache_tensor[target_block].copy_(kv_cache_tensor[donor_block])
        return 1

    return 0


def _copy_token_range_kv(
    kv_cache_tensor: Any,
    *,
    donor_block: int,
    donor_offset: int,
    target_block: int,
    target_offset: int,
    token_count: int,
    block_size: int,
    layer_recompute_mask: tuple[bool, ...] | None,
    rope_delta: int,
    rope_base: float,
) -> int:
    if token_count <= 1:
        return _copy_token_kv(
            kv_cache_tensor,
            donor_block=donor_block,
            donor_offset=donor_offset,
            target_block=target_block,
            target_offset=target_offset,
            block_size=block_size,
            layer_recompute_mask=layer_recompute_mask,
            rope_delta=rope_delta,
            rope_base=rope_base,
        )

    if getattr(kv_cache_tensor, "ndim", 0) == 4:
        blocks, layers, kv_factor, flat = kv_cache_tensor.shape
        del blocks
        if kv_factor != 2 or block_size <= 0 or flat % block_size != 0:
            return 0
        head_dim = _infer_head_dim(flat, block_size)
        if head_dim <= 0:
            return 0
        num_heads = flat // (block_size * head_dim)
        if num_heads <= 0:
            return 0
        layout = _trtllm_flat_layout()
        if layout == "nhd":
            view = kv_cache_tensor.view(
                kv_cache_tensor.shape[0],
                layers,
                kv_factor,
                block_size,
                num_heads,
                head_dim,
            )
        else:
            view = kv_cache_tensor.view(
                kv_cache_tensor.shape[0],
                layers,
                kv_factor,
                num_heads,
                block_size,
                head_dim,
            )
        copied = 0
        donor_slice = slice(donor_offset, donor_offset + token_count)
        target_slice = slice(target_offset, target_offset + token_count)
        for layer_idx in range(layers):
            if layer_recompute_mask is not None and layer_idx < len(layer_recompute_mask):
                if layer_recompute_mask[layer_idx]:
                    continue
            if layout == "nhd":
                src = view[donor_block, layer_idx, :, donor_slice, :, :]
                dst = view[target_block, layer_idx, :, target_slice, :, :]
            else:
                src = view[donor_block, layer_idx, :, :, donor_slice, :]
                dst = view[target_block, layer_idx, :, :, target_slice, :]
            dst.copy_(src)
            if rope_delta:
                dst[0].copy_(_rope_correct_k(dst[0], rope_delta, rope_base))
            copied += token_count
        return copied

    if getattr(kv_cache_tensor, "ndim", 0) == 6:
        layers = kv_cache_tensor.shape[1]
        donor_slice = slice(donor_offset, donor_offset + token_count)
        target_slice = slice(target_offset, target_offset + token_count)
        copied = 0
        for layer_idx in range(layers):
            if layer_recompute_mask is not None and layer_idx < len(layer_recompute_mask):
                if layer_recompute_mask[layer_idx]:
                    continue
            src = kv_cache_tensor[donor_block, layer_idx, :, donor_slice, :, :]
            dst = kv_cache_tensor[target_block, layer_idx, :, target_slice, :, :]
            dst.copy_(src)
            if rope_delta:
                dst[0].copy_(_rope_correct_k(dst[0], rope_delta, rope_base))
            copied += token_count
        return copied

    if getattr(kv_cache_tensor, "ndim", 0) == 5:
        donor_slice = slice(donor_offset, donor_offset + token_count)
        target_slice = slice(target_offset, target_offset + token_count)
        src = kv_cache_tensor[donor_block, :, donor_slice, :, :]
        dst = kv_cache_tensor[target_block, :, target_slice, :, :]
        dst.copy_(src)
        if rope_delta:
            dst[0].copy_(_rope_correct_k(dst[0], rope_delta, rope_base))
        return token_count

    copied = 0
    for offset in range(token_count):
        copied += _copy_token_kv(
            kv_cache_tensor,
            donor_block=donor_block,
            donor_offset=donor_offset + offset,
            target_block=target_block,
            target_offset=target_offset + offset,
            block_size=block_size,
            layer_recompute_mask=layer_recompute_mask,
            rope_delta=rope_delta,
            rope_base=rope_base,
        )
    return copied


def _infer_head_dim(flat: int, block_size: int) -> int:
    requested = os.environ.get("SEMBLEND_TRTLLM_HEAD_DIM")
    if requested:
        head_dim = int(requested)
        if head_dim > 0 and flat % (block_size * head_dim) == 0:
            return head_dim
    for head_dim in (256, 192, 160, 128, 96, 80, 64):
        if flat % (block_size * head_dim) == 0:
            return head_dim
    return 0


def _trtllm_flat_layout() -> str:
    layout = os.environ.get("SEMBLEND_TRTLLM_FLAT_LAYOUT", "hnd").strip().lower()
    if layout in {"nhd", "token_major", "token-major"}:
        return "nhd"
    return "hnd"


def _trust_non_identical_kv() -> bool:
    return os.environ.get("SEMBLEND_TRTLLM_TRUST_NON_IDENTICAL_KV", "0") == "1"


def _exact_prefix_fast_path_enabled() -> bool:
    return os.environ.get("SEMBLEND_TRTLLM_EXACT_PREFIX_FAST_PATH", "0") == "1"


def _token_prefix_fast_path_enabled() -> bool:
    return os.environ.get("SEMBLEND_TRTLLM_TOKEN_PREFIX_FAST_PATH", "0") == "1"


def _rope_correct_k(k_tensor: Any, delta: int, rope_base: float) -> Any:
    import torch

    head_dim = k_tensor.shape[-1]
    if head_dim % 2 != 0:
        return k_tensor
    if _rope_style() == "interleaved":
        inv_freq = 1.0 / (
            rope_base
            ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32, device=k_tensor.device) / head_dim
            )
        )
        angles = float(delta) * inv_freq
        cos = torch.cos(angles).to(k_tensor.dtype)
        sin = torch.sin(angles).to(k_tensor.dtype)
        even = k_tensor[..., 0::2]
        odd = k_tensor[..., 1::2]
        out = torch.empty_like(k_tensor)
        out[..., 0::2] = even * cos - odd * sin
        out[..., 1::2] = even * sin + odd * cos
        return out

    half = head_dim // 2
    inv_freq = 1.0 / (
        rope_base
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=k_tensor.device) / head_dim)
    )
    angles = float(delta) * inv_freq
    cos = torch.cos(angles).to(k_tensor.dtype)
    sin = torch.sin(angles).to(k_tensor.dtype)
    first = k_tensor[..., :half]
    second = k_tensor[..., half:]
    out = torch.empty_like(k_tensor)
    out[..., :half] = first * cos - second * sin
    out[..., half:] = second * cos + first * sin
    return out


def _rope_base(rope_config: dict[str, Any]) -> float:
    env_value = os.environ.get("SEMBLEND_TRTLLM_ROPE_BASE")
    if env_value:
        return float(env_value)
    return float(rope_config.get("rope_theta") or rope_config.get("rope_base") or 10000.0)


def _rope_style() -> str:
    style = os.environ.get("SEMBLEND_TRTLLM_ROPE_STYLE", "half").strip().lower()
    if style in {"interleaved", "gptj", "pairwise"}:
        return "interleaved"
    return "half"


class SemBlendProvider(SemanticCacheLookupProvider):
    """SemBlend implementation of TRT-LLM's SemanticCacheLookupProvider.

    Provides semantic donor discovery for TRT-LLM's KV cache system.
    When the radix tree exact-prefix match fails, this provider searches
    for semantically similar cached prompts using MiniLM embeddings.

    Args:
        model_name: Model name for tokenizer and bathtub preset lookup.
        min_similarity: Minimum cosine similarity for donor matching.
        min_reuse_ratio: Minimum alignment reuse ratio.
        max_donors: Maximum entries in the donor store.
        embedder_type: Embedder type ("minilm", "onnx-gpu", "jaccard").
        chunk_size: KV block size for alignment (default 128 for TRT-LLM).
    """

    def __init__(
        self,
        model_name: str = "",
        min_similarity: float = 0.60,
        min_reuse_ratio: float = 0.50,
        max_donors: int = 10_000,
        embedder_type: str = "minilm",
        chunk_size: int = 128,
    ) -> None:
        self._model_name = model_name or os.environ.get("SEMBLEND_MODEL_NAME", "")
        self._min_similarity = min_similarity
        self._min_reuse_ratio = min_reuse_ratio
        self._max_donors = max_donors
        self._embedder_type = embedder_type
        self._chunk_size = chunk_size

        self._pipeline = None
        self._tokenizer = None
        self._tokenizer_load_attempted = False

        self._stats = {
            "queries": 0,
            "hits": 0,
            "misses": 0,
            "registrations": 0,
            "evictions": 0,
            "avg_query_ms": 0.0,
        }

        logger.info(
            "SemBlendProvider initialized: model=%s, min_sim=%.2f, chunk_size=%d, max_donors=%d",
            self._model_name,
            self._min_similarity,
            self._chunk_size,
            self._max_donors,
        )

    def find_semantic_match(
        self,
        token_ids: list[int],
        prompt_text: str,
    ) -> SemanticMatchResult | None:
        """Find a semantically similar cached prompt.

        Called when exact prefix match is shorter than threshold.
        Runs the full SemBlend pipeline: embed -> search -> align.

        Args:
            token_ids: Token IDs of the incoming request.
            prompt_text: Decoded prompt text for embedding.

        Returns:
            SemanticMatchResult on hit, None on miss.
        """
        self._stats["queries"] += 1
        t0 = time.monotonic()

        pipeline = self._get_pipeline()
        if pipeline is None:
            self._stats["misses"] += 1
            return None

        # Use provided prompt_text, or decode from tokens
        if not prompt_text:
            prompt_text = self._tokens_to_text(token_ids) or ""
        if not prompt_text or len(token_ids) < 100:
            self._stats["misses"] += 1
            return None

        result = pipeline.find_donor(
            token_ids=token_ids,
            prompt_text=prompt_text,
        )

        elapsed_ms = (time.monotonic() - t0) * 1000
        self._update_avg_ms(elapsed_ms)

        if not result.found:
            self._stats["misses"] += 1
            return None

        self._stats["hits"] += 1

        # Build position mapping from slot actions
        position_mapping = []
        if result.position_map and result.position_map.num_pairs > 0:
            position_mapping = list(
                zip(
                    result.position_map.donor_positions,
                    result.position_map.target_positions,
                )
            )

        logger.info(
            "Semantic hit: donor=%s sim=%.3f reuse=%.2f pairs=%d (%.1fms)",
            result.donor_id,
            result.similarity,
            result.reuse_ratio,
            len(position_mapping),
            elapsed_ms,
        )

        return SemanticMatchResult(
            donor_token_ids=result.donor_tokens,
            similarity=result.similarity,
            reuse_ratio=result.reuse_ratio,
            position_mapping=position_mapping,
            donor_id=result.donor_id or "",
            metadata={
                "timings": {
                    "embed_ms": result.timings.embed_ms,
                    "lookup_ms": result.timings.lookup_ms,
                    "total_ms": result.timings.total_ms,
                },
            },
        )

    def register_completed(
        self,
        request_id: str,
        token_ids: list[int],
        prompt_text: str,
    ) -> None:
        """Register a completed request as a potential donor."""
        pipeline = self._get_pipeline()
        if pipeline is None:
            return

        if not prompt_text:
            prompt_text = self._tokens_to_text(token_ids) or ""
        if not prompt_text or len(token_ids) < 100:
            return

        pipeline.register_donor(
            request_id=request_id,
            token_ids=token_ids,
            prompt_text=prompt_text,
        )
        self._stats["registrations"] += 1

    def on_eviction(self, request_id: str) -> None:
        """Handle donor eviction (no-op for in-memory store)."""
        self._stats["evictions"] += 1

    def get_stats(self) -> dict:
        pipeline = self._pipeline
        donor_count = pipeline.donor_count if pipeline is not None else 0
        return {
            **self._stats,
            "donor_store_size": donor_count,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        enabled = os.environ.get("SEMBLEND_ENABLED", "1") == "1"
        if not enabled:
            return None

        from semblend_core.pipeline import SemBlendPipeline

        self._pipeline = SemBlendPipeline(
            max_donors=self._max_donors,
            min_similarity=self._min_similarity,
            min_reuse_ratio=self._min_reuse_ratio,
            embedder_type=self._embedder_type,
            model_name=self._model_name,
            chunk_size=self._chunk_size,
        )
        return self._pipeline

    def _get_tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer
        if self._tokenizer_load_attempted:
            return None
        self._tokenizer_load_attempted = True

        if not self._model_name:
            return None

        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_name, trust_remote_code=True
            )
        except Exception as e:
            logger.error("Failed to load tokenizer: %s", e)
        return self._tokenizer

    def _tokens_to_text(self, token_ids: list[int]) -> str | None:
        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            return None
        max_decode = 2000
        sampled = token_ids[:max_decode] if len(token_ids) > max_decode else token_ids
        return tokenizer.decode(sampled, skip_special_tokens=True)

    def _update_avg_ms(self, elapsed_ms: float) -> None:
        n = self._stats["hits"] + self._stats["misses"]
        if n > 0:
            self._stats["avg_query_ms"] = (self._stats["avg_query_ms"] * (n - 1) + elapsed_ms) / n


class SemBlendPostLoadHook(PostPrefixLoadHook):
    """SemBlend implementation of TRT-LLM's PostPrefixLoadHook.

    Applies RoPE position correction to K cache tensors after
    semantic donor KV is loaded. Uses the existing SemBlend
    Triton kernels with TRT-LLM stride computation.

    Args:
        rope_base: RoPE base frequency (default 10000.0).
    """

    def __init__(self, rope_base: float = 10000.0) -> None:
        self._rope_base = rope_base
        self._stats = {
            "corrections_applied": 0,
            "total_pairs_corrected": 0,
        }

    def on_prefix_loaded(
        self,
        kv_buffers: list[Any],
        block_table: Any,
        position_mapping: list[tuple[int, int]],
        rope_config: dict,
    ) -> None:
        """Apply RoPE correction after prefix KV load.

        For each layer's KV buffer, applies RoPE(target_pos - donor_pos)
        correction to the K cache. V cache is unchanged.
        """
        if not position_mapping:
            return

        # Check if any position pairs actually need correction
        needs_correction = any(d != t for d, t in position_mapping)
        if not needs_correction:
            return

        rope_base = rope_config.get("rope_base", self._rope_base)

        from semblend_core.rope_correction import apply_rope_delta_inplace

        total_corrected = 0
        for layer_idx, kv_buffer in enumerate(kv_buffers):
            for donor_pos, target_pos in position_mapping:
                delta = target_pos - donor_pos
                if delta != 0:
                    modified = apply_rope_delta_inplace(
                        kv_cache=kv_buffer,
                        block_table=block_table,
                        positions=[target_pos],
                        delta=delta,
                        rope_base=rope_base,
                    )
                    total_corrected += modified

        self._stats["corrections_applied"] += 1
        self._stats["total_pairs_corrected"] += total_corrected

        logger.debug(
            "RoPE correction: %d layers, %d pairs, %d positions modified",
            len(kv_buffers),
            len(position_mapping),
            total_corrected,
        )

    def get_stats(self) -> dict:
        return {**self._stats}

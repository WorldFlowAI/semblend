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
4. **Process-local** — in-process MiniLM embedding and a numpy donor
   store. No service dependency; no network hop. The adapter stores
   donor KV as an opaque object (`Any`) — typically a pool-indices
   torch.Tensor — so callers can choose the representation.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

from semblend.integration.sglang.config import SemBlendProviderConfig
from semblend.integration.sglang.sparse_plan import (
    build_sparse_plan,
    sparse_plan_enabled,
)
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
    - `last_node_id`: ID of the donor's TreeNode in the radix tree
      (``radix_tree._node_registry``). Populated by ``on_donor_inserted``
      after the radix insert completes. Surfaced as
      ``FuzzyMatchResult.donor_last_node_id`` so RadixCache.match_prefix can
      ``inc_lock_ref`` the donor and prevent LRU eviction while a recipient
      request is consuming its KV.
    """

    kv_indices: Any
    start_pos: int
    end_pos: int
    extra_key: Optional[str] = None
    last_node_id: Optional[int] = None
    # Canonical matching: donor prompt text + engine token ids for
    # tokenization-invariant alignment; retained only when
    # SEMBLEND_CANONICAL_MATCH=1.
    prompt_text: Optional[str] = None
    token_ids: Optional[List[int]] = None


def _canonical_match_enabled() -> bool:
    return os.environ.get("SEMBLEND_CANONICAL_MATCH", "") == "1"


def _paraphrase_serve_enabled() -> bool:
    return os.environ.get("SEMBLEND_PARAPHRASE_SERVE", "") == "1"


def _offsets_tokenizer(model_arch: str):
    """HF fast tokenizer for offset mapping (lazy; None on failure)."""
    try:
        from transformers import AutoTokenizer

        name = {
            "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
            "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
        }.get(model_arch, model_arch)
        return AutoTokenizer.from_pretrained(name)
    except Exception:
        logger.warning("[FUZZY] canonical match: tokenizer unavailable")
        return None


def _layer_mask_enabled(config) -> bool:
    """All copy by default; the mask is an explicit opt in via config or
    the SEMBLEND_LAYER_MASK=1 environment override."""
    if config.enable_bathtub:
        return True
    return os.environ.get("SEMBLEND_LAYER_MASK", "") == "1"


class SemBlendProviderAdapter:
    """Semantic fuzzy-match provider backed by SemBlendPipeline.

    Contract (matches the Chenxin draft ABC):
      - `register_donor`: called from `cache_on_request_finished` on the SGLang
        side. Inserts the completed request's embedding + tokens + KV handle
        into the in-process donor store.
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
    ) -> None:
        self._config = config
        self._pipeline = pipeline or self._build_pipeline(config)

        # Opaque KV handles keyed by donor request_id. Donor registration
        # inserts here; match lookups resolve donor_id -> handle to produce
        # pool-index tensors on return.
        #
        # OrderedDict-backed LRU: bounded by config.max_entries. On insert
        # over the bound we evict the least-recently-used entry. On match
        # we mark the hit donor as recently used. This keeps the adapter's
        # handle dict in lockstep with the pipeline's donor_store (both
        # use the same max_entries bound and LRU policy).
        self._donor_kv: OrderedDict[str, _DonorKVHandle] = OrderedDict()

        # Background executor for donor embedding + donor-store insertion.
        # register_donor stashes the KV handle synchronously (so
        # on_donor_inserted can find it for lock_ref) and offloads the
        # embed + store insert here so it doesn't block the scheduler's
        # request-finished callback. Single worker keeps writes serialized.
        self._register_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="semblend-register"
        )
        self._register_lock = threading.Lock()
        self._generation = 0

        self._offsets_tok = (
            _offsets_tokenizer(config.model_arch)
            if _canonical_match_enabled()
            else None
        )
        self._stats = _Stats()
        logger.info(
            "[FUZZY] semblend adapter build=headfix-2026-08-02 gates="
            "tail_reserve,position_aligned,sink,donor_sink,edge,min_tokens"
        )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_pipeline(config: SemBlendProviderConfig) -> Any:
        """Lazy-import SemBlendPipeline so tests can inject a mock."""
        from semblend_core.bathtub import RecomputeConfig
        from semblend_core.pipeline import SemBlendPipeline

        # Honor embedding_use_gpu when the caller left embedder_type at the
        # default "minilm". An explicit non-default embedder_type wins.
        embedder_type = config.embedder_type
        if embedder_type == "minilm" and config.embedding_use_gpu:
            embedder_type = "onnx-gpu"

        return SemBlendPipeline(
            max_donors=config.max_entries,
            min_similarity=config.min_similarity,
            # Canonical mode: the inner reuse gate would reject the reformat
            # class before the adapter's canonical rescue can run; the
            # adapter enforces the same floor (with canonical coverage as an
            # alternative pass), so behavior is preserved for all matches.
            min_reuse_ratio=(
                0.0 if _canonical_match_enabled() else config.min_reuse_ratio
            ),
            embedder_type=embedder_type,
            model_name=config.model_arch,
            chunk_size=config.block_size,
            recompute_config=RecomputeConfig.from_env(),
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
        # H8 probe gate: skip ALL registration work (incl. the async embed)
        # so negative-arm overhead can be attributed to match vs register.
        if os.environ.get("SEMBLEND_DISABLE_REGISTRATION"):
            self._stats.register_rejected += 1
            return False

        segment_tokens = list(token_ids[cache_start_pos:cache_end_pos])

        if len(segment_tokens) < self._config.min_match_length:
            self._stats.register_rejected += 1
            return False

        # Without prompt_text the embedder cannot produce a vector (we
        # never synthesize text from tokens here — the SGLang wrapper
        # owns the tokenizer). Reject synchronously so callers see the
        # same fast-fail behavior they did before async registration.
        if not prompt_text:
            self._stats.register_rejected += 1
            return False

        # Stash the KV handle synchronously so on_donor_inserted can find
        # it and so SGLang's scheduler sees a successful registration
        # immediately. The expensive embed + donor_store.add_donor are
        # offloaded to the background executor; donor becomes queryable
        # for future matches once the embedding lands.
        with self._register_lock:
            self._evict_lru_if_over_bound(reserve=1)
            self._donor_kv[request_id] = _DonorKVHandle(
                kv_indices=_snapshot_kv_indices(kv_cache),
                start_pos=cache_start_pos,
                end_pos=cache_end_pos,
                extra_key=extra_key,
                prompt_text=(prompt_text if _canonical_match_enabled() else None),
                token_ids=(list(token_ids) if _canonical_match_enabled() else None),
            )
            generation = self._generation
            try:
                self._register_executor.submit(
                    self._register_donor_async,
                    request_id,
                    segment_tokens,
                    prompt_text or "",
                    generation,
                    extra_key,
                )
            except Exception:
                self._donor_kv.pop(request_id, None)
                self._stats.register_rejected += 1
                logger.warning(
                    "[FUZZY] register_donor: failed to submit async registration",
                    exc_info=True,
                )
                return False

        self._stats.register_ok += 1
        logger.info(
            "[FUZZY] register_donor: queued request_id=%s tokens=%d donor_kv_size=%d",
            request_id,
            len(segment_tokens),
            len(self._donor_kv),
        )
        return True

    def _register_donor_async(
        self,
        request_id: str,
        segment_tokens: List[int],
        prompt_text: str,
        generation: int,
        extra_key: Optional[str],
    ) -> None:
        """Run the embed + donor-store insert off the scheduler thread.

        Logs per-stage timing so we can attribute long-context overhead to
        embed vs store-insert. Errors are swallowed (donor stays
        un-indexed) because this runs after the request has already
        returned to the client — raising would only kill the executor.
        """
        t_start = time.monotonic()
        try:
            t_embed_start = time.monotonic()
            embedding = self._embed(segment_tokens, prompt_text)
            t_embed_ms = (time.monotonic() - t_embed_start) * 1000
            if embedding is None:
                logger.info(
                    "[FUZZY] register_donor_async: embed returned None "
                    "request_id=%s tokens=%d embed=%.1fms",
                    request_id,
                    len(segment_tokens),
                    t_embed_ms,
                )
                # Drop the handle we stashed in the sync path; an un-embedded
                # donor cannot match.
                with self._register_lock:
                    if generation == self._generation:
                        self._donor_kv.pop(request_id, None)
                self._stats.register_rejected += 1
                return

            from semblend_core.donor_store import DonorNode

            node = DonorNode(
                request_id=request_id,
                token_ids=segment_tokens,
                embedding=embedding,
                timestamp=time.monotonic(),
                prompt_text=prompt_text,
                extra_key=extra_key,
            )
            t_store_start = time.monotonic()
            with self._register_lock:
                if generation != self._generation:
                    logger.info(
                        "[FUZZY] register_donor_async: stale generation "
                        "request_id=%s generation=%d current=%d",
                        request_id,
                        generation,
                        self._generation,
                    )
                    return
                self._pipeline._donor_store.add_donor(node)  # noqa: SLF001
            t_store_ms = (time.monotonic() - t_store_start) * 1000

            t_total_ms = (time.monotonic() - t_start) * 1000
            logger.info(
                "[FUZZY] register_donor_async: ok request_id=%s tokens=%d "
                "embed=%.1fms store=%.1fms total=%.1fms",
                request_id,
                len(segment_tokens),
                t_embed_ms,
                t_store_ms,
                t_total_ms,
            )
        except Exception as e:
            logger.warning(
                "[FUZZY] register_donor_async failed request_id=%s: %s",
                request_id,
                e,
                exc_info=True,
            )
            with self._register_lock:
                if generation == self._generation:
                    self._donor_kv.pop(request_id, None)
            self._stats.register_rejected += 1

    def _evict_lru_if_over_bound(self, reserve: int = 0) -> None:
        """Evict LRU donor handles until ``len(_donor_kv) + reserve <= max_entries``.

        Called on every register_donor with reserve=1 (we're about to add
        one). No-op when under the bound.
        """
        max_entries = max(int(self._config.max_entries), 1)
        while len(self._donor_kv) + reserve > max_entries and self._donor_kv:
            self._donor_kv.popitem(last=False)
            self._stats.donor_kv_evicted += 1

    def on_donor_inserted(
        self,
        request_id: str,
        donor_last_node_id: int,
    ) -> None:
        """Record the donor's TreeNode id (from radix_tree._node_registry).

        Called by SGLang's RadixCache.cache_finished_req AFTER the donor's
        KV has been inserted into the radix tree. The id is later surfaced
        via FuzzyMatchResult.donor_last_node_id at match time so the radix
        cache can inc_lock_ref the donor and prevent LRU eviction while the
        recipient request is consuming its KV.

        If `register_donor` rejected this donor (no embedding, too short,
        etc.) there's no `_DonorKVHandle` to update — just silently skip.
        """
        with self._register_lock:
            handle = self._donor_kv.get(request_id)
            if handle is None:
                return
            # Dataclass with default field — assign directly.
            handle.last_node_id = donor_last_node_id

    def clear(self) -> None:
        """Clear provider donor state after the owning cache is reset.

        SGLang's ``/flush_cache`` invalidates radix-tree nodes and KV slots.
        Any SemBlend donor handle that survives that reset can point at stale
        storage. This method clears both adapter-side KV handles and the
        pipeline donor store, and uses a generation counter so an async donor
        registration already in flight cannot reinsert a stale donor after the
        clear.
        """
        with self._register_lock:
            old_executor = self._register_executor
            self._generation += 1
            self._donor_kv.clear()
            self._clear_pipeline_donors()
            self._register_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="semblend-register"
            )

        try:
            old_executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            old_executor.shutdown(wait=False)
        self._stats.cache_resets += 1
        logger.info("[FUZZY] SemBlendProviderAdapter cleared donor state")

    def _clear_pipeline_donors(self) -> None:
        """Clear donor state while preserving expensive pipeline resources."""
        try:
            clear_donors = getattr(self._pipeline, "clear_donors", None)
            if callable(clear_donors):
                clear_donors()
                return

            donor_store = getattr(self._pipeline, "_donor_store", None)
            clear_store = getattr(donor_store, "clear", None)
            if callable(clear_store):
                clear_store()
                return

            # Compatibility for injected test doubles and older lightweight stubs.
            donors = getattr(donor_store, "donors", None)
            if hasattr(donors, "clear"):
                donors.clear()
                return

            raise TypeError("SemBlend pipeline does not expose a donor clear method")
        except Exception:
            logger.warning(
                "[FUZZY] SemBlend donor-state clear failed; rebuilding adapter pipeline",
                exc_info=True,
            )
            self._pipeline = self._build_pipeline(self._config)

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
            logger.debug(
                "[FUZZY] adapter.match: remaining=%d below min_match_length=%d, skip",
                len(remaining),
                self._config.min_match_length,
            )
            return None

        t_match_start = time.monotonic()
        try:
            with self._register_lock:
                result = self._pipeline.find_donor(
                    token_ids=remaining,
                    prompt_text=prompt_text or "",
                    top_k=self._config.top_k,
                    extra_key=extra_key,
                )
        except Exception as e:  # pragma: no cover
            logger.error("SemBlendPipeline.find_donor raised: %s", e, exc_info=True)
            self._stats.match_errors += 1
            return None

        if not getattr(result, "found", False):
            logger.debug(
                "[FUZZY] adapter.match: find_donor returned found=False (remaining=%d)",
                len(remaining),
            )
            self._stats.match_misses += 1
            return None

        # INFO-level visibility on every quality decision the adapter
        # makes — kept at INFO so production telemetry can monitor the
        # cosine / reuse_ratio distribution of attempted hits without
        # enabling DEBUG logging globally. Volume scales with number of
        # fuzzy match attempts (after exact prefix), which is bounded.
        logger.info(
            "[FUZZY] adapter.match: result donor_id=%s similarity=%.3f "
            "reuse_ratio=%.3f donor_kv_size=%d",
            getattr(result, "donor_id", None),
            float(getattr(result, "similarity", 0.0)),
            float(getattr(result, "reuse_ratio", 0.0)),
            len(self._donor_kv),
        )

        # Cosine gate applies to all paths. semblend_core surfaces a real
        # query-vs-primary-donor cosine even for multi-donor composite
        # results (pipeline._compute_aggregate_similarity), so this gate
        # is now meaningful on every path. We do NOT bypass the gate for
        # composite results because composite_plan presence is only a
        # construction marker, not a quality signal — relying on
        # reuse_ratio alone would let chunk-aligned matches with
        # high token-overlap but low semantic affinity slip through.
        if result.similarity < self._config.min_similarity:
            logger.info(
                "[FUZZY] adapter.match: similarity=%.3f < gate=%.3f, miss",
                float(result.similarity),
                float(self._config.min_similarity),
            )
            self._stats.match_misses += 1
            return None

        canonical_segments: List[FuzzyMatchSegment] = []
        if (
            _canonical_match_enabled()
            and not getattr(result, "segments", None)
            and result.reuse_ratio >= self._config.min_reuse_ratio
        ):
            pre_donor_id, handle_pre = self._resolve_canon_handle(result)
            if handle_pre is not None:
                canonical_segments = self._canonical_augment_segments(
                    remaining=list(remaining),
                    remaining_text=prompt_text or "",
                    donor_id=pre_donor_id,
                    handle=handle_pre,
                    donor_tokens=list(handle_pre.token_ids),
                )
        if result.reuse_ratio < self._config.min_reuse_ratio:
            # Token-level reuse starving under HIGH similarity is the
            # signature of the reformat class — canonical matching runs
            # BEFORE rejection (a canonical-covered window passes on its
            # own coverage).
            canon_donor_id, handle_for_canon = self._resolve_canon_handle(result)
            if handle_for_canon is not None:
                canonical_segments = self._canonical_augment_segments(
                    remaining=list(remaining),
                    remaining_text=prompt_text or "",
                    donor_id=canon_donor_id,
                    handle=handle_for_canon,
                    donor_tokens=list(handle_for_canon.token_ids),
                )
            canon_cover = sum(s_.length for s_ in canonical_segments) / max(
                len(remaining), 1
            )
            if canon_cover < self._config.min_reuse_ratio:
                # Verified paraphrase serve: high semantic similarity with
                # low token/canonical coverage is the paraphrase signature.
                # A fail-closed lexical fact gate over the retained donor
                # text decides; accepted spans serve the donor KV directly
                # (measured at serving scale: fact-preserving substitution
                # sits at the exact-cache quality floor).
                if (
                    _paraphrase_serve_enabled()
                    and prompt_text
                    and handle_for_canon is not None
                    and handle_for_canon.prompt_text
                    and result.similarity >= self._config.paraphrase_min_similarity
                ):
                    from semblend_core.fact_gate import spans_fact_consistent

                    if spans_fact_consistent(
                        handle_for_canon.prompt_text, prompt_text
                    ):
                        logger.info(
                            "[FUZZY] adapter.match: paraphrase serve "
                            "(similarity=%.3f, fact gate passed)",
                            float(result.similarity),
                        )
                        return self._paraphrase_result(
                            donor_id=canon_donor_id,
                            handle=handle_for_canon,
                            remaining=remaining,
                            similarity=float(result.similarity),
                        )
                    logger.info(
                        "[FUZZY] adapter.match: paraphrase candidate REJECTED "
                        "by fact gate (similarity=%.3f)",
                        float(result.similarity),
                    )
                logger.info(
                    "[FUZZY] adapter.match: reuse_ratio=%.3f < gate=%.3f "
                    "(canonical cover=%.3f), reject",
                    float(result.reuse_ratio),
                    float(self._config.min_reuse_ratio),
                    canon_cover,
                )
                self._stats.match_rejected_low_reuse += 1
                return None
            logger.info(
                "[FUZZY] adapter.match: canonical cover=%.3f rescues "
                "low token reuse=%.3f",
                canon_cover,
                float(result.reuse_ratio),
            )

        t_pipeline_ms = (time.monotonic() - t_match_start) * 1000
        t_convert_start = time.monotonic()
        converted = self._convert_result(
            pipeline_result=result,
            already_matched_len=already_matched_len,
            remaining=remaining,
            extra_key=extra_key,
            prompt_text=prompt_text,
            canonical_segments=canonical_segments,
        )
        t_convert_ms = (time.monotonic() - t_convert_start) * 1000
        logger.info(
            "[FUZZY] match timing: pipeline=%.0fms convert=%.0fms remaining=%d",
            t_pipeline_ms,
            t_convert_ms,
            len(remaining),
        )
        if converted is None:
            logger.debug(
                "[FUZZY] adapter.match: _convert_result returned None for "
                "donor_id=%s (donor_kv has %d donors, handle_present=%s)",
                getattr(result, "donor_id", None),
                len(self._donor_kv),
                getattr(result, "donor_id", None) in self._donor_kv,
            )
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
        extra_key: Optional[str],
        prompt_text: Optional[str] = None,
        canonical_segments: Optional[List[FuzzyMatchSegment]] = None,
    ) -> Optional[FuzzyMatchResult]:
        """Translate `PipelineResult` → `FuzzyMatchResult`.

        Builds segments from `position_map` if N:M alignment is required;
        otherwise returns a contiguous-span result compatible with the
        non-semantic `TokenBlockMatchProvider` path.
        """
        donor_id = pipeline_result.donor_id
        if donor_id is None:
            logger.debug("[FUZZY] _convert_result: donor_id is None, drop")
            return None
        with self._register_lock:
            handle = self._donor_kv.get(donor_id)
            if handle is None:
                logger.debug(
                    "[FUZZY] _convert_result: no handle for donor_id=%s "
                    "(donor_kv size=%d) — pipeline kept this donor but adapter "
                    "had already evicted it",
                    donor_id,
                    len(self._donor_kv),
                )
                return None
            if handle.extra_key != extra_key:
                logger.info(
                    "[FUZZY] _convert_result: rejecting donor_id=%s extra_key=%r "
                    "for query extra_key=%r",
                    donor_id,
                    handle.extra_key,
                    extra_key,
                )
                return None
            # Mark the donor as recently used for LRU. Move to end of the
            # OrderedDict so the next eviction wave skips this entry.
            self._donor_kv.move_to_end(donor_id)

        # Build layer_recompute_mask from pipeline_result.layer_deviations.
        layer_mask = None
        if _layer_mask_enabled(self._config) and pipeline_result.layer_deviations:
            layer_mask = [
                bool(d.get("shouldRecompute", False)) for d in pipeline_result.layer_deviations
            ]

        segments = self._build_segments(pipeline_result, handle, layer_mask)
        all_segments = list(segments)

        # Canonical runs are token-VERIFIED; chunk-fuzzy segments are
        # not (embedding-matched, token-different) — mixing them double-
        # covers the window and re-opens scatter-quality risk. Canonical
        # REPLACES the pipeline's segment set when present.
        if canonical_segments:
            all_segments = list(canonical_segments)
        else:
            aligned_mass = sum(
                len(_as_list(s_.target_positions)) for s_ in all_segments
            )
            if aligned_mass < max(1, len(remaining) // 4):
                all_segments.extend(
                    self._canonical_augment_segments(
                        remaining=list(remaining),
                        remaining_text=prompt_text or "",
                        donor_id=donor_id,
                        handle=handle,
                        donor_tokens=list(pipeline_result.donor_tokens),
                    )
                )

        # SGLang's prefix-cache device_indices contract treats the cached
        # span as a contiguous prefix [0..exact+N-1]. To surface a semantic
        # match whose target_positions sit at a non-prefix offset, we pick
        # the longest contiguous run from the aligner output and re-anchor
        # it at already_matched_len. The donor positions are carried
        # through as cached_start_pos so radix_cache's RoPE-correction
        # realization path (`needs_realization = cached_start_pos !=
        # exact_matched_len`) fires: realized_locs get pre-allocated and
        # donor KV is copied into fresh slots with positional encoding
        # shifted from donor positions to recipient positions.
        #
        # This is a deliberate trade-off vs. the rolled-back non-prefix-
        # anchored path. The donor's K/V values were computed in the
        # donor's preceding-token context. When we re-anchor them to the
        # recipient's prefix, the recipient's actual tokens at those
        # positions may differ from the donor's matched tokens, so
        # attention sees the donor's K/V instead of the recipient's true
        # K/V at those positions. The bathtub layer_recompute_mask is
        # designed to recompute layers where the deviation is largest;
        # see SEMANTIC_FUZZY_MATCH.md for the quality bound.
        # Chunk-fuzzy matches can report high reuse while yielding zero
        # consumable (token-identical) segments; canonical runs are the
        # consumable form of the same content. Inject them whenever the
        # pipeline produced no segments.
        if canonical_segments:
            logger.info(
                "[FUZZY] canonical: %d verified runs replace %d pipeline segments",
                len(canonical_segments),
                len(segments or []),
            )
            segments = list(canonical_segments)

        if segments:
            best = max(
                segments,
                key=lambda s: len(_as_list(s.target_positions)),
            )
            best_targets = _as_list(best.target_positions)
            if not best_targets:
                self._stats.match_hits_dropped_no_prefix_run += 1
                return None
            segments = [best]
        else:
            best = None

        if segments:
            total_covered = len(_as_list(segments[0].target_positions))
        else:
            total_covered = 0

        quality = QualitySignals(
            cosine_similarity=float(pipeline_result.similarity),
            reuse_ratio=float(pipeline_result.reuse_ratio),
            confidence_tier=str(getattr(pipeline_result, "confidence_tier", "exact")),
            passed_quality_gate=True,
        )

        # Collapse to the legacy contiguous path. cached_start_pos carries
        # the donor's source position so model_runner's
        # _correct_fuzzy_kv_rope_contiguous can RoPE-reverse from there
        # and re-apply at the recipient's already_matched_len boundary.
        sparse_plan = None
        if segments:
            head = segments[0]
            kv_cache_indices = head.donor_kv_indices
            cached_start_pos = int(_first(head.donor_positions, 0))
            out_segments = None
            if self._multi_segment_enabled():
                gated = self._gate_segments(
                    all_segments,
                    list(pipeline_result.donor_tokens),
                    list(remaining),
                )
                if not gated:
                    # Everything gated out: emitting the UNGATED head would
                    # bypass every safety gate (position-aligned self-matches
                    # and full-coverage admissions wedged the engine —
                    # measured live). No safe mass -> miss.
                    self._stats.match_hits_dropped_no_prefix_run += 1
                    return None
                # Contract Option A with the head derived from the GATED
                # plan: cached_token_count / kv_cache_indices keep v1
                # head-run semantics, but only gate-surviving mass is ever
                # emitted on either path.
                head = gated[0]
                kv_cache_indices = head.donor_kv_indices
                cached_start_pos = int(_first(head.donor_positions, 0))
                total_covered = len(_as_list(head.target_positions))
                head_target_start = int(_first(head.target_positions, -1))
                if sparse_plan_enabled():
                    sparse_plan = build_sparse_plan(
                        gated,
                        remaining_len=len(remaining),
                        min_donor_span=int(
                            os.environ.get("SEMBLEND_SPARSE_MIN_SPAN", "512")
                        ),
                        edge_shave=int(
                            os.environ.get("SEMBLEND_SPARSE_EDGE_SHAVE", "0")
                        ),
                        gap_period=int(
                            os.environ.get("SEMBLEND_SPARSE_GAP_PERIOD", "0")
                        ),
                        gap_size=int(
                            os.environ.get("SEMBLEND_SPARSE_GAP_SIZE", "64")
                        ),
                    )
                if len(gated) == 1 and head_target_start == 0:
                    # Positions are tail-relative: start 0 == anchored at the
                    # prefix boundary.
                    # A single run anchored AT the prefix boundary fits the
                    # v1 contiguous contract exactly: prefix-integrated
                    # consumption (RoPE-delta-corrected) — the near-lossless
                    # mechanism (KL ~0.004 on shifted-offset content) that
                    # also actually skips prefill. Scatter realization of
                    # the same content class measured 0.19-0.40 ROUGE
                    # (measured live) — never scatter what can be contiguous.
                    out_segments = None
                else:
                    out_segments = self._merge_segments(gated)
                segment_mass = sum(len(_as_list(s.target_positions)) for s in gated)
                logger.info(
                    "[FUZZY] emission: head=%d mass=%d prompt_remaining=%d "
                    "dropped(pa=%d tail=%d sink=%d)",
                    total_covered,
                    segment_mass,
                    len(remaining),
                    self._stats.segments_dropped_position_aligned,
                    self._stats.segments_dropped_tail_reserve,
                    self._stats.segments_dropped_sink,
                )
                logger.info(
                    "[FUZZY] segment funnel (merged=%d): built=%d gated_out=%d emitted=%d "
                    "head_run=%d segment_mass=%d dropped_short=%d dropped_identity=%d "
                    "dropped_sink=%d dropped_donor_sink=%d dropped_edge=%d "
                    "bathtub_mask=%s",
                    len(out_segments) if out_segments else 0,
                    len(all_segments),
                    len(all_segments) - len(gated),
                    len(gated),
                    total_covered,
                    segment_mass,
                    self._stats.segments_dropped_short,
                    self._stats.segments_dropped_low_identity,
                    self._stats.segments_dropped_sink,
                    self._stats.segments_dropped_donor_sink,
                    self._stats.segments_dropped_edge_trim,
                    ("%d-layers" % sum(layer_mask)) if layer_mask else "none",
                )
                # All segments gated out -> contiguous-head fallback (v1
                # behavior) already set above.
        else:
            if handle.start_pos != already_matched_len:
                self._stats.match_hits_dropped_no_prefix_run += 1
                return None
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
            sparse_plan=sparse_plan,
            layer_recompute_mask=layer_mask,
            quality_signals=quality,
            donor_last_node_id=handle.last_node_id,
        )

    def _paraphrase_result(
        self,
        donor_id: str,
        handle: "_DonorKVHandle",
        remaining: List[int],
        similarity: float,
    ) -> Optional[FuzzyMatchResult]:
        """Contiguous result serving a fact-verified paraphrase span.

        The donor KV is served directly (no token identity claim); the
        engine's realization path applies positional correction. Length
        caps at the shorter of donor coverage and the remaining window,
        minus the tail reserve so the final tokens always compute.
        """
        donor_len = handle.end_pos - handle.start_pos
        # _tail_reserve_tokens returns the last servable BOUNDARY position
        # (prompt_len minus the reserved tail), or 0 when the prompt is too
        # short for a reserve to apply.
        serve_cap = self._tail_reserve_tokens(len(remaining))
        if serve_cap <= 0:
            serve_cap = max(0, len(remaining) - 1)
        serve_len = min(donor_len, serve_cap)
        if serve_len < self._config.min_match_length:
            return None
        kv = _slice_indices(handle.kv_indices, offset=0, length=serve_len)
        return FuzzyMatchResult(
            cached_token_count=serve_len,
            cached_token_ids=list(remaining[:serve_len]),
            prompt_token_count=len(remaining),
            kv_cache_indices=kv,
            position_offset=0,
            cached_start_pos=handle.start_pos,
            donor_last_node_id=handle.last_node_id,
            quality_signals=QualitySignals(
                cosine_similarity=similarity,
                reuse_ratio=serve_len / max(1, len(remaining)),
                confidence_tier="paraphrase_verified",
                passed_quality_gate=True,
                rejection_reason=None,
            ),
        )

    def _resolve_canon_handle(self, result):
        """Resolve the donor handle for canonical alignment.

        Composite (multi-donor) results can carry a donor_id that is not a
        registry key; falling back to the composite's donor ids, then to
        the sole registered donor, keeps the rescue from silently
        skipping. Returns (donor_id, handle_or_None); a usable handle
        always carries token_ids.
        """
        donor_id = getattr(result, "donor_id", None)
        handle = self._donor_kv.get(donor_id)
        if handle is not None and handle.token_ids is not None:
            return donor_id, handle
        for cand in list(getattr(result, "donor_ids", None) or []):
            h = self._donor_kv.get(cand)
            if h is not None and h.token_ids is not None:
                logger.info(
                    "[FUZZY] canonical: donor_id=%s not registered; using "
                    "composite donor %s",
                    donor_id,
                    cand,
                )
                return cand, h
        if len(self._donor_kv) == 1:
            sole_id, h = next(iter(self._donor_kv.items()))
            if h.token_ids is not None:
                logger.info(
                    "[FUZZY] canonical: donor_id=%s not registered; falling "
                    "back to sole registered donor %s",
                    donor_id,
                    sole_id,
                )
                return sole_id, h
        logger.info(
            "[FUZZY] canonical: donor_id=%s unresolvable (registered=%d); "
            "rescue skipped",
            donor_id,
            len(self._donor_kv),
        )
        return donor_id, None

    def _canonical_augment_segments(
        self,
        remaining: List[int],
        remaining_text: str,
        donor_id: str,
        handle: "_DonorKVHandle",
        donor_tokens: List[int],
    ) -> List[FuzzyMatchSegment]:
        """Tokenization-invariant runs vs a donor, as segments.

        Runs are computed in canonical text space and token-verified, so the
        emitted segments carry exactly the token-identical guarantee the
        join machinery expects; positions are remaining-window token
        indices (target) and donor-sequence token indices (donor).
        """
        if not _canonical_match_enabled():
            logger.info("[FUZZY] canonical: disabled (env)")
            return []
        if not remaining_text or not handle.prompt_text:
            logger.info(
                "[FUZZY] canonical: missing text (remaining=%s donor=%s)",
                bool(remaining_text),
                bool(handle.prompt_text),
            )
            return []
        if self._offsets_tok is None:
            logger.info("[FUZZY] canonical: no offsets tokenizer")
            return []
        try:
            from semblend_core.canonical_alignment import resynced_token_runs

            t_enc = self._offsets_tok(
                remaining_text, add_special_tokens=False, return_offsets_mapping=True
            )
            d_enc = self._offsets_tok(
                handle.prompt_text, add_special_tokens=False, return_offsets_mapping=True
            )
            runs = resynced_token_runs(
                donor_text=handle.prompt_text,
                target_text=remaining_text,
                donor_ids_offsets=(d_enc["input_ids"], d_enc["offset_mapping"]),
                target_ids_offsets=(t_enc["input_ids"], t_enc["offset_mapping"]),
                min_run_tokens=self._config.segment_min_tokens,
            )
            # Boundary telemetry: one line per attempt so a zero-run
            # outcome is never silent (each earlier failure of this class
            # was invisible until reproduced offline).
            logger.info(
                "[FUZZY] canonical align: target_text=%d donor_text=%d "
                "target_toks=%d donor_toks=%d engine_window=%d min_run=%d "
                "-> runs=%d",
                len(remaining_text),
                len(handle.prompt_text),
                len(t_enc["input_ids"]),
                len(d_enc["input_ids"]),
                len(remaining),
                self._config.segment_min_tokens,
                len(runs),
            )
            if not runs and t_enc["input_ids"]:
                probe = min(32, len(t_enc["input_ids"]), len(d_enc["input_ids"]))
                logger.info(
                    "[FUZZY] canonical align: zero runs; head sample "
                    "target=%s donor=%s",
                    t_enc["input_ids"][:probe],
                    d_enc["input_ids"][:probe],
                )
        except Exception:
            logger.exception("[FUZZY] canonical augmentation failed")
            return []
        segments: List[FuzzyMatchSegment] = []
        dropped_by_guard = 0
        for r in runs:
            # ``remaining_text`` is the engine-decoded text of the SAME
            # window ``remaining`` covers, so run indices are already
            # window-relative — no rebase.
            tw, d0, ln = r["target_token_start"], r["donor_token_start"], r["length"]
            # Guard: run token ids must match the ENGINE tokenizations too
            # (offsets tokenizer may disagree with serving tokenizer).
            if tw + ln > len(remaining) or d0 + ln > len(donor_tokens):
                continue
            if remaining[tw : tw + ln] != donor_tokens[d0 : d0 + ln]:
                dropped_by_guard += 1
                continue
            kv_slice = _slice_indices(handle.kv_indices, offset=d0, length=ln)
            segments.append(
                FuzzyMatchSegment(
                    target_positions=list(range(tw, tw + ln)),
                    donor_positions=list(range(d0, d0 + ln)),
                    donor_node_id=handle.last_node_id,
                    donor_offset=d0,
                    length=ln,
                    donor_kv_indices=kv_slice,
                    donor_req_id=donor_id,
                )
            )
        if runs and not segments:
            logger.info(
                "[FUZZY] canonical: %d aligned runs but none survived "
                "(engine-id guard drops=%d)",
                len(runs),
                dropped_by_guard,
            )
        if segments:
            logger.info(
                "[FUZZY] canonical augmentation: %d runs, %d tokens",
                len(segments),
                sum(s_.length for s_ in segments),
            )
        return segments

    def _segment_token_identity(
        self,
        segment: FuzzyMatchSegment,
        donor_tokens: List[int],
        target_tokens: List[int],
    ) -> float:
        """Fraction of aligned positions whose donor and target token ids match.

        Reusing donor KV where the underlying tokens differ is the direct
        damage mechanism; near-duplicates that differ only in entities sit
        just below 1.0 and are exactly what the gate must keep out.
        """
        donor_positions = _as_list(segment.donor_positions)
        target_positions = _as_list(segment.target_positions)
        if not donor_positions or len(donor_positions) != len(target_positions):
            return 0.0
        same = 0
        total = 0
        for d_pos, t_pos in zip(donor_positions, target_positions):
            if 0 <= d_pos < len(donor_tokens) and 0 <= t_pos < len(target_tokens):
                total += 1
                if donor_tokens[d_pos] == target_tokens[t_pos]:
                    same += 1
        return (same / total) if total else 0.0

    def _segment_min_tokens(self) -> int:
        override = os.environ.get("SEMBLEND_SEGMENT_MIN_TOKENS")
        if override:
            try:
                return int(override)
            except ValueError:
                pass
        return self._config.segment_min_tokens

    def _tail_reserve_tokens(self, prompt_len: int) -> int:
        """Never realize into the last N target positions. Near-full-coverage
        admissions wedge the engine (request never scheduled: up-front
        realized-mass alloc + extend + locked donor break the fit math —
        py-spy-confirmed live, idle scheduler). Reserving a tail also
        guarantees a nonzero extend. Default 64; env
        SEMBLEND_TAIL_RESERVE_TOKENS."""
        raw = os.environ.get("SEMBLEND_TAIL_RESERVE_TOKENS")
        reserve = 64
        if raw:
            try:
                reserve = int(raw)
            except ValueError:
                pass
        if reserve <= 0 or prompt_len <= 4 * reserve:
            return 0
        return prompt_len - reserve

    def _sink_protect_tokens(self) -> int:
        override = os.environ.get("SEMBLEND_SINK_PROTECT_TOKENS")
        if override:
            try:
                return int(override)
            except ValueError:
                pass
        return self._config.sink_protect_tokens

    def _donor_sink_protect_tokens(self) -> int:
        override = os.environ.get("SEMBLEND_DONOR_SINK_PROTECT_TOKENS")
        if override:
            try:
                return int(override)
            except ValueError:
                pass
        return self._config.donor_sink_protect_tokens

    def _segment_edge_trim_tokens(self) -> int:
        override = os.environ.get("SEMBLEND_SEGMENT_EDGE_TRIM")
        if override:
            try:
                return int(override)
            except ValueError:
                pass
        return self._config.segment_edge_trim_tokens

    def _donor_run_head_trim_tokens(self) -> int:
        override = os.environ.get("SEMBLEND_DONOR_RUN_HEAD_TRIM")
        if override:
            try:
                return int(override)
            except ValueError:
                pass
        return self._config.donor_run_head_trim_tokens

    @staticmethod
    def _trim_tail_beyond(
        segment: FuzzyMatchSegment, max_target_pos: int
    ) -> Optional[FuzzyMatchSegment]:
        """Drop positions with target position >= max_target_pos (ascending
        runs): the reserved prompt tail must stay compute-fresh."""
        positions = _as_list(segment.target_positions)
        if not positions or positions[-1] < max_target_pos:
            return segment
        keep = 0
        for pos in positions:
            if pos >= max_target_pos:
                break
            keep += 1
        if keep == 0:
            return None
        return FuzzyMatchSegment(
            target_positions=positions[:keep],
            donor_positions=_as_list(segment.donor_positions)[:keep],
            donor_node_id=segment.donor_node_id,
            donor_offset=segment.donor_offset,
            length=keep if segment.length is not None else None,
            donor_kv_indices=(
                segment.donor_kv_indices[:keep]
                if segment.donor_kv_indices is not None
                else None
            ),
            donor_req_id=segment.donor_req_id,
            layer_recompute_mask=segment.layer_recompute_mask,
        )

    @staticmethod
    def _trim_run_edges(
        segment: FuzzyMatchSegment, head: int, tail: int
    ) -> Optional[FuzzyMatchSegment]:
        """Drop ``head`` positions from the run's start and ``tail`` from its
        end; trimmed positions are true-recomputed by the forward. Returns
        None when nothing remains. NodeRef addressing rebased."""
        positions = _as_list(segment.target_positions)
        n = len(positions)
        if head + tail <= 0:
            return segment
        if n <= head + tail:
            return None
        end = n - tail
        return FuzzyMatchSegment(
            target_positions=positions[head:end],
            donor_positions=_as_list(segment.donor_positions)[head:end],
            donor_node_id=segment.donor_node_id,
            donor_offset=(
                segment.donor_offset + head
                if segment.donor_offset is not None
                else None
            ),
            length=(end - head) if segment.length is not None else None,
            donor_kv_indices=(
                segment.donor_kv_indices[head:end]
                if segment.donor_kv_indices is not None
                else None
            ),
            donor_req_id=segment.donor_req_id,
            layer_recompute_mask=segment.layer_recompute_mask,
        )

    @staticmethod
    def _trim_sink(
        segment: FuzzyMatchSegment,
        protect: int,
        *,
        key: str = "target",
    ) -> Optional[FuzzyMatchSegment]:
        """Drop the run's positions whose ``key`` position is below ``protect``.

        key="target": never realize INTO the recipient's attention sink.
        key="donor": never realize KV COMPUTED AT the donor's sink — chunks
        computed separately each carry their own sink structure (phantom-sink
        import, arXiv 2603.20218); donor-sink K/V is the most
        context-contaminated mass a donor has. Ascending runs assumed.

        Returns None when the whole run is protected. NodeRef addressing is
        rebased (donor_offset/length) alongside the parallel position arrays
        so both consume paths stay coherent."""
        keyed = _as_list(
            segment.donor_positions if key == "donor" else segment.target_positions
        )
        positions = _as_list(segment.target_positions)
        if not keyed or keyed[0] >= protect:
            return segment
        trim = 0
        for pos in keyed:
            if pos >= protect:
                break
            trim += 1
        if trim >= len(positions):
            return None
        return FuzzyMatchSegment(
            target_positions=positions[trim:],
            donor_positions=_as_list(segment.donor_positions)[trim:],
            donor_node_id=segment.donor_node_id,
            donor_offset=(
                segment.donor_offset + trim
                if segment.donor_offset is not None
                else None
            ),
            length=(len(positions) - trim) if segment.length is not None else None,
            donor_kv_indices=(
                segment.donor_kv_indices[trim:]
                if segment.donor_kv_indices is not None
                else None
            ),
            donor_req_id=segment.donor_req_id,
            layer_recompute_mask=segment.layer_recompute_mask,
        )

    def _gate_segments(
        self,
        segments: List[FuzzyMatchSegment],
        donor_tokens: List[int],
        target_tokens: List[int],
    ) -> List[FuzzyMatchSegment]:
        min_tokens = self._segment_min_tokens()
        protect = self._sink_protect_tokens()
        tail_reserve = self._tail_reserve_tokens(len(target_tokens))
        donor_protect = self._donor_sink_protect_tokens()
        edge_trim = self._segment_edge_trim_tokens()
        head_trim = self._donor_run_head_trim_tokens()
        kept: List[FuzzyMatchSegment] = []
        for segment in segments:
            if protect > 0:
                trimmed = self._trim_sink(segment, protect)
                if trimmed is None:
                    self._stats.segments_dropped_sink += 1
                    continue
                segment = trimmed
            if donor_protect > 0:
                trimmed = self._trim_sink(segment, donor_protect, key="donor")
                if trimmed is None:
                    self._stats.segments_dropped_donor_sink += 1
                    continue
                segment = trimmed
            seg_tp = _as_list(segment.target_positions)
            seg_dp = _as_list(segment.donor_positions)
            if seg_tp and seg_dp and seg_tp[0] == seg_dp[0]:
                # Position-aligned content is the exact radix cache's job;
                # realizing it re-copies KV onto its own positions (donor-arm
                # self-matches) and tripped the admission wedge (observed live).
                self._stats.segments_dropped_position_aligned += 1
                continue
            if tail_reserve > 0:
                trimmed = self._trim_tail_beyond(segment, tail_reserve)
                if trimmed is None:
                    self._stats.segments_dropped_tail_reserve += 1
                    continue
                segment = trimmed
            if head_trim > 0 or edge_trim > 0:
                trimmed = self._trim_run_edges(
                    segment, max(edge_trim, head_trim), edge_trim
                )
                if trimmed is None:
                    self._stats.segments_dropped_edge_trim += 1
                    continue
                segment = trimmed
            length = len(_as_list(segment.target_positions))
            if length < min_tokens:
                self._stats.segments_dropped_short += 1
                continue
            identity = self._segment_token_identity(segment, donor_tokens, target_tokens)
            if identity < self._config.segment_min_token_identity:
                self._stats.segments_dropped_low_identity += 1
                continue
            kept.append(segment)
        return kept

    def _merge_segments(
        self, segments: List[FuzzyMatchSegment]
    ) -> List[FuzzyMatchSegment]:
        """Concatenate gated runs into few large scatter segments.

        Position/index arrays are explicit, so merged segments need no
        contiguity; the realizer performs one RoPE-corrected scatter copy
        per segment either way. Runs are only merged when they share the
        donor and addressing mode."""
        limit = self._config.segment_merge_max_positions
        if limit <= 0 or len(segments) <= 1:
            return segments
        merged: List[FuzzyMatchSegment] = []
        bucket: List[FuzzyMatchSegment] = []
        bucket_size = 0

        def flush() -> None:
            nonlocal bucket, bucket_size
            if not bucket:
                return
            if len(bucket) == 1:
                merged.append(bucket[0])
            else:
                first = bucket[0]
                merged.append(
                    FuzzyMatchSegment(
                        target_positions=[
                            p for s in bucket for p in _as_list(s.target_positions)
                        ],
                        donor_positions=[
                            p for s in bucket for p in _as_list(s.donor_positions)
                        ],
                        donor_node_id=first.donor_node_id,
                        donor_offset=first.donor_offset,
                        length=bucket_size,
                        donor_kv_indices=(
                            [i for s in bucket for i in _as_list(s.donor_kv_indices)]
                            if all(s.donor_kv_indices is not None for s in bucket)
                            else None
                        ),
                        donor_req_id=first.donor_req_id,
                        layer_recompute_mask=first.layer_recompute_mask,
                    )
                )
            bucket = []
            bucket_size = 0

        for segment in segments:
            seg_len = len(_as_list(segment.target_positions))
            same_group = (
                not bucket
                or (
                    segment.donor_req_id == bucket[0].donor_req_id
                    and segment.donor_node_id == bucket[0].donor_node_id
                    and (segment.donor_kv_indices is None)
                    == (bucket[0].donor_kv_indices is None)
                )
            )
            if bucket and (not same_group or bucket_size + seg_len > limit):
                flush()
            bucket.append(segment)
            bucket_size += seg_len
        flush()
        return merged

    def _multi_segment_enabled(self) -> bool:
        if self._config.multi_segment_emission:
            return True
        return os.environ.get("SEMBLEND_RETURN_SEGMENTS", "") == "1"

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
            seg_offset = seg_donor_positions[0] - handle.start_pos
            seg_length = end - start

            # Populate BOTH addressing modes:
            #   * NodeRef (preferred): donor_node_id + donor_offset + length.
            #     Resolved at consume time via radix_tree._node_registry,
            #     paired with donor_last_node_id inc_lock_ref protection.
            #   * Legacy: donor_kv_indices, a raw pool-indices slice. Kept
            #     so callers that haven't migrated to NodeRef resolution
            #     still work.
            kv_slice = _slice_indices(
                handle.kv_indices,
                offset=seg_offset,
                length=seg_length,
            )
            segments.append(
                FuzzyMatchSegment(
                    target_positions=seg_target_positions,
                    donor_positions=seg_donor_positions,
                    donor_node_id=handle.last_node_id,
                    donor_offset=seg_offset,
                    length=seg_length,
                    donor_kv_indices=kv_slice,
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
    donor_kv_evicted: int = 0
    cache_resets: int = 0
    # Match was found by the alignment pipeline but its head segment did not
    # anchor at the recipient's exact-prefix length, so SGLang's prefix-cache
    # cannot express it as device_indices. The match is dropped rather than
    # emitted as a no-op.
    match_hits_dropped_no_prefix_run: int = 0
    # Multi-segment emission gates (v0.4 line)
    segments_dropped_short: int = 0
    segments_dropped_low_identity: int = 0
    segments_dropped_sink: int = 0
    segments_dropped_donor_sink: int = 0
    segments_dropped_edge_trim: int = 0
    segments_dropped_tail_reserve: int = 0
    segments_dropped_position_aligned: int = 0

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
            "match_hits_dropped_no_prefix_run": self.match_hits_dropped_no_prefix_run,
            "donor_kv_evicted": self.donor_kv_evicted,
            "cache_resets": self.cache_resets,
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


def _snapshot_kv_indices(kv_cache: Any) -> Any:
    """Snapshot an engine-owned KV-index handle at the adapter boundary.

    SGLang passes a view into its request-to-token table. That table is reused
    after request completion, so holding the view directly can make a donor
    handle silently point at a later request's slots. This helper is deliberately
    duck typed so SemBlend does not import torch at module import time.
    """
    if kv_cache is None:
        return None

    obj = kv_cache
    detach = getattr(obj, "detach", None)
    if callable(detach):
        obj = detach()

    clone = getattr(obj, "clone", None)
    if callable(clone):
        try:
            return clone()
        except Exception:
            pass

    copy = getattr(obj, "copy", None)
    if callable(copy):
        try:
            return copy()
        except Exception:
            pass

    if isinstance(obj, (list, tuple)):
        return list(obj)

    return obj

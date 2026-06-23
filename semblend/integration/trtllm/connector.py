"""TensorRT-LLM KV connector for SemBlend semantic KV reuse."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from semblend.integration.trtllm.engine_patch import install_engine_patch
from semblend.integration.trtllm.events import TrtllmContractEmitter
from semblend.integration.trtllm.namespace import build_cache_namespace
from semblend.integration.trtllm.runtime_state import (
    clear_active_plan,
    set_active_plan,
)
from semblend.integration.trtllm.semblend_provider import SemBlendTensorRTProvider
from semblend.integration.trtllm.upstream_interface import (
    DonorLocation,
    SemanticKvEvent,
    SemanticKvLookupRequest,
    SemanticKvLookupResult,
    SemanticKvPlan,
)

logger = logging.getLogger("semblend.trtllm.connector")

try:  # pragma: no cover - exercised in a TensorRT-LLM runtime.
    from tensorrt_llm._torch.pyexecutor.kv_cache_connector import (
        KvCacheConnectorScheduler,
        KvCacheConnectorWorker,
        SchedulerOutput,
    )
except Exception:  # pragma: no cover - default in unit tests without TensorRT-LLM.
    try:
        from tensorrt_llm._torch.pyexecutor.connectors.kv_cache_connector import (
            KvCacheConnectorScheduler,
            KvCacheConnectorWorker,
            SchedulerOutput,
        )
    except Exception:

        class KvCacheConnectorScheduler:
            def __init__(self, llm_args: Any = None):
                self._llm_args = llm_args

        class KvCacheConnectorWorker:
            def __init__(self, llm_args: Any = None):
                self._llm_args = llm_args
                self._metadata = None

            def bind_connector_meta(self, metadata: object):
                self._metadata = metadata

            def get_connector_meta(self) -> object:
                return self._metadata

        class SchedulerOutput:
            new_requests: list[Any]
            cached_requests: list[Any]


@dataclass
class SemBlendConnectorMetadata:
    plans: dict[int, SemanticKvPlan] = field(default_factory=dict)
    events: list[SemanticKvEvent] = field(default_factory=list)


class SemBlendKvConnectorScheduler(KvCacheConnectorScheduler):
    """Leader-side scheduler for SemBlend TensorRT-LLM integration."""

    def __init__(self, llm_args: Any):
        super().__init__(llm_args)
        patched_models = install_engine_patch()
        namespace = build_cache_namespace(llm_args=llm_args)
        self._namespace = namespace
        self._provider = SemBlendTensorRTProvider(
            model_name=namespace.model,
            min_similarity=float(os.environ.get("SEMBLEND_MIN_SIMILARITY", "0.60")),
            min_reuse_ratio=float(os.environ.get("SEMBLEND_MIN_REUSE_RATIO", "0.50")),
            max_donors=int(os.environ.get("SEMBLEND_MAX_DONORS", "10000")),
            embedder_type=os.environ.get("SEMBLEND_EMBEDDER", "minilm"),
            chunk_size=namespace.block_size,
            allow_segmented=os.environ.get("SEMBLEND_TRTLLM_ENABLE_SEGMENTED", "0") == "1",
            min_match_length=int(os.environ.get("SEMBLEND_TRTLLM_MIN_MATCH_LENGTH", "128")),
        )
        self._pending_plans: dict[int, SemanticKvPlan] = {}
        self._events: list[SemanticKvEvent] = []
        self._event_id = 0
        self._contract_emitter = TrtllmContractEmitter.from_env(
            namespace=namespace,
            embedder_type=os.environ.get("SEMBLEND_EMBEDDER", "minilm"),
        )
        if self._contract_emitter is not None:
            self._contract_emitter.generation_reset()
        _write_audit(
            {
                "event": "scheduler_initialized",
                "namespace": namespace.to_dict(),
                "patched_engine_models": patched_models,
            }
        )

    def get_num_new_matched_tokens(self, request: Any, num_computed_tokens: int):
        result = self.get_num_new_matched_tokens_with_metadata(
            request,
            num_computed_tokens,
        )
        return result[0], result[1]

    def get_num_new_matched_tokens_with_metadata(
        self,
        request: Any,
        num_computed_tokens: int,
    ) -> tuple[int, bool, SemanticKvLookupResult | None]:
        token_ids = tuple(int(token) for token in request.get_tokens(0))
        prompt_text = _prompt_text(request)
        lookup = SemanticKvLookupRequest(
            request_id=int(request.request_id),
            token_ids=token_ids,
            prompt_text=prompt_text,
            namespace=self._namespace,
            num_computed_tokens=int(num_computed_tokens),
            block_hashes=tuple(getattr(request, "block_hashes", ()) or ()),
            cache_salt=getattr(request, "cache_salt", None),
            allow_non_identical=True,
            allow_segmented=os.environ.get("SEMBLEND_TRTLLM_ENABLE_SEGMENTED", "0") == "1",
            max_segments=int(os.environ.get("SEMBLEND_TRTLLM_MAX_SEGMENTS", "8")),
        )
        result = self._provider.lookup(lookup)
        num_new_matched_tokens = 0
        if result.found and result.plan is not None:
            num_new_matched_tokens = max(
                0,
                int(getattr(result.plan, "computed_token_count", 0)) - int(num_computed_tokens),
            )
            if not _enable_materialization():
                num_new_matched_tokens = 0
        _write_audit(
            {
                "event": "lookup",
                "request_id": int(request.request_id),
                "found": result.found,
                "materialization_enabled": _enable_materialization(),
                "num_computed_tokens": int(num_computed_tokens),
                "num_new_matched_tokens": num_new_matched_tokens,
                "prefix_token_count": (
                    int(result.plan.prefix_token_count)
                    if result.found and result.plan is not None
                    else 0
                ),
                "computed_token_count": (
                    int(getattr(result.plan, "computed_token_count", 0))
                    if result.found and result.plan is not None
                    else 0
                ),
                "kind": result.plan.kind.value if result.found and result.plan is not None else "",
                "similarity": result.similarity,
                "reuse_ratio": result.reuse_ratio,
                "rejection_reason": result.rejection_reason,
            }
        )
        if not result.found or result.plan is None:
            return 0, False, result
        if not _enable_materialization():
            return 0, False, result

        self._pending_plans[int(request.request_id)] = result.plan
        return num_new_matched_tokens, False, result

    def update_state_after_alloc(self, request: Any, block_ids: list[int]):
        req_id = int(request.request_id)
        plan = self._pending_plans.get(req_id)
        if plan is not None:
            self._pending_plans[req_id] = plan.with_target_block_ids(block_ids)
            _write_audit(
                {
                    "event": "target_allocated",
                    "request_id": req_id,
                    "block_count": len(block_ids),
                    "block_ids": list(block_ids[:16]),
                }
            )

    def build_connector_meta(self, scheduler_output: SchedulerOutput):
        metadata = SemBlendConnectorMetadata()
        for req_data in list(scheduler_output.new_requests) + list(
            scheduler_output.cached_requests
        ):
            result = getattr(req_data, "semantic_kv_result", None)
            if result is not None and not result.found:
                continue
            plan = self._pending_plans.pop(int(req_data.request_id), None)
            if plan is not None:
                metadata.plans[int(req_data.request_id)] = plan
                _write_audit(
                    {
                        "event": "plan_dispatched",
                        "request_id": int(req_data.request_id),
                        "computed_position": int(req_data.computed_position),
                        "num_scheduled_tokens": int(req_data.num_scheduled_tokens),
                        "new_block_count": len(req_data.new_block_ids),
                        "new_block_ids": list(req_data.new_block_ids[:16]),
                        "target_block_count": len(
                            plan.segments[0].target_block_ids if plan.segments else ()
                        ),
                        "target_block_ids": list(
                            (plan.segments[0].target_block_ids if plan.segments else ())[:16]
                        ),
                        "donor_block_count": len(
                            plan.segments[0].donor_block_ids if plan.segments else ()
                        ),
                        "donor_block_ids": list(
                            (plan.segments[0].donor_block_ids if plan.segments else ())[:16]
                        ),
                    }
                )
        metadata.events.extend(self._events)
        self._events = []
        return metadata

    def request_finished(self, request: Any, cache_block_ids: list[int]) -> bool:
        token_ids = list(int(token) for token in request.get_tokens(0))
        prompt_text = _prompt_text(request)
        event = self._provider.register_completed(
            request_id=str(request.request_id),
            token_ids=token_ids,
            prompt_text=prompt_text,
            namespace=self._namespace,
            block_ids=list(cache_block_ids),
            block_hashes=list(getattr(request, "block_hashes", ()) or ()),
            location=DonorLocation.worker(worker_id=0, dp_rank=0),
        )
        if event is not None:
            _write_audit(
                {
                    "event": "donor_registered",
                    "request_id": int(request.request_id),
                    "token_count": len(token_ids),
                    "block_count": len(cache_block_ids),
                    "block_ids": list(cache_block_ids[:16]),
                    "pin_donor": _pin_donors(),
                    "provider_generation": event.provider_generation,
                }
            )
            self._event_id += 1
            self._events.append(
                SemanticKvEvent(
                    schema_version=1,
                    event_id=self._event_id,
                    worker_id=0,
                    dp_rank=0,
                    data=event,
                )
            )
            if self._contract_emitter is not None:
                emitted = self._contract_emitter.donor_registered(
                    event,
                    prompt_text=prompt_text,
                    token_ids=token_ids,
                )
                _write_audit(
                    {
                        "event": "contract_event_published",
                        "request_id": int(request.request_id),
                        "kind": event.kind,
                        "published": emitted,
                    }
                )
        return _pin_donors()

    def wait_for_initialization(self):
        return


class SemBlendKvConnectorWorker(KvCacheConnectorWorker):
    """Worker-side materializer for request-local SemBlend plans."""

    def __init__(self, llm_args: Any):
        super().__init__(llm_args)
        patched_models = install_engine_patch()
        namespace = build_cache_namespace(llm_args=llm_args)
        self._provider = SemBlendTensorRTProvider(
            model_name=namespace.model,
            chunk_size=namespace.block_size,
            allow_segmented=os.environ.get("SEMBLEND_TRTLLM_ENABLE_SEGMENTED", "0") == "1",
            min_match_length=int(os.environ.get("SEMBLEND_TRTLLM_MIN_MATCH_LENGTH", "128")),
        )
        self._kv_cache_tensor = None
        self._loaded_req_ids: set[int] = set()
        _write_audit(
            {
                "event": "worker_initialized",
                "namespace": namespace.to_dict(),
                "patched_engine_models": patched_models,
            }
        )

    def register_kv_caches(self, kv_cache_tensor: Any):
        self._kv_cache_tensor = kv_cache_tensor
        _write_audit(
            {
                "event": "kv_cache_registered",
                "tensor_type": type(kv_cache_tensor).__name__,
                "shape": list(getattr(kv_cache_tensor, "shape", ()) or ()),
                "stride": (
                    list(kv_cache_tensor.stride()) if hasattr(kv_cache_tensor, "stride") else []
                ),
                "dtype": str(getattr(kv_cache_tensor, "dtype", "")),
                "device": str(getattr(kv_cache_tensor, "device", "")),
            }
        )

    def start_load_kv(self, stream: Any):
        metadata = self.get_connector_meta()
        if metadata is None or self._kv_cache_tensor is None:
            return
        for req_id, plan in metadata.plans.items():
            materialized = self._provider.materialize_plan(
                plan,
                self._kv_cache_tensor,
                stream=stream,
            )
            if materialized:
                set_active_plan(req_id, plan)
                self._loaded_req_ids.add(req_id)
                _write_audit(
                    {
                        "event": "materialized",
                        "request_id": req_id,
                        "materialized": materialized,
                        "kind": plan.kind.value,
                        "segment_count": len(plan.segments),
                        "covered_token_count": plan.covered_token_count,
                        "prefix_token_count": plan.prefix_token_count,
                        "computed_token_count": getattr(plan, "computed_token_count", 0),
                        "requires_rope_correction": plan.requires_rope_correction,
                        "rope_config": plan.namespace.rope_config,
                    }
                )
                logger.info(
                    "SemBlend TRT-LLM materialized request=%s tokens=%d kind=%s",
                    req_id,
                    materialized,
                    plan.kind.value,
                )

    def wait_for_layer_load(self, layer_idx: int, stream: Any):
        return

    def save_kv_layer(self, layer_idx: int, stream: Any):
        return

    def get_finished(
        self,
        finished_gen_req_ids: list[int],
        started_loading_req_ids: list[int],
    ) -> tuple[list[int], list[int]]:
        del finished_gen_req_ids
        finished_loading = [
            req_id for req_id in started_loading_req_ids if req_id in self._loaded_req_ids
        ]
        for req_id in finished_loading:
            self._loaded_req_ids.discard(req_id)
        return [], finished_loading

    def wait_for_save(self, stream: Any):
        del stream
        metadata = self.get_connector_meta()
        if metadata is None:
            return
        for req_id in metadata.plans:
            clear_active_plan(req_id)


def _prompt_text(request: Any) -> str:
    for attr in ("prompt", "prompt_text", "text"):
        value = getattr(request, attr, None)
        if isinstance(value, str):
            return value
    return ""


def _write_audit(payload: dict[str, Any]) -> None:
    path = os.environ.get("SEMBLEND_TRTLLM_AUDIT_PATH")
    if not path:
        return
    payload = dict(payload)
    payload.setdefault("source", "semblend.trtllm.connector")
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")
    except OSError:
        logger.debug("failed to write SemBlend TRT-LLM audit event", exc_info=True)


def _pin_donors() -> bool:
    return os.environ.get("SEMBLEND_TRTLLM_PIN_DONORS", "0") == "1"


def _enable_materialization() -> bool:
    return os.environ.get("SEMBLEND_TRTLLM_ENABLE_MATERIALIZATION", "0") == "1"

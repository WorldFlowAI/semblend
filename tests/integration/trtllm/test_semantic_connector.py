"""Tests for the TensorRT-LLM semantic connector package."""

from __future__ import annotations

import json
import sys
import types
import uuid
from types import SimpleNamespace

import pytest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _namespace():
    from semblend.integration.trtllm.namespace import build_cache_namespace

    return build_cache_namespace(
        model="Qwen/Qwen2.5-7B-Instruct",
        tokenizer="Qwen/Qwen2.5-7B-Instruct",
        model_revision="model-rev",
        tokenizer_revision="tokenizer-rev",
        block_size=4,
        kv_dtype="bfloat16",
        cache_dtype="fp8",
        quantization="awq",
        adapter="lora-a",
        rope_config={"rope_theta": 10000.0},
        tensor_parallel={"tp_size": 1, "tp_rank": 0},
    )


class _FakePipeline:
    donor_count = 1

    def __init__(self):
        self.registered = []

    def register_donor(self, **kwargs):
        self.registered.append(kwargs)

    def find_donor(self, **kwargs):
        del kwargs
        return SimpleNamespace(
            found=True,
            donor_id="req-1",
            similarity=0.91,
            reuse_ratio=0.75,
            donor_tokens=list(range(8)),
            slot_actions=[
                {"action": "copy_from_donor", "donorPos": 0, "targetPos": 0},
                {"action": "copy_from_donor", "donorPos": 1, "targetPos": 1},
                {"action": "recompute", "targetPos": 2},
            ],
            layer_deviations=[
                {"layerIdx": 0, "shouldRecompute": False},
                {"layerIdx": 1, "shouldRecompute": True},
            ],
            position_map=SimpleNamespace(donor_positions=[0, 1], target_positions=[0, 1]),
            timings=SimpleNamespace(
                embed_ms=1.0,
                lookup_ms=2.0,
                align_ms=3.0,
                bathtub_ms=4.0,
                total_ms=10.0,
            ),
            confidence_tier="verified_reuse",
            fuzzy_confidence=0.88,
            force_verify_layers=[1],
            chunk_fast_path_used=False,
        )


def test_namespace_key_changes_with_hard_fields():
    from semblend.integration.trtllm.namespace import build_cache_namespace, namespace_key

    base = _namespace()
    changed = build_cache_namespace(
        model=base.model,
        tokenizer=base.tokenizer,
        model_revision="different",
        tokenizer_revision=base.tokenizer_revision,
        block_size=base.block_size,
        kv_dtype=base.kv_dtype,
        cache_dtype=base.cache_dtype,
        quantization=base.quantization,
        rope_config=base.rope_config,
    )

    assert namespace_key(base) != namespace_key(changed)


def test_namespace_includes_env_routing_extra(monkeypatch):
    from semblend.integration.trtllm.namespace import build_cache_namespace

    monkeypatch.setenv("SEMBLEND_DONOR_TENANT", "wf-commercial")
    monkeypatch.setenv("SEMBLEND_DONOR_TEMPLATE", "wf-rag-v1")

    namespace = build_cache_namespace(model="m", tokenizer="m", block_size=16)

    assert namespace.extra["tenant"] == "wf-commercial"
    assert namespace.extra["template"] == "wf-rag-v1"


def test_trtllm_contract_emitter_builds_synapse_wire_event():
    from semblend.integration.trtllm.events import TrtllmContractEmitter
    from semblend.integration.trtllm.semblend_provider import SemBlendTensorRTProvider

    class FakeEmbedder:
        def embed(self, text):
            assert "policy" in text
            return [0.25, -0.5]

    namespace = _namespace()
    captured = []
    provider = SemBlendTensorRTProvider(chunk_size=namespace.block_size, min_match_length=1)
    provider._pipeline = _FakePipeline()
    event = provider.register_completed(
        request_id="req-1",
        token_ids=list(range(8)),
        prompt_text="policy donor text",
        namespace=namespace,
        block_ids=[10, 11],
        block_hashes=[111, 222],
    )
    assert event is not None

    emitter = TrtllmContractEmitter(
        worker_id=3,
        namespace=namespace,
        sink=captured.append,
        provider_generation=1234,
    )
    emitter._embedder = FakeEmbedder()  # noqa: SLF001

    assert emitter.donor_registered(
        event,
        prompt_text="policy donor text",
        token_ids=list(range(8)),
    )

    assert len(captured) == 1
    wire = captured[0]
    assert wire["schema_version"] == 1
    assert wire["worker_id"] == 3
    assert wire["data"]["kind"] == "donor_registered"
    assert wire["data"]["location"]["worker_id"] == 3
    assert wire["data"]["namespace"]["extra"] == namespace.extra
    assert wire["data"]["segments"][0]["provider_metadata"]


def test_connector_classes_import_without_tensorrt_llm():
    from semblend.integration.trtllm.connector import (
        SemBlendKvConnectorScheduler,
        SemBlendKvConnectorWorker,
    )

    assert SemBlendKvConnectorScheduler is not None
    assert SemBlendKvConnectorWorker is not None


def test_runtime_state_gates_active_plan_on_engine_blend_flag(monkeypatch):
    from semblend.integration.trtllm.runtime_state import (
        clear_active_plans,
        get_active_plan,
        set_active_plan,
    )

    clear_active_plans()
    plan = SimpleNamespace(request_id=7)
    set_active_plan(7, plan)

    monkeypatch.delenv("SEMBLEND_TRTLLM_ENGINE_BLEND", raising=False)
    assert get_active_plan(7) is None

    monkeypatch.setenv("SEMBLEND_TRTLLM_ENGINE_BLEND", "1")
    assert get_active_plan(7) is plan

    clear_active_plans()


def test_provider_builds_request_local_plan_from_pipeline_result(monkeypatch):
    from semblend.integration.trtllm.semblend_provider import SemBlendTensorRTProvider
    from semblend.integration.trtllm.upstream_interface import SemanticKvLookupRequest

    monkeypatch.delenv("SEMBLEND_TRTLLM_TRUST_NON_IDENTICAL_KV", raising=False)

    namespace = _namespace()
    provider = SemBlendTensorRTProvider(
        model_name=namespace.model,
        chunk_size=namespace.block_size,
        min_match_length=1,
    )
    provider._pipeline = _FakePipeline()
    event = provider.register_completed(
        request_id="req-1",
        token_ids=list(range(8)),
        prompt_text="donor text",
        namespace=namespace,
        block_ids=[10, 11],
        block_hashes=[111, 222],
    )
    assert event is not None

    result = provider.lookup(
        SemanticKvLookupRequest(
            request_id=7,
            token_ids=tuple(range(8)),
            prompt_text="query text",
            namespace=namespace,
            allow_segmented=False,
        )
    )

    assert result.found is True
    assert result.plan is not None
    assert result.plan.prefix_token_count == 2
    assert result.plan.computed_token_count == 0
    assert result.plan.engine_execution is None
    assert result.plan.publication_policy.value == "request_local"
    assert result.plan.segments[0].donor_block_ids == (10, 11)
    assert result.quality_signals["force_verify_layers"] == [1]


def test_provider_builds_engine_execution_policy_when_enabled(monkeypatch):
    from semblend.integration.trtllm.semblend_provider import SemBlendTensorRTProvider
    from semblend.integration.trtllm.upstream_interface import (
        SemanticKvEngineAttentionMode,
        SemanticKvLookupRequest,
    )

    monkeypatch.setenv("SEMBLEND_TRTLLM_ENGINE_BLEND", "1")
    monkeypatch.setenv("SEMBLEND_TRTLLM_RECOMPUTE_LAYERS", "24")
    monkeypatch.setenv("SEMBLEND_TRTLLM_FORCE_RECOMPUTE_LAYERS", "0,1,2,3")

    namespace = _namespace()
    provider = SemBlendTensorRTProvider(
        model_name=namespace.model,
        chunk_size=namespace.block_size,
        min_match_length=1,
    )
    provider._pipeline = _FakePipeline()
    event = provider.register_completed(
        request_id="req-1",
        token_ids=list(range(8)),
        prompt_text="donor text",
        namespace=namespace,
        block_ids=[10, 11],
        block_hashes=[111, 222],
    )
    assert event is not None

    result = provider.lookup(
        SemanticKvLookupRequest(
            request_id=7,
            token_ids=tuple(range(8)),
            prompt_text="query text",
            namespace=namespace,
            allow_segmented=False,
        )
    )

    assert result.plan is not None
    execution = result.plan.engine_execution
    assert execution is not None
    assert execution.attention_mode == SemanticKvEngineAttentionMode.SUFFIX_ONLY_AFTER_PREFIX
    assert execution.materialized_prefix_token_count == result.plan.prefix_token_count
    assert execution.suffix_start_position == result.plan.prefix_token_count
    assert execution.recompute_boundary_layer == 24
    assert execution.force_recompute_layers == (0, 1, 2, 3)
    assert result.plan.diagnostics["engine_execution"] == execution.to_dict()


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_engine_patch_wraps_capability_based_decoder_model(monkeypatch, tmp_path):
    from semblend.integration.trtllm.engine_patch import install_engine_patch
    from semblend.integration.trtllm.runtime_state import clear_active_plans, set_active_plan
    from semblend.integration.trtllm.upstream_interface import (
        SemanticKvEngineAttentionMode,
        SemanticKvEngineExecution,
        SemanticKvMaterializationKind,
        SemanticKvPlan,
        SemanticKvPublicationPolicy,
    )

    class FakeKvCacheParams:
        num_cached_tokens_per_seq = []

    class FakeAttentionMetadata:
        request_ids = [7]
        kv_cache_params = FakeKvCacheParams()
        prepare_calls = 0

        def prepare(self):
            self.prepare_calls += 1

    class FakeEmbedding:
        def __call__(self, input_ids):
            return input_ids.float().unsqueeze(-1)

    class FakeLayer:
        def __init__(self, layer_idx):
            self.layer_idx = layer_idx

        def __call__(
            self,
            *,
            position_ids,
            hidden_states,
            attn_metadata,
            residual,
            mrope_config,
            spec_metadata,
            **kwargs,
        ):
            del position_ids, attn_metadata, mrope_config, spec_metadata, kwargs
            return hidden_states + float(self.layer_idx + 1), residual

    class FakeNorm:
        def __call__(self, hidden_states, residual):
            return hidden_states, residual

    class FakeDecoderModel:
        def __init__(self):
            self.embed_tokens = FakeEmbedding()
            self.layers = [FakeLayer(0), FakeLayer(1), FakeLayer(2)]
            self.norm = FakeNorm()

        def forward(
            self,
            attn_metadata,
            input_ids=None,
            position_ids=None,
            inputs_embeds=None,
            mrope_config=None,
            spec_metadata=None,
            **kwargs,
        ):
            del attn_metadata, input_ids, position_ids, inputs_embeds
            del mrope_config, spec_metadata, kwargs
            return torch.full((1, 1), -1.0)

    module_name = f"tensorrt_llm._torch.models.modeling_fake_{uuid.uuid4().hex}"
    module = types.ModuleType(module_name)
    module.FakeDecoderModel = FakeDecoderModel
    sys.modules[module_name] = module

    audit_path = tmp_path / "audit.jsonl"
    monkeypatch.setenv("SEMBLEND_TRTLLM_ENGINE_BLEND", "1")
    monkeypatch.setenv("SEMBLEND_TRTLLM_ENGINE_PATCH_MODULES", module_name)
    monkeypatch.setenv("SEMBLEND_TRTLLM_AUDIT_PATH", str(audit_path))
    clear_active_plans()
    plan = SemanticKvPlan(
        request_id=7,
        namespace=_namespace(),
        kind=SemanticKvMaterializationKind.REQUEST_LOCAL_PREFIX,
        publication_policy=SemanticKvPublicationPolicy.REQUEST_LOCAL,
        prefix_token_count=2,
        engine_execution=SemanticKvEngineExecution(
            attention_mode=SemanticKvEngineAttentionMode.SUFFIX_ONLY_AFTER_PREFIX,
            materialized_prefix_token_count=2,
            suffix_start_position=2,
            recompute_boundary_layer=1,
        ),
    )
    set_active_plan(7, plan)

    patched = install_engine_patch()
    assert any(name.endswith(".FakeDecoderModel") for name in patched)

    model = FakeDecoderModel()
    out = model.forward(
        attn_metadata=FakeAttentionMetadata(),
        input_ids=torch.arange(5),
        position_ids=torch.arange(5),
    )

    assert tuple(out.shape) == (3, 1)
    events = [json.loads(line) for line in audit_path.read_text().splitlines()]
    assert any(event.get("event") == "engine_blend_boundary" for event in events)
    clear_active_plans()


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_provider_materializes_token_kv_and_preserves_recompute_layer():
    from semblend.integration.trtllm.semblend_provider import SemBlendTensorRTProvider
    from semblend.integration.trtllm.upstream_interface import (
        SemanticKvMaterializationKind,
        SemanticKvPlan,
        SemanticKvPublicationPolicy,
        SemanticKvSegment,
    )

    namespace = _namespace()
    provider = SemBlendTensorRTProvider(chunk_size=namespace.block_size, min_match_length=1)
    kv = torch.zeros(4, 2, 2, 4, 1, 4, dtype=torch.float32)
    kv[0, 0, :, 1, :, :] = 3.0
    kv[0, 1, :, 1, :, :] = 9.0

    segment = SemanticKvSegment(
        donor_id="req-1",
        donor_segment_id=0,
        donor_positions=(1,),
        target_positions=(1,),
        donor_block_ids=(0,),
        target_block_ids=(2,),
        layer_recompute_mask=(False, True),
    )
    plan = SemanticKvPlan(
        request_id=7,
        namespace=namespace,
        kind=SemanticKvMaterializationKind.REQUEST_LOCAL_PREFIX,
        publication_policy=SemanticKvPublicationPolicy.REQUEST_LOCAL,
        segments=(segment,),
        donor_ids=("req-1",),
        covered_token_count=1,
        prefix_token_count=1,
    )

    copied = provider.materialize_plan(plan, kv)

    assert copied == 1
    assert torch.allclose(kv[2, 0, :, 1, :, :], torch.full((2, 1, 4), 3.0))
    assert torch.allclose(kv[2, 1, :, 1, :, :], torch.zeros(2, 1, 4))


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_provider_materializes_trtllm_hnd_flat_4d_layout(monkeypatch):
    from semblend.integration.trtllm.semblend_provider import SemBlendTensorRTProvider
    from semblend.integration.trtllm.upstream_interface import (
        SemanticKvMaterializationKind,
        SemanticKvPlan,
        SemanticKvPublicationPolicy,
        SemanticKvSegment,
    )

    monkeypatch.setenv("SEMBLEND_TRTLLM_HEAD_DIM", "3")
    monkeypatch.delenv("SEMBLEND_TRTLLM_FLAT_LAYOUT", raising=False)

    namespace = _namespace()
    provider = SemBlendTensorRTProvider(chunk_size=namespace.block_size, min_match_length=1)
    kv = torch.zeros(3, 2, 2, 2 * namespace.block_size * 3, dtype=torch.float32)
    hnd = kv.view(3, 2, 2, 2, namespace.block_size, 3)
    hnd[0, 0, :, :, 1, :] = 7.0
    hnd[0, 1, :, :, 1, :] = 11.0

    segment = SemanticKvSegment(
        donor_id="req-1",
        donor_segment_id=0,
        donor_positions=(1,),
        target_positions=(1,),
        donor_block_ids=(0,),
        target_block_ids=(2,),
        layer_recompute_mask=(False, True),
    )
    plan = SemanticKvPlan(
        request_id=7,
        namespace=namespace,
        kind=SemanticKvMaterializationKind.REQUEST_LOCAL_PREFIX,
        publication_policy=SemanticKvPublicationPolicy.REQUEST_LOCAL,
        segments=(segment,),
        donor_ids=("req-1",),
        covered_token_count=1,
        prefix_token_count=1,
    )

    copied = provider.materialize_plan(plan, kv)

    assert copied == 1
    assert torch.allclose(hnd[2, 0, :, :, 1, :], torch.full((2, 2, 3), 7.0))
    assert torch.allclose(hnd[2, 1, :, :, 1, :], torch.zeros(2, 2, 3))


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_rope_correction_defaults_to_half_split(monkeypatch):
    from semblend.integration.trtllm.semblend_provider import _rope_correct_k

    monkeypatch.delenv("SEMBLEND_TRTLLM_ROPE_STYLE", raising=False)

    k = torch.tensor([[1.0, 2.0, 10.0, 20.0]])
    corrected = _rope_correct_k(k, delta=1, rope_base=10000.0)
    inv_freq = torch.tensor([1.0, 0.01])
    cos = torch.cos(inv_freq)
    sin = torch.sin(inv_freq)
    expected = torch.empty_like(k)
    expected[..., :2] = k[..., :2] * cos - k[..., 2:] * sin
    expected[..., 2:] = k[..., 2:] * cos + k[..., :2] * sin

    assert torch.allclose(corrected, expected)

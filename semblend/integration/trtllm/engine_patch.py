"""Backend-neutral TensorRT-LLM semantic KV engine patch.

This module lifts the prototype Qwen boundary hook into a reusable wrapper for
TRT-LLM decoder models. It is intentionally capability-based: a model is
patchable when its ``forward`` accepts TRT ``attn_metadata`` and the instance has
decoder ``layers`` plus a final ``norm``. The semantic policy itself comes from
``SemanticKvPlan.engine_execution`` and segment layer-recompute masks, not from a
model-specific branch.
"""

from __future__ import annotations

import importlib
import inspect
import os
import sys
from functools import wraps
from typing import Any

_PATCHED: set[type] = set()


def install_engine_patch() -> list[str]:
    """Install the SemBlend engine patch on compatible TRT-LLM model classes.

    Returns a list of patched ``module.Class`` names. The function is idempotent
    and safe to call from both scheduler and worker processes.
    """

    if os.environ.get("SEMBLEND_TRTLLM_ENGINE_BLEND", "0") != "1":
        return []

    patched: list[str] = []
    for module_name in _candidate_modules():
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls in _PATCHED or not _is_patchable_model_class(cls):
                continue
            if not _class_allowed(cls):
                continue
            _patch_model_class(cls)
            _PATCHED.add(cls)
            patched.append(f"{cls.__module__}.{cls.__name__}")
    return patched


def _candidate_modules() -> list[str]:
    configured = os.environ.get("SEMBLEND_TRTLLM_ENGINE_PATCH_MODULES", "")
    modules = [part.strip() for part in configured.split(",") if part.strip()]
    modules.extend(
        name
        for name in sys.modules
        if name.startswith("tensorrt_llm._torch.models.modeling_")
    )
    modules.extend(
        [
            "tensorrt_llm._torch.models.modeling_qwen",
            "tensorrt_llm._torch.models.modeling_llama",
            "tensorrt_llm._torch.models.modeling_gpt",
            "tensorrt_llm._torch.models.modeling_mistral",
        ]
    )
    return list(dict.fromkeys(modules))


def _is_patchable_model_class(cls: type) -> bool:
    forward = getattr(cls, "forward", None)
    if forward is None or getattr(forward, "_semblend_engine_patch", False):
        return False
    try:
        params = inspect.signature(forward).parameters
    except (TypeError, ValueError):
        return False
    if "attn_metadata" not in params:
        return False
    if "position_ids" not in params:
        return False
    return "input_ids" in params or "inputs_embeds" in params


def _class_allowed(cls: type) -> bool:
    configured = os.environ.get("SEMBLEND_TRTLLM_ENGINE_PATCH_CLASSES", "")
    allowed = {part.strip() for part in configured.split(",") if part.strip()}
    if allowed:
        return cls.__name__ in allowed or f"{cls.__module__}.{cls.__name__}" in allowed

    # Conservative default: patch concrete decoder model classes, not causal-LM
    # wrappers, reward heads, speculative draft wrappers, or the abstract base.
    name = cls.__name__
    if name in {"DecoderModel", "DecoderModelForCausalLM"}:
        return False
    if not name.endswith("Model"):
        return False
    excluded_tokens = ("Bert", "Reward", "Draft", "SpecDec", "VLM", "VL")
    return not any(token in name for token in excluded_tokens)


def _patch_model_class(cls: type) -> None:
    original_forward = cls.forward

    @wraps(original_forward)
    def forward(self: Any, *args: Any, **kwargs: Any) -> Any:
        if os.environ.get("SEMBLEND_TRTLLM_ENGINE_BLEND", "0") != "1":
            return original_forward(self, *args, **kwargs)

        bound = _bind_forward_args(original_forward, self, args, kwargs)
        attn_metadata = bound.arguments.get("attn_metadata")
        input_ids = bound.arguments.get("input_ids")
        inputs_embeds = bound.arguments.get("inputs_embeds")
        position_ids = bound.arguments.get("position_ids")
        mrope_config = bound.arguments.get("mrope_config")
        spec_metadata = bound.arguments.get("spec_metadata")
        extra_kwargs = dict(bound.arguments.get("kwargs", {}) or {})

        layers = getattr(self, "layers", None)
        norm = getattr(self, "norm", None)
        embed_tokens = getattr(self, "embed_tokens", None)
        if layers is None or norm is None or attn_metadata is None:
            return original_forward(self, *args, **kwargs)

        hidden_states = inputs_embeds
        if hidden_states is None:
            if input_ids is None or embed_tokens is None:
                return original_forward(self, *args, **kwargs)
            hidden_states = embed_tokens(input_ids)

        plan = _get_active_plan(attn_metadata, int(getattr(hidden_states, "shape", [0])[0]))
        if plan is None:
            return original_forward(self, *args, **kwargs)

        result = _run_engine_plan(
            layers=layers,
            norm=norm,
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            plan=plan,
            mrope_config=mrope_config,
            spec_metadata=spec_metadata,
            kwargs=extra_kwargs,
        )
        if result is None:
            return original_forward(self, *args, **kwargs)
        return result

    forward._semblend_engine_patch = True  # type: ignore[attr-defined]
    cls.forward = forward


def _bind_forward_args(forward: Any, self: Any, args: tuple[Any, ...], kwargs: dict[str, Any]):
    signature = inspect.signature(forward)
    bound = signature.bind_partial(self, *args, **kwargs)
    bound.apply_defaults()
    return bound


def _get_active_plan(attn_metadata: Any, token_count: int) -> Any | None:
    if token_count <= 1:
        return None
    request_ids = getattr(attn_metadata, "request_ids", None)
    try:
        from semblend.integration.trtllm.runtime_state import (
            get_active_plan,
            get_only_active_plan,
        )

        if request_ids is not None and len(request_ids) == 1:
            plan = get_active_plan(int(request_ids[0]))
            if plan is not None:
                return plan
        return get_only_active_plan()
    except Exception:
        return None


def _run_engine_plan(
    *,
    layers: Any,
    norm: Any,
    position_ids: Any,
    hidden_states: Any,
    attn_metadata: Any,
    plan: Any,
    mrope_config: Any,
    spec_metadata: Any,
    kwargs: dict[str, Any],
) -> Any | None:
    execution = getattr(plan, "engine_execution", None)
    if execution is None or not getattr(execution, "uses_suffix_only_attention", False):
        return None

    prefix_len = int(getattr(execution, "materialized_prefix_token_count", 0))
    total_len = int(getattr(hidden_states, "shape", [0])[0])
    if prefix_len <= 0 or prefix_len >= total_len:
        return None

    if _layerwise_blend_enabled():
        result = _run_layerwise(
            layers=layers,
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            residual=None,
            plan=plan,
            execution=execution,
            mrope_config=mrope_config,
            spec_metadata=spec_metadata,
            kwargs=kwargs,
        )
    else:
        result = _run_boundary(
            layers=layers,
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            residual=None,
            execution=execution,
            mrope_config=mrope_config,
            spec_metadata=spec_metadata,
            kwargs=kwargs,
        )
    if result is None:
        return None
    hidden_states, residual = result
    hidden_states, _ = norm(hidden_states, residual)
    return hidden_states


def _run_boundary(
    *,
    layers: Any,
    position_ids: Any,
    hidden_states: Any,
    attn_metadata: Any,
    residual: Any,
    execution: Any,
    mrope_config: Any,
    spec_metadata: Any,
    kwargs: dict[str, Any],
) -> tuple[Any, Any] | None:
    prefix_len = int(execution.materialized_prefix_token_count)
    total_len = int(hidden_states.shape[0])
    boundary = _boundary_layer(execution, len(layers))
    applied = False

    for layer in layers:
        layer_idx = int(getattr(layer, "layer_idx", 0))
        if not applied and layer_idx >= boundary:
            suffix_len = total_len - prefix_len
            hidden_states = hidden_states[prefix_len:]
            if residual is not None:
                residual = residual[prefix_len:]
            position_ids = _slice_position_ids(position_ids, prefix_len)
            _prepare_suffix_attention(attn_metadata, prefix_len, suffix_len)
            _write_audit(
                {
                    "event": "engine_blend_boundary",
                    "request_ids": list(getattr(attn_metadata, "request_ids", []) or []),
                    "boundary_layer": layer_idx,
                    "prefix_token_count": prefix_len,
                    "suffix_token_count": suffix_len,
                }
            )
            applied = True
        hidden_states, residual = layer(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            residual=residual,
            mrope_config=mrope_config,
            spec_metadata=spec_metadata,
            **kwargs,
        )
    return hidden_states, residual


def _run_layerwise(
    *,
    layers: Any,
    position_ids: Any,
    hidden_states: Any,
    attn_metadata: Any,
    residual: Any,
    plan: Any,
    execution: Any,
    mrope_config: Any,
    spec_metadata: Any,
    kwargs: dict[str, Any],
) -> tuple[Any, Any] | None:
    prefix_len = int(execution.materialized_prefix_token_count)
    total_len = int(hidden_states.shape[0])
    suffix_len = total_len - prefix_len
    full_position_ids = position_ids
    suffix_position_ids = _slice_position_ids(position_ids, prefix_len)
    recompute_mask = _recompute_mask(plan, execution, len(layers))
    reused_layers: list[int] = []
    mode = "full"

    for layer in layers:
        layer_idx = int(getattr(layer, "layer_idx", 0))
        if recompute_mask[layer_idx]:
            if mode != "full":
                _prepare_full_attention(attn_metadata, total_len, 0)
                mode = "full"
            _write_audit(
                {
                    "event": "engine_blend_layer",
                    "layer": layer_idx,
                    "mode": "full",
                    "query_len": total_len,
                    "cached_len": 0,
                }
            )
            hidden_states, residual = layer(
                position_ids=full_position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
                mrope_config=mrope_config,
                spec_metadata=spec_metadata,
                **kwargs,
            )
            continue

        if mode != "suffix":
            _prepare_suffix_attention(attn_metadata, prefix_len, suffix_len)
            mode = "suffix"
        _write_audit(
            {
                "event": "engine_blend_layer",
                "layer": layer_idx,
                "mode": "suffix",
                "query_len": suffix_len,
                "cached_len": prefix_len,
            }
        )

        prefix_hidden = hidden_states[:prefix_len]
        suffix_hidden = hidden_states[prefix_len:].clone()
        if residual is None:
            prefix_residual = prefix_hidden
            suffix_residual = None
        else:
            prefix_residual = residual[:prefix_len]
            suffix_residual = residual[prefix_len:].clone()

        suffix_hidden, suffix_residual = layer(
            position_ids=suffix_position_ids,
            hidden_states=suffix_hidden,
            attn_metadata=attn_metadata,
            residual=suffix_residual,
            mrope_config=mrope_config,
            spec_metadata=spec_metadata,
            **kwargs,
        )
        hidden_states = _torch_cat((prefix_hidden, suffix_hidden), dim=0)
        residual = (
            None
            if suffix_residual is None
            else _torch_cat((prefix_residual, suffix_residual), dim=0)
        )
        reused_layers.append(layer_idx)

    if mode != "full":
        _prepare_full_attention(attn_metadata, total_len, 0)
    _write_audit(
        {
            "event": "engine_blend_layerwise",
            "request_ids": list(getattr(attn_metadata, "request_ids", []) or []),
            "prefix_token_count": prefix_len,
            "suffix_token_count": suffix_len,
            "recompute_layers": [
                idx for idx, should_recompute in enumerate(recompute_mask) if should_recompute
            ],
            "reused_layers": reused_layers,
        }
    )
    return hidden_states, residual


def _boundary_layer(execution: Any, num_layers: int) -> int:
    boundary = execution.recompute_boundary_layer
    if boundary is None:
        boundary = 0
    return max(0, min(int(boundary), num_layers))


def _recompute_mask(plan: Any, execution: Any, num_layers: int) -> list[bool]:
    values = [False] * num_layers
    segments = getattr(plan, "segments", ()) or ()
    if segments:
        mask = getattr(segments[0], "layer_recompute_mask", None)
        if mask is not None:
            for idx, value in enumerate(mask[:num_layers]):
                values[idx] = bool(value)
    boundary = execution.recompute_boundary_layer
    if boundary is not None:
        for idx in range(min(int(boundary), num_layers)):
            values[idx] = True
    for layer_idx in getattr(execution, "force_recompute_layers", ()) or ():
        if 0 <= int(layer_idx) < num_layers:
            values[int(layer_idx)] = True
    return values


def _slice_position_ids(position_ids: Any, prefix_len: int) -> Any:
    if position_ids is None:
        return None
    if position_ids.dim() == 1:
        return position_ids[prefix_len:].clone()
    if position_ids.dim() == 2:
        return position_ids[:, prefix_len:].clone()
    if position_ids.dim() == 3:
        return position_ids[:, :, prefix_len:].clone()
    return position_ids


def _prepare_suffix_attention(attn_metadata: Any, prefix_len: int, suffix_len: int) -> None:
    import torch

    attn_metadata.seq_lens = torch.tensor([suffix_len], dtype=torch.int)
    attn_metadata.num_contexts = 1
    attn_metadata.prompt_lens = [suffix_len]
    attn_metadata.kv_cache_params.num_cached_tokens_per_seq = [prefix_len]
    attn_metadata.prepare()


def _prepare_full_attention(attn_metadata: Any, query_len: int, cached_len: int) -> None:
    import torch

    attn_metadata.seq_lens = torch.tensor([query_len], dtype=torch.int)
    attn_metadata.num_contexts = 1
    attn_metadata.prompt_lens = [query_len]
    attn_metadata.kv_cache_params.num_cached_tokens_per_seq = [cached_len]
    attn_metadata.prepare()


def _torch_cat(values: tuple[Any, Any], *, dim: int) -> Any:
    import torch

    return torch.cat(values, dim=dim)


def _layerwise_blend_enabled() -> bool:
    return os.environ.get("SEMBLEND_TRTLLM_LAYERWISE_BLEND", "0") == "1"


def _write_audit(payload: dict[str, Any]) -> None:
    try:
        from semblend.integration.trtllm.runtime_state import write_audit

        write_audit(payload)
    except Exception:
        return


__all__ = ["install_engine_patch"]

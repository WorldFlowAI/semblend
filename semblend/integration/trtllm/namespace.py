"""Namespace construction for TensorRT-LLM semantic KV reuse."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Mapping

from semblend.integration.trtllm.upstream_interface import CacheNamespace


def build_cache_namespace(
    *,
    llm_args: Any = None,
    model: str = "",
    tokenizer: str = "",
    model_revision: str = "",
    tokenizer_revision: str = "",
    kv_layout: str = "HND",
    block_size: int | None = None,
    kv_dtype: str = "",
    cache_dtype: str = "",
    quantization: str = "",
    adapter: str = "",
    rope_config: Mapping[str, Any] | None = None,
    tensor_parallel: Mapping[str, Any] | None = None,
    backend_cache_layout: str = "trtllm_pytorch_primary_pool_v1",
    extra: Mapping[str, Any] | None = None,
) -> CacheNamespace:
    """Build a conservative cache namespace from explicit fields and llm_args."""

    kv_cache_config = getattr(llm_args, "kv_cache_config", None)
    mapping = getattr(llm_args, "mapping", None)
    model_config = _first_present(
        getattr(llm_args, "model_config", None),
        getattr(llm_args, "pretrained_config", None),
        getattr(getattr(llm_args, "model", None), "model_config", None),
    )

    resolved_model = model or _string_attr(llm_args, "model") or os.environ.get(
        "SEMBLEND_MODEL_NAME", ""
    )
    resolved_tokenizer = tokenizer or _string_attr(llm_args, "tokenizer") or resolved_model
    resolved_block_size = int(
        block_size
        or getattr(kv_cache_config, "tokens_per_block", 0)
        or os.environ.get("SEMBLEND_TRTLLM_BLOCK_SIZE", "128")
    )

    model_configs = tuple(
        config
        for config in (model_config, getattr(model_config, "pretrained_config", None))
        if config is not None
    )

    resolved_rope = dict(rope_config or {})
    for src, dst in (
        ("rope_theta", "rope_theta"),
        ("rope_base", "rope_base"),
        ("rope_scaling", "rope_scaling"),
        ("max_position_embeddings", "max_position_embeddings"),
    ):
        for config in model_configs:
            value = getattr(config, src, None)
            if value is not None and dst not in resolved_rope:
                resolved_rope[dst] = _jsonable(value)
                break
    if "rope_theta" not in resolved_rope and "rope_base" not in resolved_rope:
        env_rope_base = os.environ.get("SEMBLEND_TRTLLM_ROPE_BASE")
        if env_rope_base:
            resolved_rope["rope_theta"] = float(env_rope_base)

    resolved_tp = dict(tensor_parallel or {})
    for src, dst in (
        ("tp_size", "tp_size"),
        ("tp_rank", "tp_rank"),
        ("pp_size", "pp_size"),
        ("pp_rank", "pp_rank"),
        ("cp_size", "cp_size"),
        ("cp_rank", "cp_rank"),
    ):
        value = getattr(mapping, src, None)
        if value is not None and dst not in resolved_tp:
            resolved_tp[dst] = _jsonable(value)

    resolved_quant = quantization or _string_attr(
        getattr(getattr(llm_args, "quant_config", None), "quant_algo", None), "name"
    )
    if not resolved_quant:
        resolved_quant = _string_attr(getattr(model_config, "quantization_config", None), "quant_method")

    resolved_extra = dict(extra or {})
    tenant = os.environ.get("SEMBLEND_DONOR_TENANT")
    template = os.environ.get("SEMBLEND_DONOR_TEMPLATE")
    if tenant and "tenant" not in resolved_extra:
        resolved_extra["tenant"] = tenant
    if template and "template" not in resolved_extra:
        resolved_extra["template"] = template

    return CacheNamespace(
        model=resolved_model,
        tokenizer=resolved_tokenizer,
        model_revision=model_revision or os.environ.get("SEMBLEND_MODEL_REVISION", ""),
        tokenizer_revision=tokenizer_revision
        or os.environ.get("SEMBLEND_TOKENIZER_REVISION", ""),
        kv_layout=kv_layout,
        block_size=resolved_block_size,
        kv_dtype=kv_dtype or _string_attr(llm_args, "dtype"),
        cache_dtype=cache_dtype or _string_attr(kv_cache_config, "dtype"),
        quantization=resolved_quant,
        adapter=adapter or os.environ.get("SEMBLEND_ADAPTER_ID", ""),
        rope_config=resolved_rope,
        tensor_parallel=resolved_tp,
        backend_cache_layout=backend_cache_layout,
        extra=resolved_extra,
    )


def namespace_key(namespace: CacheNamespace) -> str:
    payload = json.dumps(namespace.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _string_attr(obj: Any, name: str) -> str:
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    value = getattr(obj, name, None)
    if value is None:
        return ""
    return str(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)

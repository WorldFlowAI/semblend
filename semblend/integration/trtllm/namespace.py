"""Namespace construction for TensorRT-LLM semantic KV reuse."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from typing import Any, Mapping

from semblend.integration.dynamo.semantic_events import (
    TENANT_KEY_EXTRA_FIELD,
    tenant_key_for_salt,
)
from semblend.integration.trtllm.upstream_interface import CacheNamespace

# Isolation namespace for requests that carry no cache_salt.
#
# Every namespace key carries a non-empty salt namespace so the isolation
# check is a plain equality compare with no "absent means every donor is
# visible" escape hatch. Two requests that both lack a cache_salt share this
# sentinel and MAY reuse each other's KV — single-tenant deployments keep
# working unchanged. A request that carries a cache_salt never maps to the
# sentinel, so a salted request can never consume an unsalted donor, and an
# unsalted request can never consume a salted one. Same scheme as the vLLM
# connector, so a salt isolates the same way on every engine.
NO_CACHE_SALT_NAMESPACE = "semblend:ns:no-cache-salt"

_CACHE_SALT_NAMESPACE_PREFIX = "semblend:ns:salt:"

# Where the hashed salt lives on a CacheNamespace. ``extra`` is the only
# extension point on the upstream TensorRT-LLM type, and it already carries
# the routing-policy fields (tenant, template) that namespace_key hashes, so
# the salt rides through to_dict() on donor events and plans the same way.
CACHE_SALT_EXTRA_FIELD = "cache_salt"


def cache_salt_namespace(cache_salt: Any) -> str:
    """Isolation namespace for a request's ``cache_salt``.

    The salt is hashed rather than stored verbatim: it is tenant-identifying
    and the namespace reaches donor records, wire events and logs.
    """
    if cache_salt is None:
        return NO_CACHE_SALT_NAMESPACE
    salt = cache_salt if isinstance(cache_salt, str) else str(cache_salt)
    salt = salt.strip()
    if not salt:
        # An empty salt carries no isolation intent — treat it as absent so
        # it lands in the sentinel namespace rather than in one of its own.
        return NO_CACHE_SALT_NAMESPACE
    digest = hashlib.sha256(salt.encode("utf-8")).hexdigest()[:32]
    return _CACHE_SALT_NAMESPACE_PREFIX + digest


def bind_cache_salt(namespace: CacheNamespace, cache_salt: Any = None) -> CacheNamespace:
    """Namespace with the request's hashed ``cache_salt`` bound into ``extra``.

    Both providers bind at registration and at lookup, so a donor is only
    reachable from requests carrying the same salt.

    An absent salt leaves the namespace untouched rather than stamping the
    sentinel: namespace_key defaults the missing field to the sentinel, the
    wire shape of an unsalted deployment does not change, and a binding a
    caller already made on this namespace is never erased by a later hop
    that no longer has the raw salt.
    """
    salt_namespace = cache_salt_namespace(cache_salt)
    if salt_namespace == NO_CACHE_SALT_NAMESPACE:
        return namespace
    extra = dict(namespace.extra)
    if extra.get(CACHE_SALT_EXTRA_FIELD) == salt_namespace:
        return namespace
    return replace(namespace, extra={**extra, CACHE_SALT_EXTRA_FIELD: salt_namespace})


def bind_tenant_key(namespace: CacheNamespace, cache_salt: Any = None) -> CacheNamespace:
    """Namespace with the request's tenant key bound into ``extra``.

    Two different keys ride on a published donor. ``cache_salt`` above is the
    engine-local isolation namespace: it is hashed from the salt the engine
    stores the KV under and it is what ``namespace_key`` keys the donor by.
    ``tenant_key`` is ``tenant key v1`` — a function of the raw salt alone —
    so a router holding the request can reproduce it and place on it. Both
    are stamped, never one instead of the other.

    The derivation is imported rather than repeated: one implementation of
    the contract per package, so the TRT-LLM and vLLM publishers cannot drift
    into two different keys for one salt. The field is always stamped,
    sentinel included, so an unsalted donor is explicitly unsalted on the
    wire rather than silent.

    This binds only for the wire. It must not be bound into the namespace
    ``namespace_key`` hashes: that key is the engine-local store's, and
    moving it would strand every donor registered before the change.
    """
    value = tenant_key_for_salt(cache_salt)
    extra = dict(namespace.extra or {})
    if extra.get(TENANT_KEY_EXTRA_FIELD) == value:
        return namespace
    return replace(namespace, extra={**extra, TENANT_KEY_EXTRA_FIELD: value})


def strip_tenant_key(namespace: CacheNamespace) -> CacheNamespace:
    """Namespace without the wire-only tenant key.

    The engine-local identity of a donor is the namespace ``namespace_key``
    hashes, and the tenant key is not part of it. A provider that rebuilds a
    donor handle from a published event has to drop the field again, or the
    handle compares unequal to the request namespace it came from and every
    lookup fails closed as cross-namespace.
    """
    extra = dict(namespace.extra or {})
    if TENANT_KEY_EXTRA_FIELD not in extra:
        return namespace
    del extra[TENANT_KEY_EXTRA_FIELD]
    return replace(namespace, extra=extra)


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

    resolved_model = (
        model or _string_attr(llm_args, "model") or os.environ.get("SEMBLEND_MODEL_NAME", "")
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
        resolved_quant = _string_attr(
            getattr(model_config, "quantization_config", None), "quant_method"
        )

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
        tokenizer_revision=tokenizer_revision or os.environ.get("SEMBLEND_TOKENIZER_REVISION", ""),
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
    """Donor-store isolation key for a namespace.

    The cache_salt field is always part of the hashed payload, defaulting to
    the no-cache-salt sentinel when the namespace was never bound. That is
    what makes absent-vs-salted a guaranteed mismatch: a salted namespace
    carries a hashed salt in the same slot the sentinel occupies.

    This deliberately changes the key of every namespace that predates the
    field, so donors keyed before the salt was part of the namespace — whose
    tenancy is unknown — are unreachable after upgrade instead of being
    served to whichever tenant asks first.
    """
    payload = namespace.to_dict()
    extra = dict(payload.get("extra") or {})
    extra.setdefault(CACHE_SALT_EXTRA_FIELD, NO_CACHE_SALT_NAMESPACE)
    keyed = {**payload, "extra": extra}
    encoded = json.dumps(keyed, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


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

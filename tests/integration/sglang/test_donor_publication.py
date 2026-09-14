"""The SGLang adapter announces its donors on the event plane.

This adapter embeds off the scheduler thread and inserts into the donor
store itself, so it used to bypass ``SemBlendPipeline.register_donor``
entirely — and with it the only code that publishes ``DonorRegistered``.
A fleet fed by SGLang therefore learned about no donors at all.

It now publishes through ``SemBlendPipeline.publish_donor_registered``,
carrying both keys: the engine-local isolation namespace it registers the
donor under, and the tenant key derived from the request's raw salt, which
is the only one a router holding the request can reproduce.
"""

from __future__ import annotations

import logging

import numpy as np

from semblend.integration.dynamo import semantic_events
from semblend.integration.dynamo.semantic_events import (
    ISOLATION_EXTRA_FIELD,
    NO_TENANT_KEY,
    TENANT_KEY_EXTRA_FIELD,
    CacheNamespace,
    tenant_key_for_salt,
)
from semblend.integration.sglang.config import SemBlendProviderConfig
from semblend.integration.sglang.namespace import isolation_namespace
from semblend.integration.sglang.provider import SemBlendProviderAdapter
from semblend.integration.vllm.events import VllmContractEmitter
from semblend_core.donor_store import DonorStore
from semblend_core.pipeline import SemBlendPipeline

DIM = 8
RADIX_KEY = "sglang:radix:tenant-acme:lora-0"
SALT = "tenant-acme"
PROMPT = "the quarterly report covers every regional business unit"


class _StubEmbedder:
    dimension = DIM

    def embed(self, text: str):
        return np.ones(DIM, dtype=np.float32) / np.sqrt(DIM)


def _pipeline(events: list[dict]) -> SemBlendPipeline:
    store = DonorStore(max_entries=16, embedding_dim=DIM, min_similarity=0.60, chunk_size=16)
    pipeline = SemBlendPipeline(
        embedder_type="jaccard",
        donor_store=store,
        chunk_size=16,
        enable_pq_segments=False,
    )
    pipeline._embedder = _StubEmbedder()  # noqa: SLF001
    pipeline._event_emitter = VllmContractEmitter(  # noqa: SLF001
        worker_id=5,
        namespace=CacheNamespace(model="qwen", tokenizer="qwen", kv_layout="sglang", block_size=16),
        sink=events.append,
    )
    return pipeline


def _adapter(pipeline: SemBlendPipeline) -> SemBlendProviderAdapter:
    config = SemBlendProviderConfig(
        min_similarity=0.60,
        min_reuse_ratio=0.50,
        min_match_length=8,
        max_entries=100,
        block_size=4,
        model_arch="llama",
    )
    return SemBlendProviderAdapter(config=config, pipeline=pipeline)


def _register(adapter: SemBlendProviderAdapter, request_id: str, **kwargs) -> None:
    adapter.register_donor(
        request_id=request_id,
        token_ids=list(range(16)),
        kv_cache=list(range(100, 116)),
        cache_start_pos=0,
        cache_end_pos=16,
        prompt_text=PROMPT,
        **kwargs,
    )
    # Single-worker executor: a no-op that completes means the registration
    # queued above (embed, store insert, announce) has landed.
    adapter._register_executor.submit(lambda: None).result()  # noqa: SLF001


def _extra(events: list[dict], index: int = 0) -> dict:
    return events[index]["data"]["namespace"]["extra"]


def test_a_registered_donor_is_announced_at_all() -> None:
    events: list[dict] = []
    _register(_adapter(_pipeline(events)), "donor-a", extra_key=RADIX_KEY, cache_salt=SALT)

    assert len(events) == 1
    assert events[0]["data"]["kind"] == "donor_registered"
    assert events[0]["data"]["donor_id"] == "donor-a"


def test_the_announcement_carries_both_keys() -> None:
    events: list[dict] = []
    _register(_adapter(_pipeline(events)), "donor-a", extra_key=RADIX_KEY, cache_salt=SALT)

    extra = _extra(events)
    assert extra[ISOLATION_EXTRA_FIELD] == isolation_namespace(RADIX_KEY)
    assert extra[TENANT_KEY_EXTRA_FIELD] == tenant_key_for_salt(SALT)
    assert extra[ISOLATION_EXTRA_FIELD] != extra[TENANT_KEY_EXTRA_FIELD]


def test_neither_the_radix_key_nor_the_salt_reaches_the_wire() -> None:
    """Both values are tenant-identifying; only their digests travel."""
    import json

    events: list[dict] = []
    _register(_adapter(_pipeline(events)), "donor-a", extra_key=RADIX_KEY, cache_salt=SALT)

    wire = json.dumps(events[0])
    assert RADIX_KEY not in wire
    assert SALT not in wire


def test_distinct_salts_announce_distinct_tenant_keys() -> None:
    events: list[dict] = []
    adapter = _adapter(_pipeline(events))

    _register(adapter, "donor-a", extra_key=RADIX_KEY, cache_salt=SALT)
    _register(adapter, "donor-b", extra_key=RADIX_KEY, cache_salt="tenant-other")

    assert _extra(events, 0)[TENANT_KEY_EXTRA_FIELD] != _extra(events, 1)[TENANT_KEY_EXTRA_FIELD]


def test_an_unthreaded_salt_publishes_the_sentinel_and_warns_once(caplog) -> None:
    """The SGLang wrapper passes its radix key, which no router can rebuild.

    Until it also threads the raw salt, every donor is announced tenant-less
    — visible in the log rather than silent.
    """
    semantic_events._warned_missing_tenant_key = False  # noqa: SLF001
    events: list[dict] = []
    adapter = _adapter(_pipeline(events))

    with caplog.at_level(logging.WARNING, logger="semblend.semantic_events"):
        _register(adapter, "donor-a", extra_key=RADIX_KEY)
        _register(adapter, "donor-b", extra_key=RADIX_KEY)

    semantic_events._warned_missing_tenant_key = False  # noqa: SLF001
    assert len([r for r in caplog.records if "without a tenant key" in r.message]) == 1
    assert all(_extra(events, i)[TENANT_KEY_EXTRA_FIELD] == NO_TENANT_KEY for i in (0, 1))


def test_an_unsalted_request_publishes_the_sentinel_without_warning(caplog) -> None:
    semantic_events._warned_missing_tenant_key = False  # noqa: SLF001
    events: list[dict] = []

    with caplog.at_level(logging.WARNING, logger="semblend.semantic_events"):
        _register(_adapter(_pipeline(events)), "donor-a", extra_key=RADIX_KEY, cache_salt=None)

    semantic_events._warned_missing_tenant_key = False  # noqa: SLF001
    assert _extra(events)[TENANT_KEY_EXTRA_FIELD] == NO_TENANT_KEY
    assert not [r for r in caplog.records if "without a tenant key" in r.message]


def test_the_donor_is_registered_locally_even_if_announcing_fails() -> None:
    """Publication is best-effort; local reuse must not depend on a broker."""
    events: list[dict] = []
    pipeline = _pipeline(events)

    def _explode(*args, **kwargs):
        raise RuntimeError("broker down")

    pipeline.publish_donor_registered = _explode
    adapter = _adapter(pipeline)

    _register(adapter, "donor-a", extra_key=RADIX_KEY, cache_salt=SALT)

    assert pipeline._donor_store.get_donor("donor-a") is not None  # noqa: SLF001
    assert adapter._stats.register_ok == 1  # noqa: SLF001
    assert events == []

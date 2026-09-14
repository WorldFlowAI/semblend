"""The tenant key a worker publishes beside its engine-local isolation key.

The isolation key is derived from engine-private inputs (model, tokenizer,
block size, dtype, salt, adapter), so a router holding only the request cannot
reproduce it: keyed placement is impossible against it. The tenant key is a
function of the raw ``cache_salt`` alone, which is exactly the value a gateway
that isolates tenants already sets, so the same router can compute it.

Contract: ``tenant key v1``, written down in ``docs/tenant-key-v1.md``. These
tests pin the derivation against the shared vector in
``tests/tenant_key_v1_vector.json`` (identical bytes in every repository that
implements the contract), the event shape, the absent-salt sentinel, and that
none of it moved the isolation key.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from semblend.integration.dynamo import semantic_events
from semblend.integration.dynamo.semantic_events import (
    ISOLATION_EXTRA_FIELD,
    NO_ISOLATION_NAMESPACE,
    NO_TENANT_KEY,
    TENANT_KEY_CONTRACT,
    TENANT_KEY_EXTRA_FIELD,
    CacheNamespace,
    bind_isolation_key,
    bind_tenant_key,
    isolation_namespace,
    tenant_key_for_salt,
)
from semblend.integration.vllm.events import VllmContractEmitter
from semblend_core.donor_store import DonorStore
from semblend_core.pipeline import SemBlendPipeline

VECTOR = json.loads((Path(__file__).resolve().parents[1] / "tenant_key_v1_vector.json").read_text())

DIM = 8
CHUNK_SIZE = 16
DONOR_TEXT = "the quarterly report covers every regional business unit"


class _StubEmbedder:
    dimension = DIM

    def embed(self, text: str):
        return np.ones(DIM, dtype=np.float32) / np.sqrt(DIM)


def _namespace(**extra) -> CacheNamespace:
    return CacheNamespace(
        model="qwen",
        tokenizer="qwen",
        kv_layout="vllm",
        block_size=CHUNK_SIZE,
        extra=dict(extra) or None,
    )


def _pipeline(monkeypatch, emitter) -> SemBlendPipeline:
    monkeypatch.delenv("SEMBLEND_CDC_CHUNKS", raising=False)
    monkeypatch.setenv("SEMBLEND_CHUNK_FAST_PATH", "0")
    store = DonorStore(
        max_entries=16,
        embedding_dim=DIM,
        min_similarity=0.60,
        chunk_size=CHUNK_SIZE,
    )
    pipeline = SemBlendPipeline(
        embedder_type="jaccard",
        donor_store=store,
        chunk_size=CHUNK_SIZE,
        enable_pq_segments=False,
    )
    pipeline._embedder = _StubEmbedder()  # noqa: SLF001
    pipeline._event_emitter = emitter  # noqa: SLF001
    return pipeline


@pytest.fixture(autouse=True)
def _reset_once_per_process_warning():
    """The warning is once per process; each test needs its own first time."""
    semantic_events._warned_missing_tenant_key = False
    yield
    semantic_events._warned_missing_tenant_key = False


# ---------------------------------------------------------------------------
# The derivation
# ---------------------------------------------------------------------------


def test_tenant_key_matches_the_shared_contract_vector() -> None:
    assert VECTOR["contract"] == TENANT_KEY_CONTRACT
    assert tenant_key_for_salt(VECTOR["cache_salt"]) == VECTOR["tenant_key"]
    assert tenant_key_for_salt(None) == VECTOR["no_salt_tenant_key"]
    assert NO_TENANT_KEY == VECTOR["no_salt_tenant_key"]


def test_tenant_key_is_a_function_of_the_salt_alone() -> None:
    """Anything else in the derivation makes the key underivable by a router."""
    for left, right in (
        (_namespace(), _namespace(tenant="acme", template="qa")),
        (_namespace(), CacheNamespace("other", "other", "sglang", 32)),
    ):
        assert (
            bind_tenant_key(left, "tenant-acme").extra[TENANT_KEY_EXTRA_FIELD]
            == bind_tenant_key(right, "tenant-acme").extra[TENANT_KEY_EXTRA_FIELD]
        )


def test_an_empty_salt_is_the_sentinel_and_a_keyed_salt_never_is() -> None:
    assert tenant_key_for_salt("") == NO_TENANT_KEY
    assert tenant_key_for_salt("tenant-acme") != NO_TENANT_KEY


def test_the_raw_salt_never_reaches_the_key() -> None:
    key = tenant_key_for_salt("acme-corp-prod")
    assert "acme-corp-prod" not in key
    assert key.startswith("semblend:tenant:v1:")


def test_different_salts_give_different_keys() -> None:
    assert tenant_key_for_salt("tenant-a") != tenant_key_for_salt("tenant-b")


# ---------------------------------------------------------------------------
# The event shape
# ---------------------------------------------------------------------------


def test_the_emitter_publishes_the_tenant_key_beside_the_isolation_key() -> None:
    events: list[dict] = []
    emitter = VllmContractEmitter(worker_id=3, namespace=_namespace(), sink=events.append)

    emitter.donor_registered(
        "donor-a",
        list(range(64)),
        np.zeros(384, dtype=np.float32),
        extra_key="vllm:9254316461d3e548",
        cache_salt=VECTOR["cache_salt"],
    )

    extra = events[0]["data"]["namespace"]["extra"]
    assert extra[TENANT_KEY_EXTRA_FIELD] == VECTOR["tenant_key"]
    # The engine-local key is still the connector's own namespace string,
    # unchanged in meaning and derivation.
    assert extra[ISOLATION_EXTRA_FIELD] == isolation_namespace("vllm:9254316461d3e548")
    assert extra[ISOLATION_EXTRA_FIELD] != extra[TENANT_KEY_EXTRA_FIELD]


def test_a_request_without_a_salt_publishes_the_sentinel_without_warning(caplog) -> None:
    """No salt is a real answer; only a missing argument is a wiring gap."""
    events: list[dict] = []
    emitter = VllmContractEmitter(worker_id=3, namespace=_namespace(), sink=events.append)

    with caplog.at_level(logging.WARNING, logger="semblend.semantic_events"):
        emitter.donor_registered(
            "donor-a",
            list(range(64)),
            np.zeros(384, dtype=np.float32),
            extra_key="vllm:9254316461d3e548",
            cache_salt=None,
        )

    extra = events[0]["data"]["namespace"]["extra"]
    assert extra[TENANT_KEY_EXTRA_FIELD] == NO_TENANT_KEY
    assert not [r for r in caplog.records if "without a tenant key" in r.message]


def test_an_unthreaded_salt_warns_once_per_process(caplog) -> None:
    events: list[dict] = []
    emitter = VllmContractEmitter(worker_id=3, namespace=_namespace(), sink=events.append)

    with caplog.at_level(logging.WARNING, logger="semblend.semantic_events"):
        for i in range(3):
            emitter.donor_registered(f"donor-{i}", list(range(64)), np.zeros(384, dtype=np.float32))

    warnings = [r for r in caplog.records if "without a tenant key" in r.message]
    assert len(warnings) == 1
    assert all(
        e["data"]["namespace"]["extra"][TENANT_KEY_EXTRA_FIELD] == NO_TENANT_KEY for e in events
    )


def test_routing_hints_and_both_keys_coexist_in_extra() -> None:
    events: list[dict] = []
    emitter = VllmContractEmitter(
        worker_id=3,
        namespace=_namespace(tenant="default", template="default"),
        sink=events.append,
    )

    emitter.donor_registered(
        "donor-a",
        list(range(64)),
        np.zeros(384, dtype=np.float32),
        tenant="wf-commercial",
        template="wf-rag-v1",
        extra_key="vllm:9254316461d3e548",
        cache_salt=VECTOR["cache_salt"],
    )

    assert events[0]["data"]["namespace"]["extra"] == {
        "tenant": "wf-commercial",
        "template": "wf-rag-v1",
        ISOLATION_EXTRA_FIELD: isolation_namespace("vllm:9254316461d3e548"),
        TENANT_KEY_EXTRA_FIELD: VECTOR["tenant_key"],
    }


# ---------------------------------------------------------------------------
# The pipeline threads the salt
# ---------------------------------------------------------------------------


class _RecordingEmitter:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def donor_registered(self, request_id, token_ids, embedding, **kwargs) -> None:
        self.calls.append(dict(kwargs))


def test_register_donor_forwards_the_salt_to_the_emitter(monkeypatch) -> None:
    emitter = _RecordingEmitter()
    pipeline = _pipeline(monkeypatch, emitter)

    pipeline.register_donor(
        "donor-a",
        list(range(1000, 1200)),
        DONOR_TEXT,
        extra_key="vllm:9254316461d3e548",
        cache_salt=VECTOR["cache_salt"],
    )

    assert emitter.calls[0]["cache_salt"] == VECTOR["cache_salt"]
    assert emitter.calls[0]["extra_key"] == "vllm:9254316461d3e548"


def test_register_donor_without_the_argument_forwards_nothing(monkeypatch) -> None:
    """The emitter must be able to tell an unthreaded salt from an absent one."""
    emitter = _RecordingEmitter()
    pipeline = _pipeline(monkeypatch, emitter)

    pipeline.register_donor("donor-a", list(range(1000, 1200)), DONOR_TEXT)

    assert "cache_salt" not in emitter.calls[0]


def test_the_pipeline_end_to_end_publishes_the_contract_vector(monkeypatch) -> None:
    events: list[dict] = []
    pipeline = _pipeline(
        monkeypatch,
        VllmContractEmitter(worker_id=3, namespace=_namespace(), sink=events.append),
    )

    pipeline.register_donor(
        "donor-a",
        list(range(1000, 1200)),
        DONOR_TEXT,
        extra_key="vllm:9254316461d3e548",
        cache_salt=VECTOR["cache_salt"],
    )

    assert events[0]["data"]["namespace"]["extra"][TENANT_KEY_EXTRA_FIELD] == VECTOR["tenant_key"]


# ---------------------------------------------------------------------------
# The isolation key did not move
# ---------------------------------------------------------------------------


def test_the_isolation_namespace_is_unchanged() -> None:
    """The tenant key is additive: nothing about the engine-local key moved."""
    assert isolation_namespace(None) == NO_ISOLATION_NAMESPACE
    assert isolation_namespace("") == NO_ISOLATION_NAMESPACE
    assert isolation_namespace("semblend:ns:salt:abc") == "semblend:ns:salt:abc"
    assert isolation_namespace("tenant-acme") == "semblend:ns:salt:9332cc3fc0ec09d0f04fa8e1e08637b5"


def test_binding_a_tenant_key_leaves_the_isolation_key_alone() -> None:
    isolated = bind_isolation_key(_namespace(tenant="acme"), "vllm:9254316461d3e548")
    both = bind_tenant_key(isolated, VECTOR["cache_salt"])

    assert both.extra[ISOLATION_EXTRA_FIELD] == isolated.extra[ISOLATION_EXTRA_FIELD]
    assert both.extra["tenant"] == "acme"
    assert both.extra[TENANT_KEY_EXTRA_FIELD] == VECTOR["tenant_key"]


def test_binding_is_idempotent() -> None:
    once = bind_tenant_key(_namespace(), VECTOR["cache_salt"])
    assert bind_tenant_key(once, VECTOR["cache_salt"]) is once


# ---------------------------------------------------------------------------
# The direct emitter (the Dynamo path, used without a pipeline)
# ---------------------------------------------------------------------------


def _direct_emitter(**extra) -> semantic_events.SemBlendEventEmitter:
    return semantic_events.SemBlendEventEmitter(
        worker_id=7,
        namespace=_namespace(**extra),
        embedder=_StubEmbedder(),
    )


def test_the_direct_emitter_stamps_both_keys() -> None:
    event = _direct_emitter().donor_registered(
        "donor-a",
        list(range(64)),
        DONOR_TEXT,
        extra_key="vllm:9254316461d3e548",
        cache_salt=VECTOR["cache_salt"],
    )

    extra = event["data"]["namespace"]["extra"]
    assert extra[TENANT_KEY_EXTRA_FIELD] == VECTOR["tenant_key"]
    assert extra[ISOLATION_EXTRA_FIELD] == isolation_namespace("vllm:9254316461d3e548")


def test_the_direct_emitter_warns_once_when_no_salt_was_threaded(caplog) -> None:
    """This path had no salt argument at all, so every donor it published
    was tenant-less and nothing said so."""
    emitter = _direct_emitter()

    with caplog.at_level(logging.WARNING, logger="semblend.semantic_events"):
        events = [
            emitter.donor_registered(f"donor-{i}", list(range(64)), DONOR_TEXT, extra_key="k")
            for i in range(3)
        ]

    assert len([r for r in caplog.records if "without a tenant key" in r.message]) == 1
    assert all(
        e["data"]["namespace"]["extra"][TENANT_KEY_EXTRA_FIELD] == NO_TENANT_KEY for e in events
    )


def test_the_direct_emitter_treats_an_explicit_none_as_a_real_answer(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="semblend.semantic_events"):
        event = _direct_emitter().donor_registered(
            "donor-a", list(range(64)), DONOR_TEXT, extra_key="k", cache_salt=None
        )

    assert event["data"]["namespace"]["extra"][TENANT_KEY_EXTRA_FIELD] == NO_TENANT_KEY
    assert not [r for r in caplog.records if "without a tenant key" in r.message]

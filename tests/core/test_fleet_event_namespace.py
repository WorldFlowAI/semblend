"""Isolation on the fleet event plane.

A donor is isolated inside the engine that holds it by the ``extra_key``
bound to its DonorNode. The DonorRegistered event a worker publishes is a
second copy of that donor, consumed by a fleet router that has no access to
the local store: whatever isolation the event does not carry, the fleet does
not have. These tests pin that the key reaches the event's CacheNamespace,
and that a consumer applying the contract's namespace compatibility rule
rejects a donor from another namespace.

The compatibility rule mirrored here is ``CacheNamespace::hard_compatible_with``
from the Rust consumer (crates/synapse-sem-events): model, tokenizer,
kv_layout, block_size and the whole ``extra`` map must be equal. ``extra`` is
compared as a unit, which is exactly why the isolation key has to live inside
it.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pytest

from semblend.integration.dynamo.semantic_events import (
    ISOLATION_EXTRA_FIELD,
    NO_ISOLATION_NAMESPACE,
    NO_TENANT_KEY,
    TENANT_KEY_EXTRA_FIELD,
    CacheNamespace,
    SemBlendEventEmitter,
)
from semblend.integration.vllm.events import VllmContractEmitter
from semblend_core.donor_store import DonorNode, DonorStore
from semblend_core.pipeline import SemBlendPipeline

DIM = 8
CHUNK_SIZE = 16

# Hashed-salt shaped namespaces, as the engine connectors build them.
NS_A = "semblend:ns:salt:1111111111111111"
NS_B = "semblend:ns:salt:2222222222222222"

DONOR_TEXT = "the quarterly report covers every regional business unit"


class _StubEmbedder:
    """Same unit vector for every text; the tests are about namespaces."""

    dimension = DIM

    def embed(self, text: str):
        return np.ones(DIM, dtype=np.float32) / np.sqrt(DIM)


class _RecordingEmitter:
    """Stands in for the NATS-backed emitter and keeps the wire events."""

    def __init__(self, namespace: CacheNamespace) -> None:
        self.events: list[dict] = []
        self._inner = SemBlendEventEmitter(
            worker_id=7,
            namespace=namespace,
            embedder=_StubEmbedder(),
        )

    def donor_registered(self, request_id, token_ids, embedding, **kwargs) -> None:
        event = self._inner.donor_registered(
            request_id,
            list(token_ids),
            DONOR_TEXT,
            extra_key=kwargs.get("extra_key"),
        )
        if event is not None:
            self.events.append(event)

    def donor_evicted(self, request_id: str) -> None:  # pragma: no cover - unused
        pass


def _namespace(**extra) -> CacheNamespace:
    return CacheNamespace(
        model="qwen",
        tokenizer="qwen",
        kv_layout="vllm",
        block_size=CHUNK_SIZE,
        extra=dict(extra) or None,
    )


def _hard_compatible(left: dict, right: dict) -> bool:
    """Python mirror of the Rust consumer's ``hard_compatible_with``."""
    fields = ("model", "tokenizer", "kv_layout", "block_size")
    if any(left.get(f) != right.get(f) for f in fields):
        return False
    return dict(left.get("extra") or {}) == dict(right.get("extra") or {})


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


def _wire_namespaces(events: list[dict]) -> list[dict]:
    return [event["data"]["namespace"] for event in events]


# ---------------------------------------------------------------------------
# The blocker: register_donor must publish the donor's isolation key
# ---------------------------------------------------------------------------


def test_register_donor_publishes_a_distinct_namespace_per_isolation_key(
    monkeypatch,
) -> None:
    """Two tenants registering through the same worker must not collide.

    Without the key in the event, both donors carry the worker-level
    namespace, the fleet catalog sees one namespace, and a tenant-B request
    can be routed onto tenant-A's donor.
    """
    emitter = _RecordingEmitter(_namespace(tenant="worker-tenant"))
    pipeline = _pipeline(monkeypatch, emitter)

    pipeline.register_donor("donor-a", list(range(1000, 1200)), DONOR_TEXT, extra_key=NS_A)
    pipeline.register_donor("donor-b", list(range(2000, 2200)), DONOR_TEXT, extra_key=NS_B)

    ns_a, ns_b = _wire_namespaces(emitter.events)
    # The wire field name is part of the contract the consumer parses.
    assert ISOLATION_EXTRA_FIELD == "cache_salt"
    assert ns_a["extra"][ISOLATION_EXTRA_FIELD] == NS_A
    assert ns_b["extra"][ISOLATION_EXTRA_FIELD] == NS_B
    assert ns_a != ns_b


def test_fleet_compatibility_check_rejects_donors_from_different_tenants(
    monkeypatch,
) -> None:
    """The consumer's own rule, not just string inequality, must reject them."""
    emitter = _RecordingEmitter(_namespace(tenant="worker-tenant"))
    pipeline = _pipeline(monkeypatch, emitter)

    pipeline.register_donor("donor-a", list(range(1000, 1200)), DONOR_TEXT, extra_key=NS_A)
    pipeline.register_donor("donor-b", list(range(2000, 2200)), DONOR_TEXT, extra_key=NS_B)

    ns_a, ns_b = _wire_namespaces(emitter.events)
    assert not _hard_compatible(ns_a, ns_b)
    assert _hard_compatible(ns_a, ns_a)


def test_same_isolation_key_stays_reusable(monkeypatch) -> None:
    """Isolation must not become blanket over-isolation."""
    emitter = _RecordingEmitter(_namespace(tenant="worker-tenant"))
    pipeline = _pipeline(monkeypatch, emitter)

    pipeline.register_donor("donor-a", list(range(1000, 1200)), DONOR_TEXT, extra_key=NS_A)
    pipeline.register_donor("donor-b", list(range(2000, 2200)), DONOR_TEXT, extra_key=NS_A)

    ns_a, ns_b = _wire_namespaces(emitter.events)
    assert _hard_compatible(ns_a, ns_b)


def test_absent_key_carries_the_sentinel_and_never_matches_a_keyed_donor(
    monkeypatch,
) -> None:
    """Absent must be an explicit value on the wire, not a missing field."""
    emitter = _RecordingEmitter(_namespace(tenant="worker-tenant"))
    pipeline = _pipeline(monkeypatch, emitter)

    pipeline.register_donor("donor-plain", list(range(1000, 1200)), DONOR_TEXT)
    pipeline.register_donor("donor-keyed", list(range(2000, 2200)), DONOR_TEXT, extra_key=NS_A)

    ns_plain, ns_keyed = _wire_namespaces(emitter.events)
    assert ns_plain["extra"][ISOLATION_EXTRA_FIELD] == NO_ISOLATION_NAMESPACE
    assert not _hard_compatible(ns_plain, ns_keyed)


def test_two_unkeyed_donors_remain_mutually_reusable(monkeypatch) -> None:
    """Single-tenant deployments keep working: both land in the sentinel."""
    emitter = _RecordingEmitter(_namespace())
    pipeline = _pipeline(monkeypatch, emitter)

    pipeline.register_donor("donor-a", list(range(1000, 1200)), DONOR_TEXT)
    pipeline.register_donor("donor-b", list(range(2000, 2200)), DONOR_TEXT)

    ns_a, ns_b = _wire_namespaces(emitter.events)
    assert _hard_compatible(ns_a, ns_b)


# ---------------------------------------------------------------------------
# Emitter-level: both contract emitters bind the key themselves
# ---------------------------------------------------------------------------


def test_vllm_emitter_binds_the_isolation_key_alongside_routing_extra() -> None:
    events: list[dict] = []
    emitter = VllmContractEmitter(
        worker_id=3,
        namespace=_namespace(tenant="default-tenant", template="default-template"),
        sink=events.append,
    )
    embedding = np.zeros(384, dtype=np.float32)
    embedding[0] = 1.0

    emitter.donor_registered(
        "donor-a",
        list(range(64)),
        embedding,
        tenant="wf-commercial",
        template="wf-rag-v1",
        extra_key=NS_A,
    )
    emitter.donor_registered("donor-b", list(range(64)), embedding, extra_key=NS_B)

    ns_a, ns_b = _wire_namespaces(events)
    # Per-donor routing hints survive; both keys are bound on top. No
    # cache_salt was threaded through either call, so the tenant key is the
    # sentinel -- the isolation key is what these assertions are about.
    assert ns_a["extra"] == {
        "tenant": "wf-commercial",
        "template": "wf-rag-v1",
        ISOLATION_EXTRA_FIELD: NS_A,
        TENANT_KEY_EXTRA_FIELD: NO_TENANT_KEY,
    }
    assert ns_b["extra"][ISOLATION_EXTRA_FIELD] == NS_B
    assert not _hard_compatible(ns_a, ns_b)


def test_vllm_emitter_stamps_the_sentinel_when_no_key_is_passed() -> None:
    events: list[dict] = []
    emitter = VllmContractEmitter(
        worker_id=3,
        namespace=_namespace(),
        sink=events.append,
    )
    emitter.donor_registered("donor-a", list(range(64)), np.zeros(384, dtype=np.float32))

    assert events[0]["data"]["namespace"]["extra"] == {
        ISOLATION_EXTRA_FIELD: NO_ISOLATION_NAMESPACE,
        TENANT_KEY_EXTRA_FIELD: NO_TENANT_KEY,
    }


def test_dynamo_emitter_binds_the_isolation_key() -> None:
    emitter = SemBlendEventEmitter(
        worker_id=1,
        namespace=_namespace(),
        embedder=_StubEmbedder(),
    )

    keyed = emitter.donor_registered("d1", list(range(32)), DONOR_TEXT, extra_key=NS_A)
    plain = emitter.donor_registered("d2", list(range(32)), DONOR_TEXT)

    assert keyed["data"]["namespace"]["extra"][ISOLATION_EXTRA_FIELD] == NS_A
    assert plain["data"]["namespace"]["extra"][ISOLATION_EXTRA_FIELD] == NO_ISOLATION_NAMESPACE
    assert not _hard_compatible(keyed["data"]["namespace"], plain["data"]["namespace"])


def test_raw_isolation_key_is_hashed_before_it_reaches_the_wire() -> None:
    """A raw key is tenant-identifying; the wire carries a digest instead."""
    events: list[dict] = []
    emitter = VllmContractEmitter(
        worker_id=3,
        namespace=_namespace(),
        sink=events.append,
    )
    emitter.donor_registered(
        "donor-a",
        list(range(64)),
        np.zeros(384, dtype=np.float32),
        extra_key="acme-corp-prod",
    )

    bound = events[0]["data"]["namespace"]["extra"][ISOLATION_EXTRA_FIELD]
    assert "acme-corp-prod" not in bound
    assert bound.startswith("semblend:ns:salt:")


# ---------------------------------------------------------------------------
# Stage-0 gate: the fast-path count is scoped to the requesting namespace
# ---------------------------------------------------------------------------


def test_stage0_fast_path_gate_ignores_donors_from_other_namespaces(
    monkeypatch,
) -> None:
    """A foreign donor must not decide that this request skips the embedding."""
    monkeypatch.delenv("SEMBLEND_CDC_CHUNKS", raising=False)
    monkeypatch.setenv("SEMBLEND_CHUNK_FAST_PATH", "1")
    monkeypatch.setenv("SEMBLEND_FAST_PATH_MIN_HITS", "2")
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

    tokens = list(range(3000, 3000 + CHUNK_SIZE * 6))
    pipeline.register_donor("foreign", tokens, DONOR_TEXT, extra_key=NS_B)

    calls: list[str | None] = []
    real_find_multi_donor = store.find_multi_donor

    def _counting_find_multi_donor(*args, **kwargs):
        calls.append(kwargs.get("extra_key"))
        return real_find_multi_donor(*args, **kwargs)

    store.find_multi_donor = _counting_find_multi_donor  # type: ignore[method-assign]

    pipeline.find_donor(token_ids=tokens, prompt_text=DONOR_TEXT, extra_key=NS_A)

    # Every chunk match came from the foreign donor, so the gate never fires
    # and the composite lookup is not attempted at all.
    assert calls == []


def test_stage0_fast_path_gate_still_fires_for_same_namespace_donors(
    monkeypatch,
) -> None:
    monkeypatch.delenv("SEMBLEND_CDC_CHUNKS", raising=False)
    monkeypatch.setenv("SEMBLEND_CHUNK_FAST_PATH", "1")
    monkeypatch.setenv("SEMBLEND_FAST_PATH_MIN_HITS", "2")
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

    tokens = list(range(3000, 3000 + CHUNK_SIZE * 6))
    pipeline.register_donor("same-tenant", tokens, DONOR_TEXT, extra_key=NS_A)

    calls: list[str | None] = []
    real_find_multi_donor = store.find_multi_donor

    def _counting_find_multi_donor(*args, **kwargs):
        calls.append(kwargs.get("extra_key"))
        return real_find_multi_donor(*args, **kwargs)

    store.find_multi_donor = _counting_find_multi_donor  # type: ignore[method-assign]

    pipeline.find_donor(token_ids=tokens, prompt_text=DONOR_TEXT, extra_key=NS_A)

    assert calls and calls[0] == NS_A


# ---------------------------------------------------------------------------
# Donor store: the fail-open lookup says so
# ---------------------------------------------------------------------------


def _node(request_id: str, extra_key: str | None) -> DonorNode:
    return DonorNode(
        request_id=request_id,
        token_ids=list(range(CHUNK_SIZE * 4)),
        embedding=np.ones(DIM, dtype=np.float32) / np.sqrt(DIM),
        timestamp=time.monotonic(),
        extra_key=extra_key,
    )


def test_unnamespaced_lookup_over_namespaced_donors_warns_once_and_counts(
    caplog,
) -> None:
    store = DonorStore(max_entries=16, embedding_dim=DIM, chunk_size=CHUNK_SIZE)
    store.add_donor(_node("keyed", NS_A))
    query = np.ones(DIM, dtype=np.float32) / np.sqrt(DIM)

    with caplog.at_level(logging.WARNING, logger="semblend_core.donor_store"):
        store.find_donor(query, list(range(CHUNK_SIZE * 4)))
        store.find_donor(query, list(range(CHUNK_SIZE * 4)))
        store.find_candidates_jaccard(list(range(CHUNK_SIZE * 4)))

    warnings = [r for r in caplog.records if "unnamespaced lookup" in r.message]
    assert len(warnings) == 1
    assert store.unnamespaced_lookups == 3


def test_unnamespaced_lookup_is_silent_without_namespaced_donors(caplog) -> None:
    """Single-tenant stores must not be nagged about a key they never set."""
    store = DonorStore(max_entries=16, embedding_dim=DIM, chunk_size=CHUNK_SIZE)
    store.add_donor(_node("plain", None))
    query = np.ones(DIM, dtype=np.float32) / np.sqrt(DIM)

    with caplog.at_level(logging.WARNING, logger="semblend_core.donor_store"):
        store.find_donor(query, list(range(CHUNK_SIZE * 4)))
        store.find_candidates_jaccard(list(range(CHUNK_SIZE * 4)))

    assert [r for r in caplog.records if "unnamespaced lookup" in r.message] == []
    assert store.unnamespaced_lookups == 0


def test_keyed_lookup_is_not_counted_as_unnamespaced() -> None:
    store = DonorStore(max_entries=16, embedding_dim=DIM, chunk_size=CHUNK_SIZE)
    store.add_donor(_node("keyed", NS_A))
    query = np.ones(DIM, dtype=np.float32) / np.sqrt(DIM)

    store.find_donor(query, list(range(CHUNK_SIZE * 4)), extra_key=NS_A)
    store.find_candidates_jaccard(list(range(CHUNK_SIZE * 4)), extra_key=NS_A)

    assert store.unnamespaced_lookups == 0


def test_namespaced_donor_count_follows_eviction_and_clear() -> None:
    """The warning gate must not latch on donors the store no longer holds."""
    store = DonorStore(max_entries=2, embedding_dim=DIM, chunk_size=CHUNK_SIZE)
    store.add_donor(_node("keyed", NS_A))
    store.add_donor(_node("plain", None))
    store.add_donor(_node("plain-2", None))  # LRU-evicts the only keyed donor
    query = np.ones(DIM, dtype=np.float32) / np.sqrt(DIM)

    store.find_donor(query, list(range(CHUNK_SIZE * 4)))
    assert store.unnamespaced_lookups == 0

    store.add_donor(_node("keyed-2", NS_B))
    store.clear()
    store.find_donor(query, list(range(CHUNK_SIZE * 4)))
    assert store.unnamespaced_lookups == 0


@pytest.mark.parametrize("key", ["", "   "])
def test_blank_isolation_key_is_treated_as_absent(key: str) -> None:
    """A blank key carries no intent; it must not become a namespace of its own."""
    events: list[dict] = []
    emitter = VllmContractEmitter(
        worker_id=3,
        namespace=_namespace(),
        sink=events.append,
    )
    emitter.donor_registered(
        "donor-a",
        list(range(64)),
        np.zeros(384, dtype=np.float32),
        extra_key=key,
    )

    assert events[0]["data"]["namespace"]["extra"][ISOLATION_EXTRA_FIELD] == NO_ISOLATION_NAMESPACE

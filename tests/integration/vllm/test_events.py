from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

from semblend.integration.dynamo.nats_publisher import (
    DEFAULT_SUBJECT as NATS_DEFAULT_SUBJECT,
)
from semblend.integration.dynamo.semantic_events import (
    ISOLATION_EXTRA_FIELD,
    NO_ISOLATION_NAMESPACE,
    SEMANTIC_KV_EVENT_SUBJECT,
    CacheNamespace,
)
from semblend.integration.trtllm.events import DEFAULT_SUBJECT as TRTLLM_DEFAULT_SUBJECT
from semblend.integration.vllm.events import VllmContractEmitter, _worker_id_from_env


def test_vllm_donor_registered_carries_per_donor_routing_extra() -> None:
    events: list[dict] = []
    emitter = VllmContractEmitter(
        worker_id=3,
        namespace=CacheNamespace(
            model="qwen",
            tokenizer="qwen",
            kv_layout="vllm",
            block_size=16,
            extra={"tenant": "default-tenant", "template": "default-template"},
        ),
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
    )

    assert len(events) == 1
    data = events[0]["data"]
    assert data["kind"] == "donor_registered"
    assert data["namespace"]["extra"] == {
        "tenant": "wf-commercial",
        "template": "wf-rag-v1",
        # No extra_key on this call, so the isolation field carries the
        # sentinel: absent is an explicit value on the wire, never omitted.
        # A consumer comparing the whole extra map must see "unsalted" and
        # "salted" as different namespaces, which an omitted field cannot do.
        ISOLATION_EXTRA_FIELD: NO_ISOLATION_NAMESPACE,
    }
    assert len(data["segments"][0]["provider_metadata"]) == 384 * 4


def test_every_publisher_defaults_to_the_contract_subject() -> None:
    """One subject name across the package.

    A publisher whose default subject differs from the consumer's fails
    silently: NATS delivers nothing, no donor is ever seen, and nothing
    raises. These constants must be the same string.
    """
    assert SEMANTIC_KV_EVENT_SUBJECT == "semantic-kv-events"
    assert NATS_DEFAULT_SUBJECT == SEMANTIC_KV_EVENT_SUBJECT
    assert TRTLLM_DEFAULT_SUBJECT == SEMANTIC_KV_EVENT_SUBJECT


def test_worker_id_ordinal_comes_from_the_pod_name(monkeypatch) -> None:
    monkeypatch.delenv("SEMBLEND_WORKER_ID", raising=False)
    monkeypatch.setenv("HOSTNAME", "vllm-backend-2")
    assert _worker_id_from_env(None) == 2


def test_worker_id_is_stable_across_processes(monkeypatch) -> None:
    """The hostname fallback must not depend on the process hash seed.

    CPython salts str hashing per process, so a hash()-derived id differs
    between the scheduler process, each worker process and every restart —
    one worker would look like several to a consumer that keys donors by
    worker id. Two children started with different hash seeds must agree
    with each other and with this process.
    """
    monkeypatch.delenv("SEMBLEND_WORKER_ID", raising=False)
    monkeypatch.setenv("HOSTNAME", "vllm-backend-no-ordinal")

    def worker_id_with_hash_seed(hash_seed: str) -> int:
        out = subprocess.run(
            [
                sys.executable,
                "-c",
                "from semblend.integration.vllm.events import _worker_id_from_env;"
                "print(_worker_id_from_env(None))",
            ],
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            capture_output=True,
            text=True,
            check=True,
        )
        return int(out.stdout.strip())

    assert (
        worker_id_with_hash_seed("1") == worker_id_with_hash_seed("2") == _worker_id_from_env(None)
    )


def test_explicit_worker_id_wins(monkeypatch) -> None:
    monkeypatch.setenv("SEMBLEND_WORKER_ID", "11")
    monkeypatch.setenv("HOSTNAME", "vllm-backend-2")
    assert _worker_id_from_env(7) == 7
    assert _worker_id_from_env(None) == 11

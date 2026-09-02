from __future__ import annotations

import numpy as np

from semblend.integration.dynamo.semantic_events import CacheNamespace
from semblend.integration.vllm.events import VllmContractEmitter


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
    }
    assert len(data["segments"][0]["provider_metadata"]) == 384 * 4

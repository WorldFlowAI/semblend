"""Tests for the SemBlend SGLang adapter emitting Dynamo semantic KV events.
Uses a stub pipeline so no MiniLM/SGLang load.

register_donor, when events are configured, emits a DonorRegistered with the
embedding in provider_metadata; LRU eviction emits DonorEvicted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from semblend.integration.dynamo.semantic_events import CacheNamespace, decode_embedding
from semblend.integration.sglang.config import SemBlendProviderConfig
from semblend.integration.sglang.provider import SemBlendProviderAdapter


@dataclass
class _StubEmbedder:
    dim: int = 384

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        vec[len(text) % self.dim] = 1.0
        return vec


@dataclass
class _StubDonorStore:
    donors: list = field(default_factory=list)

    def add_donor(self, node: Any) -> None:
        self.donors.append(node)

    def clear(self) -> None:
        self.donors.clear()


class _StubPipeline:
    def __init__(self) -> None:
        self._embedder = _StubEmbedder()
        self._donor_store = _StubDonorStore()


def _config(max_entries: int) -> SemBlendProviderConfig:
    return SemBlendProviderConfig(
        min_similarity=0.6,
        min_reuse_ratio=0.5,
        min_match_length=4,
        max_entries=max_entries,
        block_size=4,
        model_arch="llama",
    )


def _namespace() -> CacheNamespace:
    return CacheNamespace(
        model="llama",
        tokenizer="llama",
        kv_layout="flashinfer-fp16",
        block_size=4,
    )


def _adapter(max_entries: int):
    adapter = SemBlendProviderAdapter(config=_config(max_entries), pipeline=_StubPipeline())
    events: list[dict] = []
    adapter.configure_events(worker_id=7, namespace=_namespace(), sink=events.append)
    return adapter, events


def test_configure_events_emits_generation_reset_at_head() -> None:
    _adapter_obj, events = _adapter(100)
    assert len(events) == 1
    assert events[0]["data"]["kind"] == "provider_generation_reset"
    assert events[0]["worker_id"] == 7


def test_register_donor_emits_contract_event_with_embedding() -> None:
    adapter, events = _adapter(100)
    ok = adapter.register_donor(
        request_id="req-abc",
        token_ids=list(range(16)),
        kv_cache=list(range(100, 116)),
        cache_start_pos=0,
        cache_end_pos=16,
        prompt_text="how do i rotate an api key",
    )
    assert ok is True
    adapter._register_executor.shutdown(wait=True)

    regs = [e for e in events if e["data"]["kind"] == "donor_registered"]
    assert len(regs) == 1
    data = regs[0]["data"]
    assert data["donor_id"] == "req-abc"
    assert data["namespace"]["model"] == "llama"
    assert data["location"] == {
        "kind": "worker",
        "worker_id": 7,
        "dp_rank": 0,
        "tier": "device",
    }
    # Embedding rode provider_metadata as 384 LE-f32 -> 1536 bytes; decodes back.
    pm = data["segments"][0]["provider_metadata"]
    assert len(pm) == 384 * 4
    vec = decode_embedding(pm)
    assert len(vec) == 384


def test_lru_eviction_emits_donor_evicted() -> None:
    # max_entries=1: registering a second donor evicts the first.
    adapter, events = _adapter(1)
    adapter.register_donor(
        request_id="donor-A",
        token_ids=list(range(8)),
        kv_cache=list(range(8)),
        cache_start_pos=0,
        cache_end_pos=8,
        prompt_text="first donor",
    )
    adapter.register_donor(
        request_id="donor-B",
        token_ids=list(range(8)),
        kv_cache=list(range(8)),
        cache_start_pos=0,
        cache_end_pos=8,
        prompt_text="second donor",
    )
    adapter._register_executor.shutdown(wait=True)

    evicts = [
        e["data"]["donor_id"]
        for e in events
        if e["data"]["kind"] == "donor_evicted"
    ]
    assert "donor-A" in evicts


def test_events_disabled_by_default_no_emission() -> None:
    # No configure_events() -> no behavior change, register still works.
    adapter = SemBlendProviderAdapter(config=_config(100), pipeline=_StubPipeline())
    ok = adapter.register_donor(
        request_id="req-x",
        token_ids=list(range(8)),
        kv_cache=list(range(8)),
        cache_start_pos=0,
        cache_end_pos=8,
        prompt_text="no events configured",
    )
    assert ok is True
    adapter._register_executor.shutdown(wait=True)  # must not raise

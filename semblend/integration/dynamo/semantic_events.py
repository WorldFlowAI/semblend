"""Build Dynamo semantic KV donor lifecycle events from SemBlend.

SemBlend sees a donor's tokens and embedding, so it builds the ``SemanticKvEvent``
dicts for the Dynamo semantic KV event contract and publishes them on the
``semantic-kv-events`` subject.

The embedding is not a typed field; it rides the per-segment opaque
``provider_metadata`` as little-endian f32 bytes (a JSON array of u8). Consumers
decode it with the same convention (:func:`decode_embedding`). ``donor_id`` is
the join key.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

SEMANTIC_KV_EVENT_SUBJECT = "semantic-kv-events"
SEMANTIC_KV_EVENT_SCHEMA_VERSION = 1


def encode_embedding(embedding: Sequence[float]) -> list[int]:
    """Pack an f32 embedding as little-endian bytes, as a JSON-friendly u8 list."""
    arr = np.asarray(embedding, dtype="<f4")  # little-endian f32
    return list(arr.tobytes())


def decode_embedding(byte_list: Sequence[int]) -> list[float]:
    """Inverse of :func:`encode_embedding`: LE-f32 byte list -> float vector."""
    return list(np.frombuffer(bytes(byte_list), dtype="<f4"))


def sequence_digest(token_ids: Sequence[int]) -> int:
    """Non-reversible u64 integrity digest over token ids (not a semantic key)."""
    payload = struct.pack(f"<{len(token_ids)}q", *[int(t) for t in token_ids])
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


@dataclass(frozen=True)
class CacheNamespace:
    model: str
    tokenizer: str
    kv_layout: str
    block_size: int
    lora: Optional[str] = None
    quant: Optional[str] = None
    # Open map for provider-defined routing keys (tenant, template, SLA, ...),
    # per the semantic-KV contract. Consumers gate donor reuse on these
    # (e.g. the Synapse scorer reads extra.tenant / extra.template).
    extra: Optional[dict] = None

    def to_wire(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "kv_layout": self.kv_layout,
            "block_size": self.block_size,
        }
        # Mirror serde skip_serializing_if=Option::is_none / empty map.
        if self.lora is not None:
            d["lora"] = self.lora
        if self.quant is not None:
            d["quant"] = self.quant
        if self.extra:
            d["extra"] = dict(self.extra)
        return d


def worker_location(worker_id: int, dp_rank: int = 0, tier: str = "device") -> dict[str, Any]:
    return {"kind": "worker", "worker_id": worker_id, "dp_rank": dp_rank, "tier": tier}


def donor_segment(
    segment_id: int,
    token_start: int,
    token_end: int,
    token_ids: Sequence[int],
    embedding: Optional[Sequence[float]],
) -> dict[str, Any]:
    seg: dict[str, Any] = {
        "segment_id": segment_id,
        "token_range": [token_start, token_end],
        "digest": sequence_digest(token_ids),
    }
    if embedding is not None:
        seg["provider_metadata"] = encode_embedding(embedding)
    return seg


def donor_registered_event(
    *,
    event_id: int,
    worker_id: int,
    donor_id: str,
    namespace: CacheNamespace,
    token_ids: Sequence[int],
    embedding: Sequence[float],
    dp_rank: int = 0,
    provider_generation: int = 0,
) -> dict[str, Any]:
    """Build a whole-sequence ``DonorRegistered`` event with the pooled embedding
    attached to the single segment's ``provider_metadata``."""
    token_count = len(token_ids)
    seg = donor_segment(0, 0, token_count, token_ids, embedding)
    return {
        "schema_version": SEMANTIC_KV_EVENT_SCHEMA_VERSION,
        "event_id": event_id,
        "worker_id": worker_id,
        "dp_rank": dp_rank,
        "data": {
            "kind": "donor_registered",
            "donor_id": donor_id,
            "namespace": namespace.to_wire(),
            "location": worker_location(worker_id, dp_rank),
            "token_count": token_count,
            "segments": [seg],
            "provider_generation": provider_generation,
        },
    }


def donor_evicted_event(
    *,
    event_id: int,
    worker_id: int,
    donor_id: str,
    dp_rank: int = 0,
    segments: Optional[Sequence[int]] = None,
    provider_generation: int = 0,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "kind": "donor_evicted",
        "donor_id": donor_id,
        "provider_generation": provider_generation,
    }
    if segments is not None:
        data["segments"] = list(segments)
    return {
        "schema_version": SEMANTIC_KV_EVENT_SCHEMA_VERSION,
        "event_id": event_id,
        "worker_id": worker_id,
        "dp_rank": dp_rank,
        "data": data,
    }


def generation_reset_event(
    *, event_id: int, worker_id: int, generation: int, dp_rank: int = 0
) -> dict[str, Any]:
    return {
        "schema_version": SEMANTIC_KV_EVENT_SCHEMA_VERSION,
        "event_id": event_id,
        "worker_id": worker_id,
        "dp_rank": dp_rank,
        "data": {"kind": "provider_generation_reset", "generation": generation},
    }


class SemBlendEventEmitter:
    """Builds and (optionally) publishes contract events for completed donors.

    The engine-local adapter calls :meth:`donor_registered` when a request
    becomes a donor; this computes the embedding with SemBlend's real embedder
    and emits the enriched contract event. ``donor_id`` MUST be the same id the
    rest of the stack uses (the engine request id).
    """

    def __init__(
        self,
        worker_id: int,
        namespace: CacheNamespace,
        embedder: Any,
        *,
        dp_rank: int = 0,
        provider_generation: int = 0,
    ) -> None:
        self._worker_id = worker_id
        self._dp_rank = dp_rank
        self._namespace = namespace
        self._embedder = embedder
        self._generation = provider_generation
        self._next_event_id = 0

    def _next_id(self) -> int:
        eid = self._next_event_id
        self._next_event_id += 1
        return eid

    def reset_head(self) -> dict[str, Any]:
        """The generation-reset event to send at stream head on (re)start."""
        return generation_reset_event(
            event_id=self._next_id(),
            worker_id=self._worker_id,
            generation=self._generation,
            dp_rank=self._dp_rank,
        )

    def donor_registered(
        self, donor_id: str, token_ids: Sequence[int], prompt_text: str
    ) -> Optional[dict[str, Any]]:
        """Build a DonorRegistered event with the donor's embedding, or None if
        the embedder is unavailable (fail closed: no embedding, no event)."""
        raw = self._embedder.embed(prompt_text)
        if raw is None:
            return None
        embedding = np.asarray(raw, dtype=np.float32).tolist()
        return donor_registered_event(
            event_id=self._next_id(),
            worker_id=self._worker_id,
            donor_id=donor_id,
            namespace=self._namespace,
            token_ids=token_ids,
            embedding=embedding,
            dp_rank=self._dp_rank,
            provider_generation=self._generation,
        )

    def donor_evicted(
        self, donor_id: str, segments: Optional[Sequence[int]] = None
    ) -> dict[str, Any]:
        return donor_evicted_event(
            event_id=self._next_id(),
            worker_id=self._worker_id,
            donor_id=donor_id,
            dp_rank=self._dp_rank,
            segments=segments,
            provider_generation=self._generation,
        )


async def publish(nats_client: Any, event: dict[str, Any]) -> None:
    """Publish one contract event on the semantic-kv-events subject. nats-py is
    imported lazily by the caller; this just serializes and sends."""
    await nats_client.publish(
        SEMANTIC_KV_EVENT_SUBJECT, json.dumps(event).encode("utf-8")
    )

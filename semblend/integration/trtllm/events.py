"""TensorRT-LLM SemBlend semantic-KV contract emitter.

This publishes the same donor lifecycle contract that Synapse consumes for
vLLM/SGLang placement. It is intentionally fail-safe: if NATS or embedding is
unavailable, inference and local TensorRT-LLM validation continue normally.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import struct
import threading
import time
from typing import Any, Sequence

from semblend.integration.trtllm.upstream_interface import (
    CacheNamespace,
    DonorRegistered,
    SemanticKvEvent,
)

logger = logging.getLogger("semblend.trtllm.events")

DEFAULT_SUBJECT = "semantic-kv-events"


class TrtllmContractEmitter:
    """Build and publish Synapse semantic-KV contract events for TRT-LLM."""

    def __init__(
        self,
        *,
        worker_id: int,
        namespace: CacheNamespace,
        sink,
        embedder_type: str = "minilm",
        dp_rank: int = 0,
        provider_generation: int = 0,
    ) -> None:
        self._worker_id = worker_id
        self._namespace = namespace
        self._sink = sink
        self._embedder_type = embedder_type
        self._dp_rank = dp_rank
        self._generation = provider_generation
        self._event_id = 0
        self._embedder = None

    @classmethod
    def from_env(
        cls,
        *,
        namespace: CacheNamespace,
        embedder_type: str = "minilm",
        worker_id: int | None = None,
        dp_rank: int = 0,
        provider_generation: int | None = None,
    ) -> "TrtllmContractEmitter | None":
        if not os.environ.get("SEMBLEND_NATS_URL"):
            return None
        publisher = _ThreadedNatsPublisher.from_env()
        if publisher is None:
            return None
        publisher.wait_ready(10.0)
        return cls(
            worker_id=_worker_id_from_env(worker_id),
            namespace=namespace,
            sink=publisher.publish,
            embedder_type=embedder_type,
            dp_rank=dp_rank,
            provider_generation=(
                int(provider_generation)
                if provider_generation is not None
                else int(time.time() * 1000)
            ),
        )

    def generation_reset(self) -> None:
        self._publish(
            {
                "schema_version": 1,
                "event_id": self._next_id(),
                "worker_id": self._worker_id,
                "dp_rank": self._dp_rank,
                "data": {
                    "kind": "provider_generation_reset",
                    "generation": self._generation,
                },
            }
        )

    def donor_registered(
        self,
        event: DonorRegistered,
        *,
        prompt_text: str,
        token_ids: Sequence[int],
    ) -> bool:
        embedding = self._embed(prompt_text)
        if embedding is None:
            return False

        wire = SemanticKvEvent(
            schema_version=1,
            event_id=self._next_id(),
            worker_id=self._worker_id,
            dp_rank=self._dp_rank,
            data=event,
        ).to_dict()
        wire["worker_id"] = self._worker_id
        wire["dp_rank"] = self._dp_rank
        data = wire.get("data", {})
        data["location"] = {
            "kind": "worker",
            "worker_id": self._worker_id,
            "dp_rank": self._dp_rank,
            "tier": "device",
        }
        segments = data.get("segments") or []
        if segments:
            segments[0]["provider_metadata"] = _encode_embedding(embedding)
            segments[0]["digest"] = _sequence_digest(token_ids)
        self._publish(wire)
        return True

    def _embed(self, prompt_text: str) -> list[float] | None:
        if not prompt_text:
            return None
        try:
            if self._embedder is None:
                from semblend_core.embedder import create_embedder

                self._embedder = create_embedder(self._embedder_type)
            raw = self._embedder.embed(_order_invariant_text(prompt_text))
            if raw is None:
                return None
            return [float(value) for value in raw]
        except Exception:
            logger.debug("TRT-LLM contract embedding failed", exc_info=True)
            return None

    def _next_id(self) -> int:
        self._event_id += 1
        return self._event_id

    def _publish(self, event: dict[str, Any]) -> None:
        try:
            self._sink(event)
        except Exception:
            logger.debug("TRT-LLM contract event publish failed", exc_info=True)


class _ThreadedNatsPublisher:
    def __init__(self, url: str, subject: str) -> None:
        self._url = url
        self._subject = subject
        self._nc: Any | None = None
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="semblend-trtllm-nats-publisher",
            daemon=True,
        )
        self._thread.start()

    @classmethod
    def from_env(cls) -> "_ThreadedNatsPublisher | None":
        url = os.environ.get("SEMBLEND_NATS_URL")
        if not url:
            return None
        subject = os.environ.get("SEMBLEND_NATS_SUBJECT", DEFAULT_SUBJECT)
        try:
            return cls(url, subject)
        except Exception:
            logger.warning("TRT-LLM NATS publisher init failed", exc_info=True)
            return None

    def wait_ready(self, timeout: float = 10.0) -> bool:
        self._ready.wait(timeout)
        return self._nc is not None

    def publish(self, event: dict[str, Any]) -> None:
        if self._nc is None:
            return
        try:
            payload = json.dumps(event).encode("utf-8")
            asyncio.run_coroutine_threadsafe(
                self._nc.publish(self._subject, payload),
                self._loop,
            )
        except Exception:
            logger.debug("TRT-LLM NATS publish enqueue failed", exc_info=True)

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connect())
        except Exception:
            logger.warning("TRT-LLM NATS connect failed; event emission disabled", exc_info=True)
            self._ready.set()
            return
        self._ready.set()
        self._loop.run_forever()

    async def _connect(self) -> None:
        import nats

        self._nc = await nats.connect(self._url)


def _worker_id_from_env(explicit: int | None = None) -> int:
    if explicit is not None:
        return int(explicit)
    raw = os.environ.get("SEMBLEND_WORKER_ID")
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass
    hostname = os.environ.get("HOSTNAME", "")
    match = re.search(r"-(\d+)$", hostname)
    if match:
        return int(match.group(1))
    return 0


def _order_invariant_text(text: str) -> str:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    if len(sentences) <= 1:
        return text
    return " ".join(sorted(sentences))


def _encode_embedding(embedding: Sequence[float]) -> list[int]:
    payload = struct.pack(f"<{len(embedding)}f", *[float(value) for value in embedding])
    return list(payload)


def _sequence_digest(token_ids: Sequence[int]) -> int:
    import hashlib

    payload = ",".join(str(int(token)) for token in token_ids).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")

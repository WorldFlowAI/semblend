"""vLLM SemBlend → semantic-KV contract emitter.

When a vLLM+SemBlend worker finalizes a donor (a finished request whose KV
becomes reusable), it emits the cross-engine semantic-KV contract events
(DonorRegistered / DonorEvicted / generation-reset) on the
``semantic-kv-events`` subject. A fleet semantic router — e.g. the Synapse
llm-d EPP scorer — consumes them to place affine requests on the worker that
holds a semantically similar donor.

This is the vLLM analogue of the SGLang emit path
(``semblend.integration.sglang.provider``): it reuses the same contract
builders (``semblend.integration.dynamo.semantic_events``) and the same
threaded NATS publisher, so every engine speaks one wire format.

Identical-embedder rule: the embedding passed to :meth:`donor_registered` MUST
be the embedding the engine stored for that donor (the same model the fleet
router uses to embed the query). The pipeline reuses its computed donor
embedding, so register-side and query-side vectors are comparable.

Enabled only when ``SEMBLEND_NATS_URL`` is set; all publishing is fail-safe and
never raises into the inference hot path.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import replace
from typing import Any, Callable, Optional, Sequence

logger = logging.getLogger("semblend.vllm.events")


def _worker_id_from_env(explicit: Optional[int]) -> int:
    """Derive a stable integer worker id: an explicit value, else
    ``SEMBLEND_WORKER_ID``, else a trailing ordinal in HOSTNAME (a StatefulSet
    pod ``vllm-backend-2`` -> 2), else a stable hash of the hostname."""
    if explicit is not None:
        return int(explicit)
    env = os.environ.get("SEMBLEND_WORKER_ID")
    if env and env.isdigit():
        return int(env)
    host = os.environ.get("HOSTNAME", "")
    match = re.search(r"(\d+)$", host)
    if match:
        return int(match.group(1))
    return abs(hash(host)) % 100000


class VllmContractEmitter:
    """Builds and publishes semantic-KV contract events for vLLM donors.

    Holds its own worker_id / namespace / event-id sequence and calls the
    functional contract builders directly, so it can reuse the donor embedding
    the pipeline already computed (rather than re-embedding).
    """

    def __init__(
        self,
        *,
        worker_id: int,
        namespace: Any,
        sink: Callable[[dict], None],
        dp_rank: int = 0,
        provider_generation: int = 0,
    ) -> None:
        self._worker_id = worker_id
        self._namespace = namespace
        self._base_extra = dict(getattr(namespace, "extra", None) or {})
        self._sink = sink
        self._dp_rank = dp_rank
        self._generation = provider_generation
        self._next_event_id = 0

    def _next_id(self) -> int:
        eid = self._next_event_id
        self._next_event_id += 1
        return eid

    @classmethod
    def from_env(
        cls,
        *,
        model: str,
        tokenizer: str,
        block_size: int,
        kv_layout: str = "vllm",
        worker_id: Optional[int] = None,
        dp_rank: int = 0,
        provider_generation: int = 0,
        tenant: Optional[str] = None,
        template: Optional[str] = None,
    ) -> Optional["VllmContractEmitter"]:
        """Build from ``SEMBLEND_NATS_URL`` / ``SEMBLEND_NATS_SUBJECT``; return
        ``None`` if NATS is unset or the toolkit is unavailable. On success,
        emits the generation-reset head event (purges this worker's stale
        donors on the consumer after a restart).

        Worker-level ``tenant`` / ``template`` / ``endpoint`` (args or
        ``SEMBLEND_DONOR_TENANT`` / ``SEMBLEND_DONOR_TEMPLATE`` /
        ``SEMBLEND_ENDPOINT_ID``) seed the namespace ``extra`` so the consumer
        can gate reuse and map the donor to a concrete endpoint;
        :meth:`donor_registered` can override tenant/template per donor."""
        if not os.environ.get("SEMBLEND_NATS_URL"):
            return None
        try:
            from semblend.integration.dynamo.nats_publisher import ThreadedNatsPublisher
            from semblend.integration.dynamo.semantic_events import (
                CacheNamespace,
                generation_reset_event,
            )
        except Exception:
            logger.warning("contract toolkit import failed; emit disabled", exc_info=True)
            return None

        publisher = ThreadedNatsPublisher.from_env()
        if publisher is None:
            return None
        publisher.wait_ready(10.0)

        wid = _worker_id_from_env(worker_id)
        extra: dict = {}
        tenant = tenant or os.environ.get("SEMBLEND_DONOR_TENANT")
        template = template or os.environ.get("SEMBLEND_DONOR_TEMPLATE")
        endpoint = os.environ.get("SEMBLEND_ENDPOINT_ID") or os.environ.get("POD_NAME")
        if tenant:
            extra["tenant"] = tenant
        if template:
            extra["template"] = template
        if endpoint:
            extra["endpoint"] = endpoint
        namespace = CacheNamespace(
            model=model or "unknown",
            tokenizer=tokenizer or model or "unknown",
            kv_layout=kv_layout,
            block_size=int(block_size),
            extra=extra or None,
        )
        emitter = cls(
            worker_id=wid,
            namespace=namespace,
            sink=publisher.publish,
            dp_rank=dp_rank,
            provider_generation=provider_generation,
        )
        try:
            emitter._sink(
                generation_reset_event(
                    event_id=emitter._next_id(),
                    worker_id=wid,
                    generation=provider_generation,
                    dp_rank=dp_rank,
                )
            )
            logger.info("vLLM contract emit enabled: worker_id=%d model=%s", wid, model)
        except Exception:
            logger.debug("generation-reset publish failed", exc_info=True)
        return emitter

    def donor_registered(
        self,
        donor_id: str,
        token_ids: Sequence[int],
        embedding: Any,
        *,
        tenant: Optional[str] = None,
        template: Optional[str] = None,
    ) -> None:
        """Emit DonorRegistered, reusing the donor's stored embedding. Per-donor
        ``tenant`` / ``template`` override the worker-level namespace extra so
        the consumer can gate reuse to the right tenant/template."""
        try:
            from semblend.integration.dynamo.semantic_events import donor_registered_event

            ns = self._namespace
            if tenant is not None or template is not None:
                extra = dict(self._base_extra)
                if tenant is not None:
                    extra["tenant"] = tenant
                if template is not None:
                    extra["template"] = template
                ns = replace(self._namespace, extra=extra or None)

            emb = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
            self._sink(
                donor_registered_event(
                    event_id=self._next_id(),
                    worker_id=self._worker_id,
                    donor_id=donor_id,
                    namespace=ns,
                    token_ids=list(token_ids),
                    embedding=emb,
                    dp_rank=self._dp_rank,
                    provider_generation=self._generation,
                )
            )
        except Exception:
            logger.debug("donor_registered emit failed id=%s", donor_id, exc_info=True)

    def donor_evicted(self, donor_id: str) -> None:
        """Emit DonorEvicted when a donor leaves the store (e.g. LRU)."""
        try:
            from semblend.integration.dynamo.semantic_events import donor_evicted_event

            self._sink(
                donor_evicted_event(
                    event_id=self._next_id(),
                    worker_id=self._worker_id,
                    donor_id=donor_id,
                    dp_rank=self._dp_rank,
                    provider_generation=self._generation,
                )
            )
        except Exception:
            logger.debug("donor_evicted emit failed id=%s", donor_id, exc_info=True)

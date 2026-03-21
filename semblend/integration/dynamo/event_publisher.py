"""Dynamo KV event publisher wrapper — emits semantic events alongside block events.

Wraps Dynamo's KvEventPublisher to publish semantic donor information
to a parallel NATS subject. The SemRouter (or any subscriber) receives
both block-level events (for RadixTree prefix matching) and semantic-level
events (for SemBlend donor search).

Event schema for semantic events (published to semblend.semantic.events):
{
    "type": "stored" | "removed",
    "worker_id": "<worker pod name>",
    "tenant_id": "<tenant>",
    "donor_id": "<request UUID>",
    "embedding": [f32; 384],  // MiniLM L2-normalized
    "num_tokens": <int>,
    "token_ids_hash": "<xxhash of first 256 tokens>"
}

Requires: dynamo Python bindings (pip install nvidia-dynamo)
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger("semblend.dynamo.publisher")


class SemBlendEventPublisher:
    """Wraps Dynamo's KvEventPublisher to add semantic event publishing.

    Intercepts publish_stored() calls, computes semantic embeddings for
    the stored blocks' token content, and publishes both the original
    block event (via inner publisher) and a semantic event (via NATS).

    Args:
        inner_publisher: Dynamo's KvEventPublisher (Python binding).
        worker_id: This worker's identifier (pod name).
        nats_url: NATS server URL for semantic events.
        nats_subject: NATS subject for semantic events.
        embedder_type: Embedder type for MiniLM.
    """

    def __init__(
        self,
        inner_publisher: Any,
        worker_id: str = "",
        nats_url: str = "nats://nats:4222",
        nats_subject: str = "semblend.semantic.events",
        embedder_type: str = "minilm",
    ) -> None:
        self._inner = inner_publisher
        self._worker_id = worker_id or os.environ.get(
            "HOSTNAME", os.environ.get("POD_NAME", "unknown")
        )
        self._nats_url = nats_url
        self._nats_subject = nats_subject
        self._embedder = None
        self._embedder_type = embedder_type
        self._nats_client = None
        self._stats = {
            "block_events_published": 0,
            "semantic_events_published": 0,
            "semantic_events_failed": 0,
        }

    def publish_stored(
        self,
        block_hashes: list,
        parent_block_hash: Any,
        token_ids: list[int],
        block_size: int,
        **kwargs: Any,
    ) -> None:
        """Publish block stored event + semantic event.

        Forwards the original block event to Dynamo's publisher,
        then computes an embedding and publishes a semantic event.
        """
        # Forward to Dynamo's publisher
        self._inner.publish_stored(
            block_hashes=block_hashes,
            parent_block_hash=parent_block_hash,
            token_ids=token_ids,
            block_size=block_size,
            **kwargs,
        )
        self._stats["block_events_published"] += 1

        # Publish semantic event (fire-and-forget, never blocks inference)
        self._publish_semantic_stored(token_ids)

    def publish_removed(self, block_hashes: list, **kwargs: Any) -> None:
        """Forward block removed event."""
        self._inner.publish_removed(block_hashes=block_hashes, **kwargs)

    def shutdown(self) -> None:
        """Forward shutdown."""
        self._inner.shutdown()
        if self._nats_client is not None:
            try:
                self._nats_client.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Semantic event publishing
    # ------------------------------------------------------------------

    def _publish_semantic_stored(self, token_ids: list[int]) -> None:
        """Compute embedding and publish semantic event (fire-and-forget)."""
        if len(token_ids) < 100:
            return

        try:
            embedder = self._get_embedder()
            if embedder is None:
                return

            # Decode tokens to text for embedding
            text = self._tokens_to_text(token_ids)
            if text is None:
                return

            # Embed text (match pipeline preprocessing)
            from semblend_core.pipeline import _order_invariant_text
            processed_text = _order_invariant_text(text)
            embedding = embedder.embed(processed_text)
            if embedding is None:
                return

            # Build semantic event
            import hashlib
            token_hash = hashlib.md5(
                str(token_ids[:256]).encode()
            ).hexdigest()[:16]

            event = {
                "type": "stored",
                "worker_id": self._worker_id,
                "tenant_id": os.environ.get("SEMBLEND_TENANT_ID", "default"),
                "donor_id": token_hash,
                "embedding": np.asarray(embedding, dtype=np.float32).tolist(),
                "num_tokens": len(token_ids),
            }

            self._publish_nats(event)
            self._stats["semantic_events_published"] += 1

        except Exception as e:
            self._stats["semantic_events_failed"] += 1
            logger.debug("Semantic event publish failed: %s", e)

    def _publish_nats(self, event: dict) -> None:
        """Publish event to NATS (fire-and-forget)."""
        client = self._get_nats_client()
        if client is None:
            return

        payload = json.dumps(event).encode("utf-8")
        try:
            client.publish(self._nats_subject, payload)
        except Exception as e:
            logger.debug("NATS publish failed: %s", e)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_embedder(self):
        if self._embedder is not None:
            return self._embedder

        from semblend_core.embedder import create_embedder
        self._embedder = create_embedder(self._embedder_type)
        return self._embedder

    def _get_nats_client(self):
        if self._nats_client is not None:
            return self._nats_client

        try:
            import asyncio

            import nats

            # Synchronous NATS client for fire-and-forget publishing
            loop = asyncio.new_event_loop()
            self._nats_client = loop.run_until_complete(
                nats.connect(self._nats_url)
            )
            logger.info("NATS connected: %s", self._nats_url)
        except Exception as e:
            logger.warning("NATS connection failed: %s", e)
        return self._nats_client

    def _tokens_to_text(self, token_ids: list[int]) -> str | None:
        model_name = os.environ.get("SEMBLEND_MODEL_NAME", "")
        if not model_name:
            return None
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            max_decode = 2000
            sampled = token_ids[:max_decode]
            return tokenizer.decode(sampled, skip_special_tokens=True)
        except Exception:
            return None

    def get_stats(self) -> dict:
        return {**self._stats}

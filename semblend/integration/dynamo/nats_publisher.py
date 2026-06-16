"""Background NATS publisher for donor lifecycle events.

Holds one persistent connection on a dedicated event-loop thread so a
synchronous engine hot path can enqueue publishes without blocking on an
event loop. Publishing is fail-safe: it never raises into the caller, so a
NATS outage cannot break inference.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from typing import Any, Optional

logger = logging.getLogger("semblend.dynamo.nats")

DEFAULT_SUBJECT = "synapse.semblend.events"


class ThreadedNatsPublisher:
    """Publish JSON events to a NATS subject from synchronous code."""

    def __init__(self, url: str, subject: str = DEFAULT_SUBJECT) -> None:
        self._url = url
        self._subject = subject
        self._nc: Optional[Any] = None
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="semblend-nats-publisher", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connect())
        except Exception:
            logger.warning("NATS connect failed; event emission disabled", exc_info=True)
            self._ready.set()
            return
        self._ready.set()
        self._loop.run_forever()

    async def _connect(self) -> None:
        import nats

        self._nc = await nats.connect(self._url)
        logger.info("NATS connected: %s subject=%s", self._url, self._subject)

    def wait_ready(self, timeout: float = 10.0) -> bool:
        """Block until the connection attempt resolves. Returns True if connected."""
        self._ready.wait(timeout=timeout)
        return self._nc is not None

    def publish(self, event: dict[str, Any]) -> None:
        """Enqueue one event for publishing. Never raises."""
        if self._nc is None:
            return
        try:
            payload = json.dumps(event).encode("utf-8")
            asyncio.run_coroutine_threadsafe(
                self._nc.publish(self._subject, payload), self._loop
            )
        except Exception:
            logger.debug("NATS publish enqueue failed", exc_info=True)

    @classmethod
    def from_env(cls) -> Optional["ThreadedNatsPublisher"]:
        """Build from SEMBLEND_NATS_URL / SEMBLEND_NATS_SUBJECT, or None if unset."""
        url = os.environ.get("SEMBLEND_NATS_URL")
        if not url:
            return None
        subject = os.environ.get("SEMBLEND_NATS_SUBJECT", DEFAULT_SUBJECT)
        try:
            return cls(url, subject)
        except Exception:
            logger.warning("NATS publisher init failed", exc_info=True)
            return None

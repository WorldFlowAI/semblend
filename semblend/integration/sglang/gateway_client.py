"""Synapse Gateway client stub for the gateway embedding-backend mode.

Real implementation will be filled in during G3 when we wire the gRPC
path. For G1 this is a typed stub so the provider adapter can be
constructed and exercised in `embedding_backend="gateway"` mode without
a real gateway available.

Semantics of the real client:
  - register(donor) -> donor_id: POST donor embedding + metadata to the
    Synapse Gateway cuVS CAGRA index.
  - find_candidates(query_embedding, top_k, extra_key, timeout_ms)
    -> list[(donor_id, cosine)]: async ANN lookup with deadline.
    Returns an empty list on timeout (never raises) so the provider can
    fall back to cold prefill without blocking the match path.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SynapseGatewayError(RuntimeError):
    """Raised on unrecoverable gateway errors (not timeouts — those return empty)."""


class SynapseGatewayClient:
    """Placeholder gateway client.

    The stub records calls in-memory so tests can assert on them without
    network access. Replace with a gRPC/HTTP implementation in G3.
    """

    def __init__(self, url: str, timeout_ms: int = 3) -> None:
        if not url:
            raise SynapseGatewayError("gateway_url must be set for embedding_backend='gateway'")
        self._url = url
        self._timeout_ms = timeout_ms
        self._registered: List[Tuple[str, int]] = []

    @property
    def url(self) -> str:
        return self._url

    @property
    def registered(self) -> List[Tuple[str, int]]:
        """Return list of (donor_id, num_tokens) for every donor registered via this stub."""
        return list(self._registered)

    def register(
        self,
        donor_id: str,
        embedding: np.ndarray,
        token_ids: List[int],
        extra_key: Optional[str] = None,
    ) -> str:
        """Register a donor with the gateway.

        Stub: records the call and returns `donor_id` unchanged.
        """
        _ = embedding, extra_key  # unused in stub
        self._registered.append((donor_id, len(token_ids)))
        logger.debug("SynapseGatewayClient(stub).register(%s, n=%d)", donor_id, len(token_ids))
        return donor_id

    def find_candidates(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        extra_key: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """Find candidates via the remote ANN index.

        Stub: returns an empty list (miss). The provider's caller must
        treat this the same as a local miss — safe degradation to cold
        prefill.
        """
        _ = query_embedding, top_k, extra_key
        logger.debug("SynapseGatewayClient(stub).find_candidates -> []")
        return []

"""Token-level KV cache client for Synapse's REST API.

Implements Tier 1 exact-prefix-match KV caching. Token hashing matches
the Rust gateway: SHA-256 of little-endian-packed uint32 token IDs.

Endpoints:
    PUT    /api/v1/kv-cache/{token_hash}  -- store KV state
    GET    /api/v1/kv-cache/{token_hash}  -- retrieve KV state
    DELETE /api/v1/kv-cache/{token_hash}  -- evict KV state
    GET    /api/v1/kv-cache/stats         -- store statistics
"""

from __future__ import annotations

import base64
import hashlib
import logging
import struct
from dataclasses import dataclass
from typing import Any

import numpy as np
import requests

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KvStats:
    """KV cache store statistics."""

    entry_count: int
    total_bytes: int
    max_entries: int
    fill_ratio: float


@dataclass(frozen=True)
class LoadKvResult:
    """Result of loading a KV state from the cache."""

    num_tokens: int
    num_layers: int
    num_heads: int
    head_dim: int
    kv_data: np.ndarray


def compute_token_hash(token_ids: list[int]) -> str:
    """Compute the SHA-256 token hash matching the Rust gateway.

    The hash is computed over a little-endian packed array of uint32 token IDs:
        SHA-256(struct.pack("<" + "I" * len(token_ids), *token_ids))

    Args:
        token_ids: List of integer token IDs.

    Returns:
        Hex-encoded SHA-256 digest string.
    """
    if not token_ids:
        raise ValueError("token_ids must not be empty")

    fmt = "<" + "I" * len(token_ids)
    packed = struct.pack(fmt, *token_ids)
    return hashlib.sha256(packed).hexdigest()


class SynapseKVClient:
    """Client for Synapse's token-level KV cache REST API.

    Stores and retrieves KV state blocks keyed by SHA-256 token hash,
    enabling exact-prefix-match KV cache sharing across vLLM instances.

    Args:
        base_url: Synapse gateway base URL (e.g. "http://localhost:8080").
        timeout: Request timeout in seconds.
        tenant_id: Tenant identifier for multi-tenant isolation.
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        tenant_id: str = "default",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._tenant_id = tenant_id
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def tenant_id(self) -> str:
        return self._tenant_id

    def store_kv_state(
        self,
        token_ids: list[int],
        kv_data: np.ndarray,
        num_layers: int,
        num_heads: int,
        head_dim: int,
    ) -> dict[str, Any]:
        """Store a KV state block for the given token sequence.

        Args:
            token_ids: Token IDs that produced this KV state.
            kv_data: KV tensor data as a numpy array with dtype float16.
                Shape: [num_layers, 2, num_heads, num_tokens, head_dim].
            num_layers: Number of transformer layers.
            num_heads: Number of KV attention heads per layer.
            head_dim: Dimension of each attention head.

        Returns:
            Response dict with keys: tokenHash, bytesStored, overwroteExisting.

        Raises:
            requests.HTTPError: On non-2xx response.
            ValueError: On invalid input.
        """
        if kv_data.dtype != np.float16:
            raise ValueError(
                f"kv_data must be float16, got {kv_data.dtype}"
            )

        token_hash = compute_token_hash(token_ids)
        kv_b64 = base64.standard_b64encode(kv_data.tobytes()).decode("ascii")
        num_tokens = len(token_ids)

        body = {
            "numTokens": num_tokens,
            "numLayers": num_layers,
            "numHeads": num_heads,
            "headDim": head_dim,
            "kvData": kv_b64,
            "tenantId": self._tenant_id,
        }

        url = f"{self._base_url}/api/v1/kv-cache/{token_hash}"
        resp = self._session.put(url, json=body, timeout=self._timeout)
        resp.raise_for_status()
        return resp.json()

    def load_kv_state(self, token_ids: list[int]) -> LoadKvResult | None:
        """Load a KV state block for the given token sequence.

        Args:
            token_ids: Token IDs to look up.

        Returns:
            LoadKvResult with the KV tensor data, or None if not found.

        Raises:
            requests.HTTPError: On non-2xx/404 response.
        """
        token_hash = compute_token_hash(token_ids)
        url = f"{self._base_url}/api/v1/kv-cache/{token_hash}"

        resp = self._session.get(url, timeout=self._timeout)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()

        data = resp.json()
        kv_bytes = base64.standard_b64decode(data["kvData"])
        kv_array = np.frombuffer(kv_bytes, dtype=np.float16).reshape(
            data["numLayers"],
            2,
            data["numHeads"],
            data["numTokens"],
            data["headDim"],
        )

        return LoadKvResult(
            num_tokens=data["numTokens"],
            num_layers=data["numLayers"],
            num_heads=data["numHeads"],
            head_dim=data["headDim"],
            kv_data=kv_array.copy(),
        )

    def load_kv_state_by_hash(
        self, token_hash: str
    ) -> LoadKvResult | None:
        """Load a KV state block by its pre-computed token hash.

        Unlike ``load_kv_state``, this skips hash computation and uses
        the hash directly. Used by Tier 3 to load donor KV state when
        only the donor_id (token hash) is known.

        Args:
            token_hash: Hex-encoded SHA-256 token hash.

        Returns:
            LoadKvResult with the KV tensor data, or None if not found.

        Raises:
            requests.HTTPError: On non-2xx/404 response.
        """
        url = f"{self._base_url}/api/v1/kv-cache/{token_hash}"

        resp = self._session.get(url, timeout=self._timeout)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()

        data = resp.json()
        kv_bytes = base64.standard_b64decode(data["kvData"])
        kv_array = np.frombuffer(kv_bytes, dtype=np.float16).reshape(
            data["numLayers"],
            2,
            data["numHeads"],
            data["numTokens"],
            data["headDim"],
        )

        return LoadKvResult(
            num_tokens=data["numTokens"],
            num_layers=data["numLayers"],
            num_heads=data["numHeads"],
            head_dim=data["headDim"],
            kv_data=kv_array.copy(),
        )

    def delete_kv_state(self, token_ids: list[int]) -> bool:
        """Delete a KV state block for the given token sequence.

        Args:
            token_ids: Token IDs whose KV state should be evicted.

        Returns:
            True if the entry was deleted, False if it was not found.

        Raises:
            requests.HTTPError: On non-2xx/404 response.
        """
        token_hash = compute_token_hash(token_ids)
        url = f"{self._base_url}/api/v1/kv-cache/{token_hash}"

        resp = self._session.delete(url, timeout=self._timeout)
        if resp.status_code == 404:
            return False
        resp.raise_for_status()
        return True

    def get_stats(self) -> KvStats:
        """Retrieve KV cache store statistics.

        Returns:
            KvStats with current store metrics.

        Raises:
            requests.HTTPError: On non-2xx response.
        """
        url = f"{self._base_url}/api/v1/kv-cache/stats"
        resp = self._session.get(url, timeout=self._timeout)
        resp.raise_for_status()

        data = resp.json()
        return KvStats(
            entry_count=data["entryCount"],
            total_bytes=data["totalBytes"],
            max_entries=data["maxEntries"],
            fill_ratio=data["fillRatio"],
        )

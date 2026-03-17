"""Segment-level KV cache client for Synapse's REST API.

Implements Tier 3 approximate KV transfer. Segments represent semantic
blocks of a prompt (system prompt, RAG context, user query) with their
own KV tensors. Segment hashing uses Blake3 of (segment_type:text).

Endpoints:
    PUT    /api/v1/kv-cache/segments/{segment_hash}      -- store segment KV
    GET    /api/v1/kv-cache/segments/{segment_hash}      -- retrieve segment KV
    DELETE /api/v1/kv-cache/segments/{segment_hash}      -- evict segment KV
    POST   /api/v1/kv-cache/segments/search              -- semantic search
    GET    /api/v1/kv-cache/segments/stats                -- segment statistics
    POST   /api/v1/kv-cache/segments/plan                -- compute transfer plan
    POST   /api/v1/kv-cache/segments/plan/find           -- find donor + plan
    POST   /api/v1/kv-cache/segments/delta-tree/insert   -- insert DELTA tree node
"""

from __future__ import annotations

import base64
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import requests

logger = logging.getLogger(__name__)

# Try to use the blake3 package for native performance; fall back to
# hashlib.blake2b (available in stdlib) with a 32-byte digest to approximate
# Blake3's output size when blake3 is not installed.
try:
    import blake3 as _blake3

    def _blake3_hash(data: bytes) -> str:
        return _blake3.blake3(data).hexdigest()

except ImportError:
    logger.debug(
        "blake3 package not available, falling back to hashlib.blake2b"
    )

    def _blake3_hash(data: bytes) -> str:
        return hashlib.blake2b(data, digest_size=32).hexdigest()


@dataclass(frozen=True)
class SegmentTypeCount:
    """Count of cached segments for a given type."""

    segment_type: str
    count: int


@dataclass(frozen=True)
class SegmentKvStats:
    """Segment KV cache store statistics."""

    entry_count: int
    total_bytes: int
    max_entries: int
    fill_ratio: float
    type_counts: list[SegmentTypeCount] = field(default_factory=list)


@dataclass(frozen=True)
class LoadSegmentResult:
    """Result of loading a segment's KV state."""

    segment_type: str
    num_tokens: int
    num_layers: int
    num_heads: int
    head_dim: int
    kv_data: np.ndarray
    embedding: list[float] | None


@dataclass(frozen=True)
class SegmentSearchHit:
    """A single result from semantic segment search."""

    segment_hash: str
    segment_type: str
    num_tokens: int
    similarity: float
    text_preview: str | None


@dataclass(frozen=True)
class SegmentSearchResult:
    """Full response from a segment search query."""

    results: list[SegmentSearchHit]
    total_searched: int


@dataclass(frozen=True)
class KvSlotAction:
    """A single slot action in a KV transfer plan.

    Either copy a KV vector from the donor or mark as placeholder
    for fresh computation.
    """

    action: str  # "copy_from_donor" or "placeholder"
    donor_pos: int | None  # Only for copy_from_donor
    target_pos: int


@dataclass(frozen=True)
class KvTransferPlan:
    """A KV cache transfer plan from the Rust KV Editor.

    Describes which token positions can reuse donor KV vectors
    and which need fresh computation (placeholders).
    """

    donor_id: str
    target_len: int
    donor_len: int
    slot_actions: list[KvSlotAction]
    copy_positions: list[int]
    placeholder_positions: list[int]
    num_copied: int
    num_placeholders: int
    reuse_ratio: float
    viable: bool


@dataclass(frozen=True)
class FindDonorResult:
    """Result from the find-donor-and-plan endpoint."""

    found: bool
    plan: KvTransferPlan | None
    tree_size: int


def compute_segment_hash(segment_type: str, text: str) -> str:
    """Compute the segment content hash matching the Rust gateway.

    The hash is Blake3 of the concatenation: segment_type + ":" + text.

    Args:
        segment_type: Classification string (e.g. "system_prompt").
        text: The segment's textual content.

    Returns:
        Hex-encoded hash digest string.
    """
    payload = f"{segment_type}:{text}".encode("utf-8")
    return _blake3_hash(payload)


class SynapseSegmentClient:
    """Client for Synapse's segment-level KV cache REST API.

    Stores and retrieves per-segment KV tensors for Tier 3 approximate
    KV transfer, with optional embedding-based semantic search.

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

    def store_segment_kv(
        self,
        segment_hash: str,
        segment_type: str,
        kv_data: np.ndarray,
        num_tokens: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        embedding: list[float] | None = None,
        text_preview: str | None = None,
    ) -> dict[str, Any]:
        """Store a segment's KV state.

        Args:
            segment_hash: Content hash (from compute_segment_hash).
            segment_type: Classification (system_prompt, rag_context, etc.).
            kv_data: KV tensor data as float16 numpy array.
            num_tokens: Number of tokens in this segment.
            num_layers: Number of transformer layers.
            num_heads: Number of KV attention heads per layer.
            head_dim: Dimension of each attention head.
            embedding: Optional embedding vector for semantic search.
            text_preview: Optional text preview for debugging.

        Returns:
            Response dict with keys: segmentHash, bytesStored, overwroteExisting.

        Raises:
            requests.HTTPError: On non-2xx response.
            ValueError: On invalid input.
        """
        if kv_data.dtype != np.float16:
            raise ValueError(
                f"kv_data must be float16, got {kv_data.dtype}"
            )

        kv_b64 = base64.standard_b64encode(kv_data.tobytes()).decode("ascii")

        body: dict[str, Any] = {
            "segmentType": segment_type,
            "numTokens": num_tokens,
            "numLayers": num_layers,
            "numHeads": num_heads,
            "headDim": head_dim,
            "kvData": kv_b64,
            "tenantId": self._tenant_id,
        }

        if embedding is not None:
            body["embedding"] = embedding
        if text_preview is not None:
            body["textPreview"] = text_preview

        url = f"{self._base_url}/api/v1/kv-cache/segments/{segment_hash}"
        resp = self._session.put(url, json=body, timeout=self._timeout)
        resp.raise_for_status()
        return resp.json()

    def load_segment_kv(
        self, segment_hash: str
    ) -> LoadSegmentResult | None:
        """Load a segment's KV state by its content hash.

        Args:
            segment_hash: Content hash of the segment.

        Returns:
            LoadSegmentResult with KV data, or None if not found.

        Raises:
            requests.HTTPError: On non-2xx/404 response.
        """
        url = f"{self._base_url}/api/v1/kv-cache/segments/{segment_hash}"
        resp = self._session.get(url, timeout=self._timeout)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()

        data = resp.json()
        kv_bytes = base64.standard_b64decode(data["kvData"])
        kv_array = np.frombuffer(kv_bytes, dtype=np.float16).copy()

        return LoadSegmentResult(
            segment_type=data["segmentType"],
            num_tokens=data["numTokens"],
            num_layers=data["numLayers"],
            num_heads=data["numHeads"],
            head_dim=data["headDim"],
            kv_data=kv_array,
            embedding=data.get("embedding"),
        )

    def delete_segment_kv(self, segment_hash: str) -> bool:
        """Delete a segment's KV state.

        Args:
            segment_hash: Content hash of the segment to evict.

        Returns:
            True if the entry was deleted, False if not found.

        Raises:
            requests.HTTPError: On non-2xx/404 response.
        """
        url = f"{self._base_url}/api/v1/kv-cache/segments/{segment_hash}"
        resp = self._session.delete(url, timeout=self._timeout)
        if resp.status_code == 404:
            return False
        resp.raise_for_status()
        return True

    def search_segments(
        self,
        embedding: list[float],
        top_k: int = 5,
        threshold: float = 0.85,
        segment_type: str | None = None,
    ) -> SegmentSearchResult:
        """Search for cached segments by embedding similarity.

        Args:
            embedding: Query embedding vector.
            top_k: Maximum number of results to return.
            threshold: Minimum cosine similarity threshold (0.0 to 1.0).
            segment_type: Optional filter by segment type.

        Returns:
            SegmentSearchResult with matching segments sorted by similarity.

        Raises:
            requests.HTTPError: On non-2xx response.
            ValueError: If embedding is empty.
        """
        if not embedding:
            raise ValueError("embedding must not be empty")

        body: dict[str, Any] = {
            "embedding": embedding,
            "topK": top_k,
            "threshold": threshold,
            "tenantId": self._tenant_id,
        }

        if segment_type is not None:
            body["segmentType"] = segment_type

        url = f"{self._base_url}/api/v1/kv-cache/segments/search"
        resp = self._session.post(url, json=body, timeout=self._timeout)
        resp.raise_for_status()

        data = resp.json()
        hits = [
            SegmentSearchHit(
                segment_hash=r["segmentHash"],
                segment_type=r["segmentType"],
                num_tokens=r["numTokens"],
                similarity=r["similarity"],
                text_preview=r.get("textPreview"),
            )
            for r in data["results"]
        ]

        return SegmentSearchResult(
            results=hits,
            total_searched=data["totalSearched"],
        )

    def request_transfer_plan(
        self,
        donor_tokens: list[int],
        target_tokens: list[int],
        donor_id: str,
        min_reuse_ratio: float = 0.5,
        layer_deviations: list[float] | None = None,
        num_layers: int | None = None,
        layer_deviation_threshold: float = 0.3,
    ) -> KvTransferPlan:
        """Request a KV transfer plan from the gateway's KV Editor.

        Computes token-level alignment between donor and target sequences,
        producing a plan that describes which positions can reuse donor KV
        vectors and which need fresh computation.

        Args:
            donor_tokens: Token IDs of the cached donor sequence.
            target_tokens: Token IDs of the new target sequence.
            donor_id: Identifier for the donor cache entry.
            min_reuse_ratio: Minimum ratio of reusable tokens (0.0 to 1.0).
            layer_deviations: Optional per-layer deviation scores for
                CacheBlend-style layer recomputation hints.
            num_layers: Number of transformer layers (required if
                layer_deviations is provided).
            layer_deviation_threshold: Threshold above which a layer
                needs full recomputation.

        Returns:
            KvTransferPlan with slot actions and viability flag.

        Raises:
            requests.HTTPError: On non-2xx response.
        """
        body: dict[str, Any] = {
            "donorTokens": donor_tokens,
            "targetTokens": target_tokens,
            "donorId": donor_id,
            "minReuseRatio": min_reuse_ratio,
            "layerDeviationThreshold": layer_deviation_threshold,
        }

        if layer_deviations is not None:
            body["layerDeviations"] = layer_deviations
        if num_layers is not None:
            body["numLayers"] = num_layers

        url = f"{self._base_url}/api/v1/kv-cache/segments/plan"
        resp = self._session.post(url, json=body, timeout=self._timeout)
        resp.raise_for_status()

        return self._parse_transfer_plan_response(resp.json())

    def find_donor_and_plan(
        self,
        target_tokens: list[int],
        min_reuse_ratio: float = 0.5,
        layer_deviation_threshold: float = 0.3,
    ) -> FindDonorResult:
        """Search the DELTA tree for the best donor and compute a plan.

        Queries the gateway's DELTA tree for the closest cached request
        and computes a KV transfer plan if a suitable donor exists.

        Args:
            target_tokens: Token IDs of the new target sequence.
            min_reuse_ratio: Minimum ratio of reusable tokens.
            layer_deviation_threshold: Per-layer deviation threshold.

        Returns:
            FindDonorResult with the plan (if a donor was found).

        Raises:
            requests.HTTPError: On non-2xx response.
        """
        body: dict[str, Any] = {
            "targetTokens": target_tokens,
            "minReuseRatio": min_reuse_ratio,
            "layerDeviationThreshold": layer_deviation_threshold,
        }

        url = f"{self._base_url}/api/v1/kv-cache/segments/plan/find"
        resp = self._session.post(url, json=body, timeout=self._timeout)
        resp.raise_for_status()

        data = resp.json()
        plan = None
        if data.get("plan") is not None:
            plan = self._parse_plan_from_data(data["plan"])

        return FindDonorResult(
            found=data["found"],
            plan=plan,
            tree_size=data["treeSize"],
        )

    def insert_delta_node(
        self,
        request_id: str,
        token_ids: list[int],
        embedding: list[float] | None = None,
        kv_resident: bool = True,
    ) -> dict[str, Any]:
        """Insert a request node into the DELTA tree.

        After computing KV state for a request, insert it into the
        DELTA tree so future similar requests can reuse its KV cache.

        Args:
            request_id: Unique identifier for this request.
            token_ids: Token IDs of the request.
            embedding: Optional embedding vector for semantic search.
            kv_resident: Whether the KV state is currently in cache.

        Returns:
            Response dict with requestId and treeSize.

        Raises:
            requests.HTTPError: On non-2xx response.
        """
        body: dict[str, Any] = {
            "requestId": request_id,
            "tokenIds": token_ids,
            "kvResident": kv_resident,
        }

        if embedding is not None:
            body["embedding"] = embedding

        url = f"{self._base_url}/api/v1/kv-cache/segments/delta-tree/insert"
        resp = self._session.post(url, json=body, timeout=self._timeout)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _parse_plan_from_data(plan_data: dict[str, Any]) -> KvTransferPlan:
        """Parse a KvTransferPlan from JSON data."""
        slot_actions = [
            KvSlotAction(
                action=sa["action"],
                donor_pos=sa.get("donorPos"),
                target_pos=sa["targetPos"],
            )
            for sa in plan_data.get("slotActions", [])
        ]

        return KvTransferPlan(
            donor_id=plan_data["donorId"],
            target_len=plan_data["targetLen"],
            donor_len=plan_data["donorLen"],
            slot_actions=slot_actions,
            copy_positions=plan_data.get("copyPositions", []),
            placeholder_positions=plan_data.get("placeholderPositions", []),
            num_copied=plan_data["numCopied"],
            num_placeholders=plan_data["numPlaceholders"],
            reuse_ratio=plan_data["reuseRatio"],
            viable=plan_data.get("viable", True),
        )

    def _parse_transfer_plan_response(
        self, data: dict[str, Any]
    ) -> KvTransferPlan:
        """Parse a TransferPlanResponse from the gateway."""
        plan = self._parse_plan_from_data(data["plan"])
        # Override viable from the top-level response
        return KvTransferPlan(
            donor_id=plan.donor_id,
            target_len=plan.target_len,
            donor_len=plan.donor_len,
            slot_actions=plan.slot_actions,
            copy_positions=plan.copy_positions,
            placeholder_positions=plan.placeholder_positions,
            num_copied=plan.num_copied,
            num_placeholders=plan.num_placeholders,
            reuse_ratio=plan.reuse_ratio,
            viable=data.get("viable", True),
        )

    def get_stats(self) -> SegmentKvStats:
        """Retrieve segment KV cache store statistics.

        Returns:
            SegmentKvStats with current store metrics and type breakdown.

        Raises:
            requests.HTTPError: On non-2xx response.
        """
        url = f"{self._base_url}/api/v1/kv-cache/segments/stats"
        resp = self._session.get(url, timeout=self._timeout)
        resp.raise_for_status()

        data = resp.json()
        type_counts = [
            SegmentTypeCount(
                segment_type=tc["segmentType"],
                count=tc["count"],
            )
            for tc in data.get("typeCounts", [])
        ]

        return SegmentKvStats(
            entry_count=data["entryCount"],
            total_bytes=data["totalBytes"],
            max_entries=data["maxEntries"],
            fill_ratio=data["fillRatio"],
            type_counts=type_counts,
        )

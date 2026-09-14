"""Tenant isolation for the SGLang HiCache backend and the provider adapter.

Both paths hand donors to a store whose filter is fail-open: a lookup that
supplies no isolation key sees every donor, whoever produced it. So each
path derives a namespace from the isolation value SGLang carries
(``extra_key`` / ``cache_salt``), binds it at registration AND at lookup,
and maps an absent value to a sentinel rather than to None — that is what
makes "no key" and "some key" a guaranteed mismatch in both directions
while two unkeyed requests still share donors, as a single-tenant
deployment always has.

The raw isolation value is tenant-identifying, so only its hash is ever
recorded or logged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np
import pytest

from semblend.integration.sglang.config import SemBlendProviderConfig
from semblend.integration.sglang.hicache_backend import (
    SemBlendHiCacheStorage,
    _SemBlendDonorIndex,
)
from semblend.integration.sglang.provider import SemBlendProviderAdapter
from semblend.integration.sglang.radix_backend import (
    NO_EXTRA_KEY_NAMESPACE,
    _isolation_namespace,
)
from semblend.integration.sglang.types import FuzzyMatchResult

TENANT_A = "tenant-a-secret-salt"
TENANT_B = "tenant-b-secret-salt"
NS_A = _isolation_namespace(TENANT_A)
NS_B = _isolation_namespace(TENANT_B)


def _unit_vector(seed: int, dim: int = 384) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


# ---------------------------------------------------------------------
# HiCache donor index
# ---------------------------------------------------------------------


class TestHiCacheDonorIndexNamespaces:
    def test_donor_is_invisible_to_another_namespace(self):
        index = _SemBlendDonorIndex(max_entries=10, min_similarity=0.5)
        emb = _unit_vector(1)
        index.register("hash-a", emb, [1, 2, 3], namespace=NS_A)

        # Identical embedding, different tenant: similarity is irrelevant.
        assert index.find_semantic_match(emb, namespace=NS_B) is None

    def test_same_namespace_reuse_preserved(self):
        index = _SemBlendDonorIndex(max_entries=10, min_similarity=0.5)
        emb = _unit_vector(2)
        index.register("hash-a", emb, [1, 2, 3], namespace=NS_A)

        assert index.find_semantic_match(emb, namespace=NS_A) == "hash-a"

    def test_unkeyed_requests_still_share_donors(self):
        index = _SemBlendDonorIndex(max_entries=10, min_similarity=0.5)
        emb = _unit_vector(3)
        index.register("hash-a", emb, [1, 2, 3])

        assert index.find_semantic_match(emb) == "hash-a"

    def test_absent_and_salted_never_match_either_way(self):
        index = _SemBlendDonorIndex(max_entries=10, min_similarity=0.5)
        emb = _unit_vector(4)

        index.register("keyed", emb, [1, 2, 3], namespace=NS_A)
        assert index.find_semantic_match(emb, namespace=NO_EXTRA_KEY_NAMESPACE) is None

        unkeyed = _SemBlendDonorIndex(max_entries=10, min_similarity=0.5)
        unkeyed.register("unkeyed", emb, [1, 2, 3])
        assert unkeyed.find_semantic_match(emb, namespace=NS_A) is None

    def test_same_prompt_from_two_tenants_registers_twice(self):
        index = _SemBlendDonorIndex(max_entries=10, min_similarity=0.5)
        emb = _unit_vector(5)

        # Same storage key from both tenants: deduping across namespaces
        # would leave the second tenant with no donor it may see.
        index.register("shared-hash", emb, [1, 2, 3], namespace=NS_A)
        index.register("shared-hash", emb, [4, 5, 6], namespace=NS_B)

        assert index.size == 2
        assert index.get_token_ids("shared-hash", namespace=NS_A) == [1, 2, 3]
        assert index.get_token_ids("shared-hash", namespace=NS_B) == [4, 5, 6]

    def test_rejections_are_counted(self):
        index = _SemBlendDonorIndex(max_entries=10, min_similarity=0.5)
        emb = _unit_vector(6)
        index.register("hash-a", emb, [1, 2, 3], namespace=NS_A)
        index.register("hash-b", emb, [4, 5, 6], namespace=NS_A)

        assert index.namespace_rejections == 0
        assert index.find_semantic_match(emb, namespace=NS_B) is None
        assert index.namespace_rejections == 2


# ---------------------------------------------------------------------
# HiCacheStorage backend
# ---------------------------------------------------------------------


class _StubTokenEmbedder:
    """Token-id embedder, so `set()` needs no model."""

    def embed_tokens(self, token_ids: list[int]) -> np.ndarray:
        return _unit_vector(sum(token_ids) % 1000)


@pytest.fixture
def storage() -> SemBlendHiCacheStorage:
    backend = SemBlendHiCacheStorage(storage_config=None)
    backend._embedder = _StubTokenEmbedder()  # noqa: SLF001 — test injection
    return backend


class TestHiCacheStorageNamespaces:
    def test_donor_registration_binds_the_calling_tenant(self, storage):
        storage.register_token_ids("hash-a", [11, 22, 33], extra_key=TENANT_A)
        storage.set("hash-a", value="kv-a")

        probe = _StubTokenEmbedder().embed_tokens([11, 22, 33])
        index = storage._donor_index  # noqa: SLF001 — test inspection
        assert index.find_semantic_match(probe, namespace=NS_B) is None
        assert index.find_semantic_match(probe, namespace=NS_A) == "hash-a"

    def test_exact_lookup_is_namespace_gated(self, storage):
        extra_a = {"extra_key": TENANT_A}
        extra_b = {"extra_key": TENANT_B}
        storage.set("hash-a", value="kv-a", extra_info=extra_a)

        assert storage.get("hash-a", extra_info=extra_b) is None
        assert storage.exists("hash-a", extra_info=extra_b) is False
        assert storage.batch_exists(["hash-a"], extra_info=extra_b) == 0

        # Same tenant still reads its own entry.
        assert storage.get("hash-a", extra_info=extra_a) == "kv-a"
        assert storage.batch_exists(["hash-a"], extra_info=extra_a) == 1

    def test_unkeyed_deployment_behaviour_unchanged(self, storage):
        storage.set("hash-a", value="kv-a")

        assert storage.get("hash-a") == "kv-a"
        assert storage.exists("hash-a") is True
        assert storage.batch_exists(["hash-a"]) == 1

    def test_salted_entry_is_invisible_to_unkeyed_lookup(self, storage):
        storage.set("hash-a", value="kv-a", extra_info={"extra_key": TENANT_A})

        assert storage.get("hash-a") is None
        assert storage.exists("hash-a") is False

    def test_rejections_are_reported_in_stats(self, storage):
        storage.set("hash-a", value="kv-a", extra_info={"extra_key": TENANT_A})
        assert storage.get_stats()["cross_namespace_rejected"] == 0

        storage.get("hash-a", extra_info={"extra_key": TENANT_B})
        assert storage.get_stats()["cross_namespace_rejected"] == 1

    def test_raw_salt_never_reaches_a_record(self, storage):
        storage.register_token_ids("hash-a", [11, 22, 33], extra_key=TENANT_A)
        storage.set("hash-a", value="kv-a")

        recorded = [
            str(storage._kv_store["hash-a"][0]),  # noqa: SLF001 — test inspection
            *[str(key) for key in storage._donor_index._entries],  # noqa: SLF001
        ]
        assert recorded
        for value in recorded:
            assert TENANT_A not in value


# ---------------------------------------------------------------------
# Provider adapter
# ---------------------------------------------------------------------


@dataclass
class _StubEmbedder:
    dim: int = 384

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        vec[len(text) % self.dim] = 1.0
        return vec


@dataclass
class _StubPosMap:
    donor_positions: list
    target_positions: list


@dataclass
class _StubPipelineResult:
    found: bool
    donor_id: Optional[str] = None
    donor_ids: List[str] = field(default_factory=list)
    similarity: float = 0.0
    reuse_ratio: float = 0.0
    donor_tokens: list = field(default_factory=list)
    slot_actions: list = field(default_factory=list)
    layer_deviations: list = field(default_factory=list)
    position_map: _StubPosMap = field(default_factory=lambda: _StubPosMap([], []))
    confidence_tier: str = "exact"
    composite_plan: Optional[Any] = None


class _StubDonorStore:
    def __init__(self) -> None:
        self.donors: list = []

    def add_donor(self, node: Any) -> None:
        self.donors.append(node)

    def get_donor(self, donor_id: str) -> Any:
        for node in self.donors:
            if node.request_id == donor_id:
                return node
        return None

    def clear(self) -> None:
        self.donors.clear()


class _FailOpenPipeline:
    """Pipeline whose donor search ignores extra_key entirely.

    That is the core store's real behaviour for an absent key, and it puts
    the whole burden of isolation on the adapter — which is what these
    tests are about.
    """

    def __init__(self) -> None:
        self._embedder = _StubEmbedder()
        self._donor_store = _StubDonorStore()
        self.next_result = _StubPipelineResult(found=False)
        self.find_donor_calls: list = []

    def find_donor(self, token_ids, prompt_text="", top_k=5, extra_key=None, **kwargs):
        self.find_donor_calls.append({"extra_key": extra_key, **kwargs})
        return self.next_result

    def clear_donors(self) -> None:
        self._donor_store.clear()


@pytest.fixture
def pipeline() -> _FailOpenPipeline:
    return _FailOpenPipeline()


@pytest.fixture
def adapter(pipeline) -> SemBlendProviderAdapter:
    config = SemBlendProviderConfig(
        min_similarity=0.60,
        min_reuse_ratio=0.50,
        min_match_length=8,
        max_entries=100,
        block_size=4,
        enable_bathtub=True,
        model_arch="llama",
    )
    return SemBlendProviderAdapter(config=config, pipeline=pipeline)


def _register(adapter, request_id: str, extra_key: Optional[str], base: int = 100) -> None:
    adapter.register_donor(
        request_id=request_id,
        token_ids=list(range(16)),
        kv_cache=list(range(base, base + 16)),
        cache_start_pos=0,
        cache_end_pos=16,
        prompt_text=f"registration {request_id}",
        extra_key=extra_key,
    )
    # Single-worker executor: a no-op that completes means the donor insert
    # queued above has landed. Draining this way keeps the adapter usable
    # for the next registration.
    adapter._register_executor.submit(lambda: None).result()  # noqa: SLF001


def _hit_result(donor_id: str, donor_ids: Optional[List[str]] = None) -> _StubPipelineResult:
    return _StubPipelineResult(
        found=True,
        donor_id=donor_id,
        donor_ids=list(donor_ids or []),
        similarity=0.90,
        reuse_ratio=0.85,
        donor_tokens=list(range(16)),
        position_map=_StubPosMap(
            donor_positions=list(range(16)),
            target_positions=list(range(16)),
        ),
    )


class TestProviderRegistration:
    def test_binds_hashed_namespace_not_the_raw_extra_key(self, adapter, pipeline):
        _register(adapter, "donor-A", TENANT_A)

        node = pipeline._donor_store.donors[0]  # noqa: SLF001 — test inspection
        assert node.extra_key == NS_A
        assert TENANT_A not in node.extra_key

        handle = adapter._donor_kv["donor-A"]  # noqa: SLF001 — test inspection
        assert handle.namespace == NS_A
        assert TENANT_A not in handle.namespace

    def test_absent_extra_key_binds_the_sentinel(self, adapter, pipeline):
        _register(adapter, "donor-A", None)

        node = pipeline._donor_store.donors[0]  # noqa: SLF001 — test inspection
        assert node.extra_key == NO_EXTRA_KEY_NAMESPACE


class TestProviderLookup:
    def test_lookup_never_reaches_the_core_with_none(self, adapter, pipeline):
        adapter.match(
            prompt_token_ids=list(range(32)),
            already_matched_len=0,
            prompt_text="query",
        )

        # None means "every donor is visible" to the core store — the
        # unkeyed request must arrive carrying the sentinel instead.
        assert pipeline.find_donor_calls[-1]["extra_key"] == NO_EXTRA_KEY_NAMESPACE

    def test_lookup_passes_the_hashed_namespace(self, adapter, pipeline):
        adapter.match(
            prompt_token_ids=list(range(32)),
            already_matched_len=0,
            prompt_text="query",
            extra_key=TENANT_A,
        )

        passed = pipeline.find_donor_calls[-1]["extra_key"]
        assert passed == NS_A
        assert TENANT_A not in passed

    def test_keyed_same_tenant_match_is_served(self, adapter, pipeline):
        _register(adapter, "donor-A", TENANT_A)
        pipeline.next_result = _hit_result("donor-A")

        result = adapter.match(
            prompt_token_ids=list(range(32)),
            already_matched_len=0,
            prompt_text="query",
            extra_key=TENANT_A,
        )

        assert isinstance(result, FuzzyMatchResult)
        assert result.cached_token_count == 16
        assert list(result.kv_cache_indices) == list(range(100, 116))
        assert adapter.stats()["cross_namespace_rejected"] == 0

    def test_foreign_donor_is_rejected_and_counted(self, adapter, pipeline):
        _register(adapter, "donor-A", TENANT_A)
        pipeline.next_result = _hit_result("donor-A")

        result = adapter.match(
            prompt_token_ids=list(range(32)),
            already_matched_len=0,
            prompt_text="query",
            extra_key=TENANT_B,
        )

        assert result is None
        assert adapter.stats()["cross_namespace_rejected"] == 1

    def test_unkeyed_request_cannot_consume_a_keyed_donor(self, adapter, pipeline):
        _register(adapter, "donor-A", TENANT_A)
        pipeline.next_result = _hit_result("donor-A")

        result = adapter.match(
            prompt_token_ids=list(range(32)),
            already_matched_len=0,
            prompt_text="query",
        )

        assert result is None
        assert adapter.stats()["cross_namespace_rejected"] == 1


class TestProviderCompositeResults:
    def test_composite_dropped_whole_when_any_donor_is_foreign(self, adapter, pipeline):
        _register(adapter, "donor-A", TENANT_A, base=100)
        _register(adapter, "donor-B", TENANT_B, base=200)
        # Primary donor belongs to the requester; the second does not. The
        # plan mixes their KV, so trimming is not an option.
        pipeline.next_result = _hit_result("donor-A", donor_ids=["donor-A", "donor-B"])

        result = adapter.match(
            prompt_token_ids=list(range(32)),
            already_matched_len=0,
            prompt_text="query",
            extra_key=TENANT_A,
        )

        assert result is None
        assert adapter.stats()["cross_namespace_rejected"] == 1

    def test_composite_served_when_every_donor_shares_the_namespace(self, adapter, pipeline):
        _register(adapter, "donor-A", TENANT_A, base=100)
        _register(adapter, "donor-B", TENANT_A, base=200)
        pipeline.next_result = _hit_result("donor-A", donor_ids=["donor-A", "donor-B"])

        result = adapter.match(
            prompt_token_ids=list(range(32)),
            already_matched_len=0,
            prompt_text="query",
            extra_key=TENANT_A,
        )

        assert isinstance(result, FuzzyMatchResult)
        assert adapter.stats()["cross_namespace_rejected"] == 0

    def test_composite_donor_evicted_by_the_adapter_resolves_from_the_store(
        self, adapter, pipeline
    ):
        _register(adapter, "donor-A", TENANT_A, base=100)
        _register(adapter, "donor-B", TENANT_A, base=200)
        # The adapter's handle LRU is smaller than the pipeline's store, so
        # a composite donor can outlive its handle; the pipeline store still
        # proves its tenancy and the plan survives.
        del adapter._donor_kv["donor-B"]  # noqa: SLF001 — simulate handle eviction
        pipeline.next_result = _hit_result("donor-A", donor_ids=["donor-A", "donor-B"])

        result = adapter.match(
            prompt_token_ids=list(range(32)),
            already_matched_len=0,
            prompt_text="query",
            extra_key=TENANT_A,
        )

        assert isinstance(result, FuzzyMatchResult)

    def test_composite_donor_unknown_everywhere_fails_closed(self, adapter, pipeline):
        _register(adapter, "donor-A", TENANT_A, base=100)
        pipeline.next_result = _hit_result("donor-A", donor_ids=["donor-A", "ghost-donor"])

        result = adapter.match(
            prompt_token_ids=list(range(32)),
            already_matched_len=0,
            prompt_text="query",
            extra_key=TENANT_A,
        )

        assert result is None
        assert adapter.stats()["cross_namespace_rejected"] == 1


class TestProviderCanonicalRescue:
    def test_sole_donor_fallback_is_namespace_gated(self, adapter):
        _register(adapter, "donor-A", TENANT_A)
        adapter._donor_kv["donor-A"].token_ids = list(range(16))  # noqa: SLF001
        result = _hit_result("composite-xyz")

        # "There is only one registered donor" says nothing about whose it
        # is: the rescue must not hand tenant B tenant A's donor.
        donor_id, handle = adapter._resolve_canon_handle(result, NS_B)  # noqa: SLF001
        assert handle is None
        assert adapter.stats()["cross_namespace_rejected"] == 1

        donor_id, handle = adapter._resolve_canon_handle(result, NS_A)  # noqa: SLF001
        assert (donor_id, handle) == ("donor-A", adapter._donor_kv["donor-A"])  # noqa: SLF001

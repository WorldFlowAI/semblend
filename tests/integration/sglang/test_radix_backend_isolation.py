"""Tenant isolation for the SGLang RadixCache backend's semantic donor path.

SGLang keys its own radix tree on the request's ``extra_key`` — the field its
API layer builds from ``cache_salt`` (and ``lora_id``) — so it is the
per-request isolation boundary operators already set per tenant. SemBlend's
donor store must honour the same boundary: a donor registered by a request
carrying one extra_key must never be visible to a request carrying another,
and a keyed request must never consume an unkeyed donor.

Two requests that both lack an extra_key DO share donors — that is the
single-tenant deployment, and the older SGLang releases whose match_prefix
key is a bare token list, and both must keep working.

SGLang is not importable on CPU CI, so its key and request shapes are
reproduced here from the versions this backend targets and the subclass is
driven directly through match_prefix / cache_finished_req.
"""

from __future__ import annotations

import numpy as np
import pytest

EMBED_DIM = 8

# Donor and near-identical target: four tokens differ out of 600, which is
# the shape of two tenants sending the same templated prompt with different
# private bodies. The difference sits inside the first 256 tokens because
# that window is the store's self-exclusion key; a target that shared it
# would be treated as the donor itself rather than as another request.
DONOR_TOKENS = list(range(1000, 1600))
TARGET_TOKENS = DONOR_TOKENS[:100] + [7777, 7778, 7779, 7780] + DONOR_TOKENS[104:]

# Same unit vector for every prompt: cosine similarity 1.0. Isolation must
# not depend on the prompts being dissimilar — every prompt here is a perfect
# semantic match so the only thing that can keep a donor away is the namespace.
_UNIT_EMBEDDING = np.ones(EMBED_DIM, dtype=np.float32) / np.sqrt(EMBED_DIM)


class _FakeRadixKey:
    """sglang.srt.mem_cache.radix_cache.RadixKey, as far as the backend reads it.

    ``cache_salt`` is carried too because builds differ on where the salt
    lives: some copy it onto the key, some leave it on the Req alone.
    """

    def __init__(self, token_ids, extra_key=None, cache_salt=None):
        self.token_ids = list(token_ids)
        self.extra_key = extra_key
        self.cache_salt = cache_salt

    def __len__(self):
        return len(self.token_ids)

    def __iter__(self):
        return iter(self.token_ids)


class _FakeMatchPrefixParams:
    """sglang.srt.mem_cache.base_prefix_cache.MatchPrefixParams."""

    def __init__(self, key, req=None):
        self.key = key
        self.req = req


class _FakeReq:
    """The subset of sglang's Req the donor path reads."""

    def __init__(self, rid, token_ids, extra_key=None, cache_salt=None):
        self.rid = rid
        self.origin_input_ids = list(token_ids)
        self.extra_key = extra_key
        self.cache_salt = cache_salt


def _tokens_of(key):
    if isinstance(key, (list, tuple)):
        return list(key)
    if hasattr(key, "token_ids"):
        return list(key.token_ids)
    return _tokens_of(key.key)


class _FakeRadixCache:
    """Exact-sequence stand-in for SGLang's RadixCache.

    Returns a full-length hit only for a token sequence that was cached
    verbatim and a miss for everything else, so the near-identical target
    always misses the tree and reaches the donor store, while the donor's
    own tokens (which the backend re-queries once it picks a donor) are
    found. SGLang's own extra_key partitioning of the tree is deliberately
    not modelled: the layer under test is SemBlend's donor store, which
    must hold the boundary on its own.
    """

    def __init__(self, *args, **kwargs):
        self._cached: list[list[int]] = []

    def match_prefix(self, key, **kwargs):
        token_ids = _tokens_of(key)
        if token_ids in self._cached:
            return (list(range(len(token_ids))), None)
        return ([], None)

    def cache_finished_req(self, req, *args, **kwargs):
        self._cached.append(list(req.origin_input_ids))


@pytest.fixture(autouse=True)
def deterministic_env(monkeypatch):
    monkeypatch.setenv("SEMBLEND_ENABLED", "1")
    monkeypatch.setenv("SEMBLEND_MAX_DONORS", "16")
    monkeypatch.setenv("SEMBLEND_MIN_SIMILARITY", "0.60")


@pytest.fixture
def backend():
    import semblend.integration.sglang.radix_backend as module

    return module


def _make_cache(module):
    """A SemBlendRadixCache over the fake tree, with embedding stubbed out.

    Embedding for real needs a tokenizer and a MiniLM model; the donor path
    under test only needs a vector, and a constant one makes every prompt
    a perfect match.
    """
    cls = module.get_semblend_radix_cache_class(_FakeRadixCache)
    cache = cls()
    cache._embed_tokens = lambda token_ids: _UNIT_EMBEDDING
    return cache


def _register(cache, extra_key):
    cache.cache_finished_req(_FakeReq("donor", DONOR_TOKENS, extra_key))


def _target_key(extra_key, shape="params"):
    """The match_prefix key shape a given SGLang version passes."""
    req = _FakeReq("target", TARGET_TOKENS, extra_key)
    radix_key = _FakeRadixKey(TARGET_TOKENS, extra_key)
    if shape == "params":
        return _FakeMatchPrefixParams(radix_key, req)
    if shape == "radix_key":
        return radix_key
    return list(TARGET_TOKENS)


def _lookup(cache, extra_key, shape="params"):
    """Drive match_prefix with the key shape a given SGLang version passes."""
    return cache.match_prefix(_target_key(extra_key, shape))


class TestNamespaceDerivation:
    """The namespace is a pure function of the key's isolation fields."""

    def test_distinct_extra_keys_give_distinct_namespaces(self, backend):
        ns_a = backend._key_namespace(_FakeRadixKey([1], "tenant-a"))
        ns_b = backend._key_namespace(_FakeRadixKey([1], "tenant-b"))

        assert ns_a != ns_b

    def test_same_extra_key_gives_same_namespace(self, backend):
        ns_1 = backend._key_namespace(_FakeRadixKey([1], "tenant-a"))
        ns_2 = backend._key_namespace(_FakeRadixKey([2], "tenant-a"))

        assert ns_1 == ns_2

    def test_extra_key_is_not_stored_verbatim(self, backend):
        ns = backend._key_namespace(_FakeRadixKey([1], "acme-corp"))

        assert "acme-corp" not in ns

    @pytest.mark.parametrize("extra_key", [None, "", "   "])
    def test_absent_extra_key_maps_to_sentinel(self, backend, extra_key):
        ns = backend._key_namespace(_FakeRadixKey([1], extra_key))

        assert ns == backend.NO_EXTRA_KEY_NAMESPACE

    def test_bare_token_list_maps_to_sentinel(self, backend):
        """Older SGLang passes the token list itself; it carries no isolation."""
        ns = backend._key_namespace([1, 2, 3])

        assert ns == backend.NO_EXTRA_KEY_NAMESPACE

    def test_key_without_isolation_attributes_maps_to_sentinel(self, backend):
        class _Legacy:
            token_ids = [1, 2, 3]

        ns = backend._key_namespace(_Legacy())

        assert ns == backend.NO_EXTRA_KEY_NAMESPACE

    def test_keyed_namespace_never_equals_sentinel(self, backend):
        ns = backend._key_namespace(_FakeRadixKey([1], "tenant-a"))

        assert ns != backend.NO_EXTRA_KEY_NAMESPACE

    def test_params_wrapper_exposes_inner_key_namespace(self, backend):
        """MatchPrefixParams must not mask the RadixKey it wraps."""
        wrapped = _FakeMatchPrefixParams(_FakeRadixKey([1], "tenant-a"))
        bare = _FakeRadixKey([1], "tenant-a")

        assert backend._key_namespace(wrapped) == backend._key_namespace(bare)

    def test_cache_salt_attribute_is_honoured(self, backend):
        """A build that keeps cache_salt separate from extra_key still isolates."""

        class _SaltedKey:
            token_ids = [1]
            extra_key = None
            cache_salt = "tenant-a"

        ns = backend._key_namespace(_SaltedKey())

        assert ns != backend.NO_EXTRA_KEY_NAMESPACE
        assert "tenant-a" not in ns

    def test_request_and_key_derive_one_namespace(self, backend):
        """Registration (from Req) and lookup (from key) must agree."""
        from_req = backend._req_namespace(_FakeReq("r", [1], "tenant-a"))
        from_key = backend._key_namespace(_FakeRadixKey([1], "tenant-a"))

        assert from_req == from_key


class TestSplitFieldNamespaceComposition:
    """The namespace is composed per field, not taken from the first object.

    SGLang spreads the isolation fields across the objects it hands the two
    sides of the donor path: the LoRA-derived extra_key rides on the
    RadixKey, the tenant's cache_salt stays on the Req, and MatchPrefixParams
    wraps both. Registration sees the Req and lookup sees the key, so unless
    both compose from the same field set wherever it appears, a salted
    request registers under one namespace and looks up in another.
    """

    def test_key_does_not_mask_the_requests_salt(self, backend):
        """The six-line reproduction: a salted request read the unsalted pool.

        Taking the first object in the chain that carried any isolation value
        stopped at the RadixKey, whose extra_key is the LoRA id alone. The
        Req's cache_salt was dropped, so the salted request resolved to the
        very namespace unsalted requests on the same adapter register into.
        """
        salted = _FakeReq("target", TARGET_TOKENS, "lora-7", cache_salt="tenant-a")
        key = _FakeMatchPrefixParams(_FakeRadixKey(TARGET_TOKENS, "lora-7"), salted)
        unsalted = _FakeReq("public", DONOR_TOKENS, "lora-7")

        assert backend._key_namespace(key) == backend._req_namespace(salted)
        assert backend._key_namespace(key) != backend._req_namespace(unsalted)

    def test_salt_on_the_key_and_lora_on_the_request_also_agree(self, backend):
        """The mirror split: neither side may depend on which object carries what."""
        req = _FakeReq("target", TARGET_TOKENS, "lora-7", cache_salt="tenant-a")
        key = _FakeMatchPrefixParams(
            _FakeRadixKey(TARGET_TOKENS, cache_salt="tenant-a"),
            _FakeReq("target", TARGET_TOKENS, "lora-7"),
        )

        assert backend._key_namespace(key) == backend._req_namespace(req)

    def test_request_below_the_first_wrapper_is_still_read(self, backend):
        """A one-level walk missed a Req nested deeper and fell to the sentinel."""
        req = _FakeReq("target", TARGET_TOKENS, cache_salt="tenant-a")
        nested = _FakeMatchPrefixParams(_FakeMatchPrefixParams(_FakeRadixKey(TARGET_TOKENS), req))

        assert backend._key_namespace(nested) == backend._req_namespace(req)
        assert backend._key_namespace(nested) != backend.NO_EXTRA_KEY_NAMESPACE

    def test_outer_wrapper_wins_a_field_it_carries(self, backend):
        """Breadth first: the outermost object that has a field supplies it."""
        inner = _FakeRadixKey(TARGET_TOKENS, "lora-7")
        outer = _FakeMatchPrefixParams(inner)
        outer.extra_key = "lora-9"

        assert backend._key_namespace(outer) == backend._key_namespace(
            _FakeRadixKey(TARGET_TOKENS, "lora-9")
        )

    def test_self_referential_chain_terminates(self, backend):
        """A key that points back at its own request must not loop forever."""
        req = _FakeReq("target", TARGET_TOKENS, cache_salt="tenant-a")
        key = _FakeMatchPrefixParams(_FakeRadixKey(TARGET_TOKENS), req)
        req.key = key

        assert backend._key_namespace(key) == backend._req_namespace(req)

    @pytest.mark.parametrize("shape", ["params", "radix_key", "bare_list"])
    def test_register_then_lookup_land_in_one_namespace(self, backend, shape):
        """Every key shape the module supports agrees with registration.

        A bare token list carries no isolation value at all, so its
        deployment is the unsalted one — that is the namespace both sides
        must land in there, and it must be the same one either way.
        """
        extra_key = None if shape == "bare_list" else "tenant-a"
        donor_req = _FakeReq("donor", DONOR_TOKENS, extra_key)

        assert backend._key_namespace(_target_key(extra_key, shape)) == backend._req_namespace(
            donor_req
        )


class TestDonorStoreIsolation:
    """The in-process store behind the RadixCache subclass."""

    def _store(self, module):
        return module._SemBlendDonorStore(max_entries=8, min_similarity=0.50)

    def test_donor_not_visible_across_extra_keys(self, backend):
        store = self._store(backend)
        store.add_donor(tuple(DONOR_TOKENS), _UNIT_EMBEDDING, namespace="ns-tenant-a")

        assert store.find_donor(_UNIT_EMBEDDING, namespace="ns-tenant-b") is None

    def test_donor_visible_within_same_extra_key(self, backend):
        store = self._store(backend)
        store.add_donor(tuple(DONOR_TOKENS), _UNIT_EMBEDDING, namespace="ns-tenant-a")

        donor = store.find_donor(_UNIT_EMBEDDING, namespace="ns-tenant-a")

        assert donor is not None
        assert donor.token_ids == tuple(DONOR_TOKENS)

    def test_keyed_request_cannot_see_unkeyed_donor(self, backend):
        store = self._store(backend)
        store.add_donor(
            tuple(DONOR_TOKENS),
            _UNIT_EMBEDDING,
            namespace=backend.NO_EXTRA_KEY_NAMESPACE,
        )

        assert store.find_donor(_UNIT_EMBEDDING, namespace="ns-tenant-a") is None

    def test_unkeyed_request_cannot_see_keyed_donor(self, backend):
        store = self._store(backend)
        store.add_donor(tuple(DONOR_TOKENS), _UNIT_EMBEDDING, namespace="ns-tenant-a")

        # The default namespace is the no-extra-key sentinel, so a caller
        # that supplies nothing still cannot reach a keyed donor.
        assert store.find_donor(_UNIT_EMBEDDING) is None

    def test_unkeyed_donors_still_shared_single_tenant(self, backend):
        store = self._store(backend)
        store.add_donor(
            tuple(DONOR_TOKENS),
            _UNIT_EMBEDDING,
            namespace=backend.NO_EXTRA_KEY_NAMESPACE,
        )

        donor = store.find_donor(_UNIT_EMBEDDING)

        assert donor is not None
        assert donor.token_ids == tuple(DONOR_TOKENS)

    def test_rejections_are_counted(self, backend):
        store = self._store(backend)
        store.add_donor(tuple(DONOR_TOKENS), _UNIT_EMBEDDING, namespace="ns-tenant-a")
        store.add_donor(tuple(TARGET_TOKENS), _UNIT_EMBEDDING, namespace="ns-tenant-a")

        store.find_donor(_UNIT_EMBEDDING, namespace="ns-tenant-b")

        assert store.namespace_rejections == 2

    def test_high_similarity_does_not_override_isolation(self, backend):
        """An identical-embedding donor from another tenant is still refused."""
        store = self._store(backend)
        store.add_donor(tuple(DONOR_TOKENS), _UNIT_EMBEDDING, namespace="ns-tenant-a")

        # Cosine similarity is exactly 1.0 — the highest possible score.
        assert store.find_donor(_UNIT_EMBEDDING, namespace="ns-tenant-b") is None

    def test_same_prompt_registers_once_per_tenant(self, backend):
        """Dedup is per namespace: tenant B's copy must not be dropped as a
        duplicate of tenant A's, or B would never get a donor it may see."""
        store = self._store(backend)
        store.add_donor(tuple(DONOR_TOKENS), _UNIT_EMBEDDING, namespace="ns-tenant-a")
        store.add_donor(tuple(DONOR_TOKENS), _UNIT_EMBEDDING, namespace="ns-tenant-b")

        assert store.size == 2
        assert store.find_donor(_UNIT_EMBEDDING, namespace="ns-tenant-b") is not None


class TestRadixCachePath:
    """End to end through match_prefix and cache_finished_req."""

    def test_cross_extra_key_request_gets_no_donor(self, backend):
        cache = _make_cache(backend)
        _register(cache, "tenant-a")

        _lookup(cache, "tenant-b")

        stats = cache.get_semblend_stats()
        assert stats["semantic_hits"] == 0
        assert stats["misses"] == 1
        assert stats["cross_namespace_rejected"] == 1

    def test_same_extra_key_request_reuses_donor(self, backend):
        cache = _make_cache(backend)
        _register(cache, "tenant-a")

        _lookup(cache, "tenant-a")

        stats = cache.get_semblend_stats()
        assert stats["semantic_hits"] == 1
        assert stats["cross_namespace_rejected"] == 0

    def test_unkeyed_request_gets_no_keyed_donor(self, backend):
        cache = _make_cache(backend)
        _register(cache, "tenant-a")

        _lookup(cache, None)

        stats = cache.get_semblend_stats()
        assert stats["semantic_hits"] == 0
        assert stats["cross_namespace_rejected"] == 1

    def test_keyed_request_gets_no_unkeyed_donor(self, backend):
        cache = _make_cache(backend)
        _register(cache, None)

        _lookup(cache, "tenant-a")

        stats = cache.get_semblend_stats()
        assert stats["semantic_hits"] == 0
        assert stats["cross_namespace_rejected"] == 1

    def test_unkeyed_deployment_still_reuses(self, backend):
        cache = _make_cache(backend)
        _register(cache, None)

        _lookup(cache, None)

        assert cache.get_semblend_stats()["semantic_hits"] == 1

    @pytest.mark.parametrize("shape", ["params", "radix_key", "bare_list"])
    def test_every_key_shape_reuses_within_one_tenant(self, backend, shape):
        """The key wrapper SGLang uses must not change the outcome."""
        cache = _make_cache(backend)
        extra_key = None if shape == "bare_list" else "tenant-a"
        _register(cache, extra_key)

        _lookup(cache, extra_key, shape=shape)

        assert cache.get_semblend_stats()["semantic_hits"] == 1

    @pytest.mark.parametrize("shape", ["params", "radix_key"])
    def test_every_key_shape_isolates_across_tenants(self, backend, shape):
        cache = _make_cache(backend)
        _register(cache, "tenant-a")

        _lookup(cache, "tenant-b", shape=shape)

        stats = cache.get_semblend_stats()
        assert stats["semantic_hits"] == 0
        assert stats["cross_namespace_rejected"] == 1

    def test_bare_list_request_gets_no_keyed_donor(self, backend):
        """An older-shaped key carries no isolation and must not reach a keyed donor."""
        cache = _make_cache(backend)
        _register(cache, "tenant-a")

        _lookup(cache, None, shape="bare_list")

        stats = cache.get_semblend_stats()
        assert stats["semantic_hits"] == 0
        assert stats["cross_namespace_rejected"] == 1

    def test_donor_records_its_namespace(self, backend):
        cache = _make_cache(backend)
        _register(cache, "tenant-a")

        (entry,) = cache._semblend_donor_store._entries.values()
        expected = backend._req_namespace(_FakeReq("donor", DONOR_TOKENS, "tenant-a"))

        assert entry.namespace == expected
        assert "tenant-a" not in entry.namespace

    def test_isolation_is_reported_in_stats(self, backend):
        cache = _make_cache(backend)
        _register(cache, "tenant-a")

        _lookup(cache, "tenant-b")
        _lookup(cache, "tenant-c")

        assert cache.get_semblend_stats()["cross_namespace_rejected"] == 2

    def test_salted_request_does_not_reach_the_unsalted_pool(self, backend):
        """End to end for the split-field reproduction.

        The donor is an unsalted request on the same LoRA adapter, so it
        sits in the extra_key-only namespace the salted lookup used to
        resolve to. It must be refused, not served.
        """
        cache = _make_cache(backend)
        cache.cache_finished_req(_FakeReq("public", DONOR_TOKENS, "lora-7"))

        salted = _FakeReq("target", TARGET_TOKENS, "lora-7", cache_salt="tenant-a")
        cache.match_prefix(_FakeMatchPrefixParams(_FakeRadixKey(TARGET_TOKENS, "lora-7"), salted))

        stats = cache.get_semblend_stats()
        assert stats["semantic_hits"] == 0
        assert stats["cross_namespace_rejected"] == 1

    def test_salted_request_still_reuses_its_own_donor(self, backend):
        """The same split must not cost a tenant reuse of its own KV."""
        cache = _make_cache(backend)
        cache.cache_finished_req(_FakeReq("donor", DONOR_TOKENS, "lora-7", cache_salt="tenant-a"))

        salted = _FakeReq("target", TARGET_TOKENS, "lora-7", cache_salt="tenant-a")
        cache.match_prefix(_FakeMatchPrefixParams(_FakeRadixKey(TARGET_TOKENS, "lora-7"), salted))

        stats = cache.get_semblend_stats()
        assert stats["semantic_hits"] == 1
        assert stats["cross_namespace_rejected"] == 0


class TestHiCacheReset:
    """The sibling HiCacheStorage backend's reset, covered here with it.

    clear() dropped the KV store and the two token-id maps but left the
    semantic donor index standing, so a donor outlived the KV it points at
    and a later lookup in the same namespace could still match it.
    """

    def _storage(self):
        from semblend.integration.sglang.hicache_backend import SemBlendHiCacheStorage

        storage = SemBlendHiCacheStorage()
        storage._compute_embedding = lambda token_ids: _UNIT_EMBEDDING
        return storage

    def _store_donor(self, storage, key="hash-a"):
        storage.register_token_ids(key, list(DONOR_TOKENS))
        storage.set(key, value="kv-blob")

    def test_clear_drops_the_semantic_donor_index(self):
        storage = self._storage()
        self._store_donor(storage)
        assert storage._donor_index.size == 1

        storage.clear()

        assert storage._donor_index.size == 0
        assert storage._donor_index.find_semantic_match(_UNIT_EMBEDDING) is None
        assert storage.get_stats()["donor_index_size"] == 0

    def test_clear_drops_donors_in_every_namespace(self):
        storage = self._storage()
        storage.register_token_ids("hash-a", list(DONOR_TOKENS), extra_key="tenant-a")
        storage.set("hash-a", value="kv-a")
        storage.register_token_ids("hash-b", list(DONOR_TOKENS), extra_key="tenant-b")
        storage.set("hash-b", value="kv-b")
        assert storage._donor_index.size == 2

        storage.clear()

        assert storage._donor_index.size == 0

    def test_clear_still_drops_the_kv_store(self):
        storage = self._storage()
        self._store_donor(storage)

        storage.clear()

        assert storage.get("hash-a") is None
        assert storage.exists("hash-a") is False
        assert storage.get_stats()["kv_store_size"] == 0

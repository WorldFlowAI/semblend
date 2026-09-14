"""Tenant isolation for the vLLM connector's semantic donor path.

vLLM mixes ``cache_salt`` into its own prefix-cache block hashes, so it is
the per-request isolation key operators already set per tenant. SemBlend's
donor store must honour the same boundary: a donor registered by a request
carrying one salt must never be visible to a request carrying another, and
a salted request must never consume an unsalted donor.

Two requests that both lack a cache_salt DO share donors — that is the
single-tenant deployment and it must keep working.

These tests cover both branches the connector can take: the default
pipeline path (SEMBLEND_USE_PIPELINE=1) and the legacy in-connector donor
store. vLLM and LMCache are not importable on CPU CI, so the vLLM module
surface is mocked and the connector is driven directly.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

EMBED_DIM = 8

# Donor and near-identical target: four tokens differ out of 600, which is
# the shape of two tenants sending the same templated prompt with different
# private bodies.
DONOR_TOKENS = list(range(1000, 1600))
TARGET_TOKENS = DONOR_TOKENS[:300] + [7777, 7778, 7779, 7780] + DONOR_TOKENS[304:]

DONOR_TEXT = "quarterly revenue summary for the western region " * 12
TARGET_TEXT = "quarterly revenue summary for the western region " * 12


@pytest.fixture(autouse=True)
def mock_vllm_imports():
    """Mock the vLLM module surface the connector imports at module scope."""
    installed = {}
    for mod_name in [
        "vllm",
        "vllm.config",
        "vllm.distributed",
        "vllm.distributed.kv_transfer",
        "vllm.distributed.kv_transfer.kv_connector",
        "vllm.distributed.kv_transfer.kv_connector.v1",
        "vllm.forward_context",
        "vllm.v1",
        "vllm.v1.core",
        "vllm.v1.core.kv_cache_manager",
        "vllm.v1.core.sched",
        "vllm.v1.core.sched.output",
        "vllm.v1.kv_cache_interface",
        "vllm.v1.request",
        # The connector imports torch at module scope; CI runs without it.
        "torch",
        "torch.cuda",
        "triton",
    ]:
        if mod_name not in sys.modules:
            installed[mod_name] = MagicMock()
            sys.modules[mod_name] = installed[mod_name]

    base_name = "vllm.distributed.kv_transfer.kv_connector.v1.base"
    if base_name not in sys.modules:
        mock_base = MagicMock()
        mock_base.KVConnectorBase_V1 = type("KVConnectorBase_V1", (), {})
        installed[base_name] = mock_base
        sys.modules[base_name] = mock_base

    yield

    for mod_name, mod in installed.items():
        if sys.modules.get(mod_name) is mod:
            del sys.modules[mod_name]


@pytest.fixture
def connector_module(mock_vllm_imports):
    import semblend_kv_connector.semblend_connector as module

    return module


class _StubEmbedder:
    """Same unit vector for every text: cosine similarity 1.0.

    Isolation must not depend on the prompts being dissimilar — these
    tests make every prompt a perfect semantic match so the only thing
    that can keep the donor away is the namespace.
    """

    dimension = EMBED_DIM

    def embed(self, text: str):
        return np.ones(EMBED_DIM, dtype=np.float32) / np.sqrt(EMBED_DIM)


class _FakeLMCache:
    """LMCache connector stand-in that always misses.

    Deliberately bare: the connector delegates unknown attributes here, so
    anything the test forgot to wire raises AttributeError instead of
    silently passing.
    """

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return (0, False)


class _FakeRequest:
    """The subset of vllm.v1.request.Request the donor path reads."""

    def __init__(self, request_id, token_ids, prompt, cache_salt=None):
        self.request_id = request_id
        self.all_token_ids = list(token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.prompt = prompt
        self.cache_salt = cache_salt


def _make_pipeline():
    from semblend_core.donor_store import DonorStore
    from semblend_core.pipeline import SemBlendPipeline

    store = DonorStore(
        max_entries=16,
        embedding_dim=EMBED_DIM,
        min_similarity=0.60,
        chunk_size=32,
    )
    pipeline = SemBlendPipeline(
        embedder_type="jaccard",
        donor_store=store,
        chunk_size=32,
        enable_pq_segments=False,
    )
    pipeline._embedder = _StubEmbedder()  # noqa: SLF001
    return pipeline


def _make_connector(module, pipeline=None):
    """A scheduler-side connector with the LMCache/GPU machinery stubbed out.

    Instantiating for real needs vLLM, LMCache and CUDA; the donor path
    under test only needs the donor store, the pipeline and the stats dict.
    """
    connector = object.__new__(module.SemBlendConnectorV1)
    connector._lmcache = _FakeLMCache()
    connector._enabled = True
    connector._pipeline = pipeline
    connector._donor_store = module.SemBlendDonorStore(max_entries=8, min_similarity=0.50)
    # Never load a sentence-transformers model in CI: the legacy path then
    # falls back to token-set Jaccard, which is the matcher under test.
    connector._donor_store.get_embedding = lambda text: None
    connector._local_embedder = None
    connector._cached_tokenizer = None
    connector._fingerprint_enabled = False
    connector._disable_rope_correction = True
    connector._use_partial_attn = False
    connector._model_runner_patched = False
    connector._active_hook = None
    connector._position_maps = {}
    connector._donor_token_map = {}
    connector._donor_matched_reqs = set()
    connector._stats = {
        "lmcache_hits": 0,
        "semblend_hits": 0,
        "semblend_misses": 0,
        "total_lookups": 0,
        "total_saves": 0,
        "partial_attn_applied": 0,
        "cross_namespace_rejected": 0,
    }
    return connector


@pytest.fixture(autouse=True)
def deterministic_env(monkeypatch):
    monkeypatch.setenv("SEMBLEND_CHUNK_FAST_PATH", "0")
    monkeypatch.delenv("SEMBLEND_MULTI_DONOR", raising=False)
    monkeypatch.delenv("SEMBLEND_PARAPHRASE_SERVE", raising=False)
    monkeypatch.delenv("SEMBLEND_FORCE_DELTA", raising=False)


class TestNamespaceDerivation:
    """The namespace is a pure function of the request's cache_salt."""

    def test_distinct_salts_give_distinct_namespaces(self, connector_module):
        ns_a = connector_module._request_namespace(_FakeRequest("a", [1], "x", "tenant-a"))
        ns_b = connector_module._request_namespace(_FakeRequest("b", [1], "x", "tenant-b"))

        assert ns_a != ns_b

    def test_same_salt_gives_same_namespace(self, connector_module):
        ns_1 = connector_module._request_namespace(_FakeRequest("a", [1], "x", "tenant-a"))
        ns_2 = connector_module._request_namespace(_FakeRequest("b", [1], "x", "tenant-a"))

        assert ns_1 == ns_2

    def test_salt_is_not_stored_verbatim(self, connector_module):
        ns = connector_module._request_namespace(_FakeRequest("a", [1], "x", "acme-corp"))

        assert "acme-corp" not in ns

    @pytest.mark.parametrize("salt", [None, "", "   "])
    def test_absent_salt_maps_to_sentinel(self, connector_module, salt):
        ns = connector_module._request_namespace(_FakeRequest("a", [1], "x", salt))

        assert ns == connector_module.NO_CACHE_SALT_NAMESPACE

    def test_request_without_cache_salt_attribute_maps_to_sentinel(self, connector_module):
        class _Legacy:
            request_id = "a"

        ns = connector_module._request_namespace(_Legacy())

        assert ns == connector_module.NO_CACHE_SALT_NAMESPACE

    def test_salted_namespace_never_equals_sentinel(self, connector_module):
        ns = connector_module._request_namespace(_FakeRequest("a", [1], "x", "tenant-a"))

        assert ns != connector_module.NO_CACHE_SALT_NAMESPACE


class TestLegacyDonorStoreIsolation:
    """The in-connector store used when SEMBLEND_USE_PIPELINE=0."""

    def _store(self, module):
        return module.SemBlendDonorStore(max_entries=8, min_similarity=0.50)

    def test_donor_not_visible_across_cache_salts(self, connector_module):
        store = self._store(connector_module)
        store.add_donor("donor", DONOR_TOKENS, DONOR_TEXT, namespace="ns-tenant-a")

        assert store.find_donor(TARGET_TOKENS, namespace="ns-tenant-b") is None

    def test_donor_visible_within_same_cache_salt(self, connector_module):
        store = self._store(connector_module)
        store.add_donor("donor", DONOR_TOKENS, DONOR_TEXT, namespace="ns-tenant-a")

        match = store.find_donor(TARGET_TOKENS, namespace="ns-tenant-a")

        assert match is not None
        assert match.donor.request_id == "donor"

    def test_salted_request_cannot_see_unsalted_donor(self, connector_module):
        store = self._store(connector_module)
        store.add_donor(
            "donor",
            DONOR_TOKENS,
            DONOR_TEXT,
            namespace=connector_module.NO_CACHE_SALT_NAMESPACE,
        )

        assert store.find_donor(TARGET_TOKENS, namespace="ns-tenant-a") is None

    def test_unsalted_request_cannot_see_salted_donor(self, connector_module):
        store = self._store(connector_module)
        store.add_donor("donor", DONOR_TOKENS, DONOR_TEXT, namespace="ns-tenant-a")

        # The default namespace is the no-cache-salt sentinel, so a caller
        # that supplies nothing still cannot reach a salted donor.
        assert store.find_donor(TARGET_TOKENS) is None

    def test_unsalted_donors_still_shared_single_tenant(self, connector_module):
        store = self._store(connector_module)
        store.add_donor(
            "donor",
            DONOR_TOKENS,
            DONOR_TEXT,
            namespace=connector_module.NO_CACHE_SALT_NAMESPACE,
        )

        match = store.find_donor(TARGET_TOKENS)

        assert match is not None
        assert match.donor.request_id == "donor"

    def test_rejections_are_counted(self, connector_module):
        store = self._store(connector_module)
        store.add_donor("d1", DONOR_TOKENS, DONOR_TEXT, namespace="ns-tenant-a")
        store.add_donor("d2", DONOR_TOKENS, DONOR_TEXT, namespace="ns-tenant-a")

        store.find_donor(TARGET_TOKENS, namespace="ns-tenant-b")

        assert store.namespace_rejections == 2

    def test_high_similarity_does_not_override_isolation(self, connector_module):
        """An exact-prompt donor from another tenant is still refused."""
        store = self._store(connector_module)
        store.add_donor("donor", DONOR_TOKENS, DONOR_TEXT, namespace="ns-tenant-a")

        # One token apart: the highest-similarity case short of self-match.
        near_identical = DONOR_TOKENS[:-1] + [4242]

        assert store.find_donor(near_identical, namespace="ns-tenant-b") is None


class TestLegacyConnectorPath:
    """End to end through get_num_new_matched_tokens, pipeline disabled."""

    def _register(self, connector, salt):
        connector._register_donor(_FakeRequest("donor", DONOR_TOKENS, DONOR_TEXT, salt))

    def test_cross_salt_request_gets_no_donor(self, connector_module):
        connector = _make_connector(connector_module, pipeline=None)
        self._register(connector, "tenant-a")

        connector.get_num_new_matched_tokens(
            _FakeRequest("victim", TARGET_TOKENS, TARGET_TEXT, "tenant-b"), 0
        )

        assert connector._stats["semblend_hits"] == 0
        assert connector._stats["semblend_misses"] == 1
        assert connector._stats["cross_namespace_rejected"] == 1

    def test_same_salt_request_reuses_donor(self, connector_module):
        connector = _make_connector(connector_module, pipeline=None)
        self._register(connector, "tenant-a")

        connector.get_num_new_matched_tokens(
            _FakeRequest("sibling", TARGET_TOKENS, TARGET_TEXT, "tenant-a"), 0
        )

        assert connector._stats["semblend_hits"] == 1
        assert connector._stats["cross_namespace_rejected"] == 0

    def test_unsalted_request_gets_no_salted_donor(self, connector_module):
        connector = _make_connector(connector_module, pipeline=None)
        self._register(connector, "tenant-a")

        connector.get_num_new_matched_tokens(
            _FakeRequest("outsider", TARGET_TOKENS, TARGET_TEXT, None), 0
        )

        assert connector._stats["semblend_hits"] == 0
        assert connector._stats["cross_namespace_rejected"] == 1

    def test_unsalted_deployment_still_reuses(self, connector_module):
        connector = _make_connector(connector_module, pipeline=None)
        self._register(connector, None)

        connector.get_num_new_matched_tokens(
            _FakeRequest("sibling", TARGET_TOKENS, TARGET_TEXT, None), 0
        )

        assert connector._stats["semblend_hits"] == 1

    def test_donor_records_its_namespace(self, connector_module):
        connector = _make_connector(connector_module, pipeline=None)
        self._register(connector, "tenant-a")

        entry = connector._donor_store._entries["donor"]
        expected = connector_module._request_namespace(
            _FakeRequest("donor", DONOR_TOKENS, DONOR_TEXT, "tenant-a")
        )

        assert entry.namespace == expected

    def test_isolation_is_reported_in_stats(self, connector_module):
        connector = _make_connector(connector_module, pipeline=None)
        self._register(connector, "tenant-a")

        connector.get_num_new_matched_tokens(
            _FakeRequest("victim", TARGET_TOKENS, TARGET_TEXT, "tenant-b"), 0
        )

        assert connector.get_stats()["cross_namespace_rejected"] == 1


class TestPipelineConnectorPath:
    """End to end on the DEFAULT path (SEMBLEND_USE_PIPELINE=1).

    Uses the real semblend_core pipeline and donor store, so this covers
    extra_key propagation at both the registration and the lookup call
    site, not just the connector's own post-check.
    """

    def _connector_with_donor(self, connector_module, donor_salt):
        connector = _make_connector(connector_module, pipeline=_make_pipeline())
        connector._register_donor(_FakeRequest("donor", DONOR_TOKENS, DONOR_TEXT, donor_salt))
        return connector

    def test_registration_binds_namespace_as_extra_key(self, connector_module):
        connector = self._connector_with_donor(connector_module, "tenant-a")

        node = connector._pipeline._donor_store.get_donor("donor")
        expected = connector_module._request_namespace(
            _FakeRequest("donor", DONOR_TOKENS, DONOR_TEXT, "tenant-a")
        )

        assert node.extra_key == expected

    def test_cross_salt_request_gets_no_donor(self, connector_module):
        connector = self._connector_with_donor(connector_module, "tenant-a")

        connector.get_num_new_matched_tokens(
            _FakeRequest("victim", TARGET_TOKENS, TARGET_TEXT, "tenant-b"), 0
        )

        assert connector._stats["semblend_hits"] == 0
        assert connector._stats["semblend_misses"] == 1
        assert connector._stats["cross_namespace_rejected"] == 1

    def test_same_salt_request_reuses_donor(self, connector_module):
        connector = self._connector_with_donor(connector_module, "tenant-a")

        connector.get_num_new_matched_tokens(
            _FakeRequest("sibling", TARGET_TOKENS, TARGET_TEXT, "tenant-a"), 0
        )

        assert connector._stats["semblend_hits"] == 1
        assert connector._stats["semblend_misses"] == 0
        assert connector._stats["cross_namespace_rejected"] == 0

    def test_unsalted_request_gets_no_salted_donor(self, connector_module):
        connector = self._connector_with_donor(connector_module, "tenant-a")

        connector.get_num_new_matched_tokens(
            _FakeRequest("outsider", TARGET_TOKENS, TARGET_TEXT, None), 0
        )

        assert connector._stats["semblend_hits"] == 0
        assert connector._stats["cross_namespace_rejected"] == 1

    def test_salted_request_gets_no_unsalted_donor(self, connector_module):
        connector = self._connector_with_donor(connector_module, None)

        connector.get_num_new_matched_tokens(
            _FakeRequest("tenant", TARGET_TOKENS, TARGET_TEXT, "tenant-a"), 0
        )

        assert connector._stats["semblend_hits"] == 0
        assert connector._stats["cross_namespace_rejected"] == 1

    def test_unsalted_deployment_still_reuses(self, connector_module):
        connector = self._connector_with_donor(connector_module, None)

        connector.get_num_new_matched_tokens(
            _FakeRequest("sibling", TARGET_TOKENS, TARGET_TEXT, None), 0
        )

        assert connector._stats["semblend_hits"] == 1


class TestPipelinePostCheckFailsClosed:
    """The connector's own gate, for when the store's filter is bypassed."""

    def test_candidate_from_another_namespace_is_dropped(self, connector_module):
        connector = _make_connector(connector_module, pipeline=_make_pipeline())
        connector._register_donor(_FakeRequest("donor", DONOR_TOKENS, DONOR_TEXT, "tenant-a"))

        class _Candidate:
            found = True
            donor_id = "donor"

        kept = connector._filter_candidates_by_namespace([_Candidate()], "ns-tenant-b")

        assert kept == []
        assert connector._stats["cross_namespace_rejected"] == 1

    def test_candidate_from_same_namespace_is_kept(self, connector_module):
        connector = _make_connector(connector_module, pipeline=_make_pipeline())
        connector._register_donor(_FakeRequest("donor", DONOR_TOKENS, DONOR_TEXT, "tenant-a"))
        namespace = connector_module._request_namespace(
            _FakeRequest("donor", DONOR_TOKENS, DONOR_TEXT, "tenant-a")
        )

        class _Candidate:
            found = True
            donor_id = "donor"

        candidate = _Candidate()
        kept = connector._filter_candidates_by_namespace([candidate], namespace)

        assert kept == [candidate]
        assert connector._stats["cross_namespace_rejected"] == 0

    def test_unknown_donor_is_rejected(self, connector_module):
        """Fail closed: a donor whose namespace cannot be read is refused."""
        connector = _make_connector(connector_module, pipeline=_make_pipeline())

        class _Candidate:
            found = True
            donor_id = "never-registered"

        kept = connector._filter_candidates_by_namespace([_Candidate()], "ns-tenant-a")

        assert kept == []
        assert connector._stats["cross_namespace_rejected"] == 1

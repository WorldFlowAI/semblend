"""Tenant isolation for the TensorRT-LLM providers' semantic donor path.

A donor registered by a request carrying one ``cache_salt`` must never be
visible to a request carrying another, and a salted request must never
consume an unsalted donor. Two requests that both lack a cache_salt DO share
donors — that is the single-tenant deployment and it must keep working.

Both providers are covered: ``SemBlendTensorRTProvider`` (the connector's
provider, driven through ``SemanticKvLookupRequest``) and the exported
``SemBlendProvider`` (TRT-LLM's upstream ``SemanticCacheLookupProvider``
contract, which carries the salt as a keyword). The real semblend_core
pipeline and donor store are used so extra_key propagation is exercised at
both the registration and the lookup call site.
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

EMBED_DIM = 8
BLOCK_SIZE = 32

# Donor and near-identical target: four tokens differ out of 600, which is
# the shape of two tenants sending the same templated prompt with different
# private bodies.
DONOR_TOKENS = list(range(1000, 1600))
TARGET_TOKENS = DONOR_TOKENS[:300] + [7777, 7778, 7779, 7780] + DONOR_TOKENS[304:]

DONOR_TEXT = "quarterly revenue summary for the western region " * 12
TARGET_TEXT = "quarterly revenue summary for the western region " * 12

SALT_A = "tenant-a"
SALT_B = "tenant-b"


@pytest.fixture(autouse=True)
def deterministic_env(monkeypatch):
    monkeypatch.setenv("SEMBLEND_CHUNK_FAST_PATH", "0")
    for name in (
        "SEMBLEND_ENABLED",
        "SEMBLEND_MULTI_DONOR",
        "SEMBLEND_PARAPHRASE_SERVE",
        "SEMBLEND_DONOR_TENANT",
        "SEMBLEND_DONOR_TEMPLATE",
        "SEMBLEND_TRTLLM_EXACT_PREFIX_FAST_PATH",
        "SEMBLEND_TRTLLM_TOKEN_PREFIX_FAST_PATH",
        "SEMBLEND_TRTLLM_FORCE_RECOMPUTE_LAYERS",
        "SEMBLEND_FORCE_RECOMPUTE_LAYERS",
        "SEMBLEND_TRTLLM_ENGINE_BLEND",
    ):
        monkeypatch.delenv(name, raising=False)


def _namespace():
    from semblend.integration.trtllm.namespace import build_cache_namespace

    return build_cache_namespace(
        model="Qwen/Qwen2.5-7B-Instruct",
        tokenizer="Qwen/Qwen2.5-7B-Instruct",
        block_size=BLOCK_SIZE,
        kv_dtype="bfloat16",
        rope_config={"rope_theta": 10000.0},
    )


class _StubEmbedder:
    """Same unit vector for every text: cosine similarity 1.0.

    Isolation must not depend on the prompts being dissimilar — these
    tests make every prompt a perfect semantic match so the only thing
    that can keep the donor away is the namespace.
    """

    dimension = EMBED_DIM

    def embed(self, text: str):
        return np.ones(EMBED_DIM, dtype=np.float32) / np.sqrt(EMBED_DIM)


def _make_pipeline():
    from semblend_core.donor_store import DonorStore
    from semblend_core.pipeline import SemBlendPipeline

    store = DonorStore(
        max_entries=16,
        embedding_dim=EMBED_DIM,
        min_similarity=0.60,
        chunk_size=BLOCK_SIZE,
    )
    pipeline = SemBlendPipeline(
        embedder_type="jaccard",
        donor_store=store,
        chunk_size=BLOCK_SIZE,
        enable_pq_segments=False,
    )
    pipeline._embedder = _StubEmbedder()  # noqa: SLF001
    return pipeline


# ----------------------------------------------------------------------
# SemBlendTensorRTProvider helpers
# ----------------------------------------------------------------------


def _tensorrt_provider():
    from semblend.integration.trtllm.semblend_provider import SemBlendTensorRTProvider

    provider = SemBlendTensorRTProvider(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        chunk_size=BLOCK_SIZE,
        min_match_length=1,
    )
    provider._pipeline = _make_pipeline()  # noqa: SLF001
    return provider


def _register_tensorrt(provider, salt, request_id="donor", token_ids=None):
    return provider.register_completed(
        request_id=request_id,
        token_ids=list(token_ids or DONOR_TOKENS),
        prompt_text=DONOR_TEXT,
        namespace=_namespace(),
        block_ids=list(range(20)),
        cache_salt=salt,
    )


def _lookup_request(salt, token_ids=None):
    from semblend.integration.trtllm.upstream_interface import SemanticKvLookupRequest

    return SemanticKvLookupRequest(
        request_id=7,
        token_ids=tuple(token_ids or TARGET_TOKENS),
        prompt_text=TARGET_TEXT,
        namespace=_namespace(),
        cache_salt=salt,
    )


def _lookup_tensorrt(provider, salt, token_ids=None):
    return provider.lookup(_lookup_request(salt, token_ids))


# ----------------------------------------------------------------------
# SemBlendProvider helpers
# ----------------------------------------------------------------------


def _legacy_provider():
    from semblend.integration.trtllm.semblend_provider import SemBlendProvider

    provider = SemBlendProvider(model_name="Qwen/Qwen2.5-7B-Instruct", chunk_size=BLOCK_SIZE)
    provider._pipeline = _make_pipeline()  # noqa: SLF001
    return provider


def _register_legacy(provider, salt, request_id="donor"):
    provider.register_completed(request_id, list(DONOR_TOKENS), DONOR_TEXT, cache_salt=salt)


def _lookup_legacy(provider, salt, token_ids=None):
    return provider.find_semantic_match(
        list(token_ids or TARGET_TOKENS), TARGET_TEXT, cache_salt=salt
    )


def _pre_fix_namespace_key(namespace) -> str:
    """The key formula before the salt was part of the namespace."""
    payload = json.dumps(namespace.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class TestNamespaceDerivation:
    """The isolation key is a pure function of the namespace and the salt."""

    def test_distinct_salts_give_distinct_keys(self):
        from semblend.integration.trtllm.namespace import bind_cache_salt, namespace_key

        base = _namespace()

        assert namespace_key(bind_cache_salt(base, SALT_A)) != namespace_key(
            bind_cache_salt(base, SALT_B)
        )

    def test_same_salt_gives_same_key(self):
        from semblend.integration.trtllm.namespace import bind_cache_salt, namespace_key

        assert namespace_key(bind_cache_salt(_namespace(), SALT_A)) == namespace_key(
            bind_cache_salt(_namespace(), SALT_A)
        )

    def test_salted_key_differs_from_unsalted_key(self):
        from semblend.integration.trtllm.namespace import bind_cache_salt, namespace_key

        base = _namespace()

        assert namespace_key(bind_cache_salt(base, SALT_A)) != namespace_key(base)

    def test_salt_is_not_stored_verbatim(self):
        from semblend.integration.trtllm.namespace import bind_cache_salt, cache_salt_namespace

        bound = bind_cache_salt(_namespace(), "acme-corp")

        assert "acme-corp" not in json.dumps(bound.to_dict())
        assert "acme-corp" not in cache_salt_namespace("acme-corp")

    @pytest.mark.parametrize("salt", [None, "", "   "])
    def test_absent_salt_maps_to_sentinel(self, salt):
        from semblend.integration.trtllm.namespace import (
            NO_CACHE_SALT_NAMESPACE,
            bind_cache_salt,
            cache_salt_namespace,
            namespace_key,
        )

        base = _namespace()

        assert cache_salt_namespace(salt) == NO_CACHE_SALT_NAMESPACE
        # Absent leaves the namespace object untouched (the wire shape of an
        # unsalted deployment does not change) and keys like the sentinel.
        assert bind_cache_salt(base, salt) == base
        assert namespace_key(bind_cache_salt(base, salt)) == namespace_key(base)

    def test_salted_namespace_never_equals_sentinel(self):
        from semblend.integration.trtllm.namespace import (
            NO_CACHE_SALT_NAMESPACE,
            cache_salt_namespace,
        )

        assert cache_salt_namespace(SALT_A) != NO_CACHE_SALT_NAMESPACE

    def test_binding_does_not_mutate_the_base_namespace(self):
        from semblend.integration.trtllm.namespace import CACHE_SALT_EXTRA_FIELD, bind_cache_salt

        base = _namespace()
        bound = bind_cache_salt(base, SALT_A)

        assert CACHE_SALT_EXTRA_FIELD not in base.extra
        assert CACHE_SALT_EXTRA_FIELD in bound.extra

    def test_hard_fields_still_change_the_key_under_the_same_salt(self):
        from semblend.integration.trtllm.namespace import (
            bind_cache_salt,
            build_cache_namespace,
            namespace_key,
        )

        base = _namespace()
        other_model = build_cache_namespace(model="other", tokenizer="other", block_size=BLOCK_SIZE)

        assert namespace_key(bind_cache_salt(base, SALT_A)) != namespace_key(
            bind_cache_salt(other_model, SALT_A)
        )

    def test_key_is_not_backward_compatible_with_pre_fix_donors(self):
        """Donors keyed before the salt was in the namespace have unknown
        tenancy and must be unreachable after upgrade."""
        from semblend.integration.trtllm.namespace import namespace_key

        base = _namespace()

        assert namespace_key(base) != _pre_fix_namespace_key(base)


class TestTensorRTProviderIsolation:
    """SemBlendTensorRTProvider on its default (pipeline) path."""

    def test_registration_binds_salt_as_extra_key(self):
        from semblend.integration.trtllm.namespace import bind_cache_salt, namespace_key

        provider = _tensorrt_provider()
        _register_tensorrt(provider, SALT_A)

        node = provider._pipeline._donor_store.get_donor("donor")  # noqa: SLF001

        assert node.extra_key == namespace_key(bind_cache_salt(_namespace(), SALT_A))

    def test_registered_event_carries_hashed_salt_not_raw(self):
        from semblend.integration.trtllm.namespace import (
            CACHE_SALT_EXTRA_FIELD,
            cache_salt_namespace,
        )

        provider = _tensorrt_provider()
        event = _register_tensorrt(provider, SALT_A)

        assert event.namespace.extra[CACHE_SALT_EXTRA_FIELD] == cache_salt_namespace(SALT_A)
        assert SALT_A not in json.dumps(event.to_dict())

    def test_unsalted_registration_leaves_the_isolation_key_unstamped(self):
        """Absent salt, absent isolation field — but an explicit tenant key.

        The two fields answer differently on purpose: ``namespace_key``
        defaults the missing isolation field to the sentinel, so an unsalted
        deployment's wire shape does not change, while the tenant key is
        always stamped so a consumer never has to guess whether a namespace
        predates the contract.
        """
        from semblend.integration.dynamo.semantic_events import (
            NO_TENANT_KEY,
            TENANT_KEY_EXTRA_FIELD,
        )
        from semblend.integration.trtllm.namespace import (
            CACHE_SALT_EXTRA_FIELD,
            strip_tenant_key,
        )

        provider = _tensorrt_provider()
        event = _register_tensorrt(provider, None)

        assert CACHE_SALT_EXTRA_FIELD not in event.namespace.extra
        assert event.namespace.extra[TENANT_KEY_EXTRA_FIELD] == NO_TENANT_KEY
        assert strip_tenant_key(event.namespace) == _namespace()

    def test_cross_salt_request_gets_no_donor(self):
        provider = _tensorrt_provider()
        _register_tensorrt(provider, SALT_A)

        result = _lookup_tensorrt(provider, SALT_B)

        assert result.found is False
        assert provider.get_stats()["hits"] == 0
        assert provider.get_stats()["cross_namespace_rejected"] == 1

    def test_same_salt_request_reuses_donor(self):
        provider = _tensorrt_provider()
        _register_tensorrt(provider, SALT_A)

        result = _lookup_tensorrt(provider, SALT_A)

        assert result.found is True
        assert result.plan.donor_ids == ("donor",)
        assert provider.get_stats()["cross_namespace_rejected"] == 0

    def test_unsalted_request_gets_no_salted_donor(self):
        provider = _tensorrt_provider()
        _register_tensorrt(provider, SALT_A)

        result = _lookup_tensorrt(provider, None)

        assert result.found is False
        assert provider.get_stats()["cross_namespace_rejected"] == 1

    def test_salted_request_gets_no_unsalted_donor(self):
        provider = _tensorrt_provider()
        _register_tensorrt(provider, None)

        result = _lookup_tensorrt(provider, SALT_A)

        assert result.found is False
        assert provider.get_stats()["cross_namespace_rejected"] == 1

    def test_unsalted_deployment_still_reuses(self):
        provider = _tensorrt_provider()
        _register_tensorrt(provider, None)

        result = _lookup_tensorrt(provider, None)

        assert result.found is True
        assert result.plan.donor_ids == ("donor",)

    def test_high_similarity_does_not_override_isolation(self):
        """An exact-prompt donor from another tenant is still refused."""
        provider = _tensorrt_provider()
        _register_tensorrt(provider, SALT_A)

        # One token apart: the highest-similarity case short of self-match.
        result = _lookup_tensorrt(provider, SALT_B, token_ids=DONOR_TOKENS[:-1] + [4242])

        assert result.found is False

    def test_every_foreign_donor_is_counted(self):
        provider = _tensorrt_provider()
        _register_tensorrt(provider, SALT_A, request_id="d1")
        _register_tensorrt(provider, SALT_A, request_id="d2", token_ids=DONOR_TOKENS[:-1])

        _lookup_tensorrt(provider, SALT_B)

        assert provider.get_stats()["cross_namespace_rejected"] == 2

    def test_plan_carries_the_salted_namespace(self):
        from semblend.integration.trtllm.namespace import bind_cache_salt

        provider = _tensorrt_provider()
        _register_tensorrt(provider, SALT_A)

        result = _lookup_tensorrt(provider, SALT_A)

        assert result.plan.namespace == bind_cache_salt(_namespace(), SALT_A)


class TestTensorRTProviderFastPaths:
    """The token fast paths scan donor handles directly; they must gate too."""

    @pytest.mark.parametrize(
        "flag",
        [
            "SEMBLEND_TRTLLM_EXACT_PREFIX_FAST_PATH",
            "SEMBLEND_TRTLLM_TOKEN_PREFIX_FAST_PATH",
        ],
    )
    def test_cross_salt_request_gets_no_donor(self, monkeypatch, flag):
        monkeypatch.setenv(flag, "1")
        provider = _tensorrt_provider()
        _register_tensorrt(provider, SALT_A)

        result = _lookup_tensorrt(provider, SALT_B)

        assert result.found is False
        assert provider.get_stats()["cross_namespace_rejected"] == 1

    @pytest.mark.parametrize(
        "flag",
        [
            "SEMBLEND_TRTLLM_EXACT_PREFIX_FAST_PATH",
            "SEMBLEND_TRTLLM_TOKEN_PREFIX_FAST_PATH",
        ],
    )
    def test_same_salt_request_reuses_donor(self, monkeypatch, flag):
        monkeypatch.setenv(flag, "1")
        provider = _tensorrt_provider()
        _register_tensorrt(provider, SALT_A)

        result = _lookup_tensorrt(provider, SALT_A)

        assert result.found is True
        assert result.plan.donor_ids == ("donor",)


class TestTensorRTProviderPlanGateFailsClosed:
    """The handle gate at plan build, for when the store's filter is bypassed."""

    def _foreign_result(self):
        return SimpleNamespace(
            found=True,
            donor_id="donor",
            donor_tokens=list(DONOR_TOKENS),
            similarity=1.0,
            reuse_ratio=1.0,
            slot_actions=[],
            layer_deviations=None,
            position_map=SimpleNamespace(donor_positions=[0, 1], target_positions=[0, 1]),
            confidence_tier="exact",
            chunk_fast_path_used=False,
        )

    def test_handle_from_another_salt_is_not_materializable(self):
        from semblend.integration.trtllm.semblend_provider import _bind_lookup_request

        provider = _tensorrt_provider()
        _register_tensorrt(provider, SALT_A)

        plan = provider._build_plan(  # noqa: SLF001
            _bind_lookup_request(_lookup_request(SALT_B)), self._foreign_result()
        )

        assert plan is None
        assert provider.get_stats()["cross_namespace_rejected"] == 1

    def test_handle_from_same_salt_is_materializable(self):
        from semblend.integration.trtllm.semblend_provider import _bind_lookup_request

        provider = _tensorrt_provider()
        _register_tensorrt(provider, SALT_A)

        plan = provider._build_plan(  # noqa: SLF001
            _bind_lookup_request(_lookup_request(SALT_A)), self._foreign_result()
        )

        assert plan is not None
        assert provider.get_stats()["cross_namespace_rejected"] == 0


class TestLegacyProviderIsolation:
    """SemBlendProvider: the exported SemanticCacheLookupProvider contract."""

    def test_registration_binds_salt_as_extra_key(self):
        from semblend.integration.trtllm.namespace import bind_cache_salt, namespace_key

        provider = _legacy_provider()
        _register_legacy(provider, SALT_A)

        node = provider._pipeline._donor_store.get_donor("donor")  # noqa: SLF001

        assert node.extra_key == namespace_key(bind_cache_salt(provider._namespace, SALT_A))

    def test_unsalted_registration_binds_the_sentinel_key(self):
        """An unsalted donor is keyed, never left with a fail-open None."""
        from semblend.integration.trtllm.namespace import namespace_key

        provider = _legacy_provider()
        _register_legacy(provider, None)

        node = provider._pipeline._donor_store.get_donor("donor")  # noqa: SLF001

        assert node.extra_key == namespace_key(provider._namespace)

    def test_salt_is_not_stored_on_the_donor(self):
        provider = _legacy_provider()
        _register_legacy(provider, "acme-corp")

        node = provider._pipeline._donor_store.get_donor("donor")  # noqa: SLF001

        assert "acme-corp" not in (node.extra_key or "")

    def test_cross_salt_request_gets_no_donor(self):
        provider = _legacy_provider()
        _register_legacy(provider, SALT_A)

        match = _lookup_legacy(provider, SALT_B)

        assert match is None
        assert provider.get_stats()["hits"] == 0
        assert provider.get_stats()["misses"] == 1
        assert provider.get_stats()["cross_namespace_rejected"] == 1

    def test_same_salt_request_reuses_donor(self):
        provider = _legacy_provider()
        _register_legacy(provider, SALT_A)

        match = _lookup_legacy(provider, SALT_A)

        assert match is not None
        assert match.donor_id == "donor"
        assert provider.get_stats()["hits"] == 1
        assert provider.get_stats()["cross_namespace_rejected"] == 0

    def test_unsalted_request_gets_no_salted_donor(self):
        provider = _legacy_provider()
        _register_legacy(provider, SALT_A)

        match = _lookup_legacy(provider, None)

        assert match is None
        assert provider.get_stats()["cross_namespace_rejected"] == 1

    def test_salted_request_gets_no_unsalted_donor(self):
        provider = _legacy_provider()
        _register_legacy(provider, None)

        match = _lookup_legacy(provider, SALT_A)

        assert match is None
        assert provider.get_stats()["cross_namespace_rejected"] == 1

    def test_unsalted_deployment_still_reuses(self):
        provider = _legacy_provider()
        _register_legacy(provider, None)

        match = _lookup_legacy(provider, None)

        assert match is not None
        assert match.donor_id == "donor"

    def test_high_similarity_does_not_override_isolation(self):
        provider = _legacy_provider()
        _register_legacy(provider, SALT_A)

        match = _lookup_legacy(provider, SALT_B, token_ids=DONOR_TOKENS[:-1] + [4242])

        assert match is None

    def test_upstream_contract_without_salt_still_works(self):
        """TRT-LLM calls the two-argument form; that is the unsalted tenant."""
        provider = _legacy_provider()
        provider.register_completed("donor", list(DONOR_TOKENS), DONOR_TEXT)

        match = provider.find_semantic_match(list(TARGET_TOKENS), TARGET_TEXT)

        assert match is not None
        assert match.donor_id == "donor"


class TestLegacyProviderPostCheckFailsClosed:
    """The provider's own gate, for when the store's filter is bypassed."""

    def test_donor_from_another_namespace_is_rejected(self):
        provider = _legacy_provider()
        _register_legacy(provider, SALT_A)

        allowed = provider._namespace_allows(  # noqa: SLF001
            "donor", provider._request_namespace(SALT_B)
        )

        assert allowed is False
        assert provider.get_stats()["cross_namespace_rejected"] == 1

    def test_donor_from_same_namespace_is_allowed(self):
        provider = _legacy_provider()
        _register_legacy(provider, SALT_A)

        allowed = provider._namespace_allows(  # noqa: SLF001
            "donor", provider._request_namespace(SALT_A)
        )

        assert allowed is True
        assert provider.get_stats()["cross_namespace_rejected"] == 0

    def test_unknown_donor_is_rejected(self):
        """Fail closed: a donor whose namespace cannot be read is refused."""
        provider = _legacy_provider()

        allowed = provider._namespace_allows(  # noqa: SLF001
            "never-registered", provider._request_namespace(SALT_A)
        )

        assert allowed is False
        assert provider.get_stats()["cross_namespace_rejected"] == 1


class _RecordingEmitter:
    """Stands in for the contract emitter the pipeline publishes through."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def donor_registered(self, request_id, token_ids, embedding, **kwargs) -> None:
        self.calls.append(dict(kwargs))


class TestTenantKeyThreading:
    """Both providers forward the raw salt, not only the namespace.

    The namespace isolates the donor inside this worker. The tenant key is
    the one identity a fleet router can recompute from the request it is
    placing, and it is derived from the salt alone, so the salt itself has
    to reach the pipeline.
    """

    def test_the_tensorrt_provider_forwards_the_raw_salt(self):
        provider = _tensorrt_provider()
        emitter = _RecordingEmitter()
        provider._pipeline._event_emitter = emitter  # noqa: SLF001

        _register_tensorrt(provider, SALT_A)

        assert emitter.calls[0]["cache_salt"] == SALT_A

    def test_the_legacy_provider_forwards_the_raw_salt(self):
        provider = _legacy_provider()
        emitter = _RecordingEmitter()
        provider._pipeline._event_emitter = emitter  # noqa: SLF001

        _register_legacy(provider, SALT_A)

        assert emitter.calls[0]["cache_salt"] == SALT_A

    def test_an_unsalted_registration_still_reaches_the_emitter_as_a_salt(self):
        """Absent is a real answer, not an unthreaded argument.

        Both providers default ``cache_salt`` to ``None``, so forwarding it
        publishes the sentinel key instead of tripping the "nobody wired the
        salt" warning.
        """
        for provider, register in (
            (_tensorrt_provider(), _register_tensorrt),
            (_legacy_provider(), _register_legacy),
        ):
            emitter = _RecordingEmitter()
            provider._pipeline._event_emitter = emitter  # noqa: SLF001

            register(provider, None)

            assert emitter.calls[0]["cache_salt"] is None

    def test_distinct_salts_give_distinct_published_tenant_keys(self):
        from semblend.integration.dynamo.semantic_events import tenant_key_for_salt

        provider = _tensorrt_provider()
        emitter = _RecordingEmitter()
        provider._pipeline._event_emitter = emitter  # noqa: SLF001

        _register_tensorrt(provider, SALT_A, request_id="donor-a")
        _register_tensorrt(provider, SALT_B, request_id="donor-b")

        keys = [tenant_key_for_salt(c["cache_salt"]) for c in emitter.calls]
        assert keys[0] != keys[1]


class TestTensorRTPublishesTheTenantKey:
    """The TRT-LLM contract publisher stamps the tenant key on the wire.

    A TRT-LLM worker used to publish only the engine-local isolation key, so
    a fleet fed by it fell back to an identity no router can construct and
    none of its donors could be selected by tenant.
    """

    def test_the_published_event_carries_the_tenant_key(self):
        from semblend.integration.dynamo.semantic_events import (
            TENANT_KEY_EXTRA_FIELD,
            tenant_key_for_salt,
        )

        event = _register_tensorrt(_tensorrt_provider(), SALT_A)

        assert event.namespace.extra[TENANT_KEY_EXTRA_FIELD] == tenant_key_for_salt(SALT_A)

    def test_distinct_salts_publish_distinct_tenant_keys(self):
        from semblend.integration.dynamo.semantic_events import TENANT_KEY_EXTRA_FIELD

        provider = _tensorrt_provider()
        a = _register_tensorrt(provider, SALT_A, request_id="donor-a")
        b = _register_tensorrt(provider, SALT_B, request_id="donor-b")

        assert (
            a.namespace.extra[TENANT_KEY_EXTRA_FIELD] != b.namespace.extra[TENANT_KEY_EXTRA_FIELD]
        )

    def test_the_engine_local_key_did_not_move(self):
        """The tenant key is wire-only: it must not enter the donor's key.

        ``namespace_key`` is what the local store is keyed by, so a tenant
        key inside it would strand every donor registered before the change.
        """
        from semblend.integration.trtllm.namespace import (
            bind_cache_salt,
            namespace_key,
        )

        provider = _tensorrt_provider()
        _register_tensorrt(provider, SALT_A)

        expected = namespace_key(bind_cache_salt(_namespace(), SALT_A))
        stored = provider._pipeline._donor_store.get_donor("donor")  # noqa: SLF001
        assert stored.extra_key == expected

    def test_a_handle_rebuilt_from_a_published_event_still_matches_the_request(self):
        """The wire-only key is stripped again when a handle is rebuilt.

        The handle's namespace is compared for equality against the lookup
        request's, so a handle carrying the extra field would reject every
        lookup as cross-namespace.
        """
        from semblend.integration.trtllm.namespace import bind_cache_salt

        provider = _tensorrt_provider()
        event = _register_tensorrt(provider, SALT_A)

        rebuilt = _tensorrt_provider()
        rebuilt.register_donor(event)

        assert rebuilt._donors["donor"].namespace == bind_cache_salt(  # noqa: SLF001
            _namespace(), SALT_A
        )

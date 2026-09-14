"""Cache-salt wiring at the two TensorRT-LLM call sites that drive the providers.

The providers themselves are covered by test_cache_salt_isolation.py. What is
covered here is whether the callers actually hand them the salt:

* ``SemBlendKvConnectorScheduler.request_finished`` — the registration side of
  the connector. A donor registered without the salt lands in the no-cache-salt
  namespace, where every unsalted request can consume it, however carefully the
  lookup side gates.
* ``TRTLLMPyTorchBackend`` — the backend the ``semblend-trtllm`` launcher hook
  drives. Its ``register_donor`` / ``find_semantic_donor`` pair calls
  semblend_core directly, and semblend_core's donor filter is fail-open: an
  absent extra_key means every donor is visible.

Both are exercised against the real semblend_core pipeline and donor store, so
extra_key propagation is checked where it actually has to hold, and a salted
donor must never be reachable from an unsalted request (or from another salt)
while two unsalted requests still share donors as before.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

EMBED_DIM = 8
BLOCK_SIZE = 32
MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Donor and near-identical target: four tokens differ out of 600, the shape of
# two tenants sending the same templated prompt with different private bodies.
DONOR_TOKENS = list(range(1000, 1600))
TARGET_TOKENS = DONOR_TOKENS[:300] + [7777, 7778, 7779, 7780] + DONOR_TOKENS[304:]

DONOR_TEXT = "quarterly revenue summary for the western region " * 12
TARGET_TEXT = "quarterly revenue summary for the western region " * 12

SALT_A = "tenant-a"
SALT_B = "tenant-b"


@pytest.fixture(autouse=True)
def deterministic_env(monkeypatch):
    monkeypatch.setenv("SEMBLEND_CHUNK_FAST_PATH", "0")
    monkeypatch.setenv("SEMBLEND_MODEL_NAME", MODEL)
    monkeypatch.setenv("SEMBLEND_TRTLLM_BLOCK_SIZE", str(BLOCK_SIZE))
    for name in (
        "SEMBLEND_ENABLED",
        "SEMBLEND_MULTI_DONOR",
        "SEMBLEND_PARAPHRASE_SERVE",
        "SEMBLEND_DONOR_TENANT",
        "SEMBLEND_DONOR_TEMPLATE",
        "SEMBLEND_MODEL_REVISION",
        "SEMBLEND_TOKENIZER_REVISION",
        "SEMBLEND_ADAPTER_ID",
        "SEMBLEND_TRTLLM_ROPE_BASE",
        "SEMBLEND_TRTLLM_AUDIT_PATH",
        "SEMBLEND_TRTLLM_ENABLE_SEGMENTED",
        "SEMBLEND_TRTLLM_ENABLE_MATERIALIZATION",
        "SEMBLEND_TRTLLM_EXACT_PREFIX_FAST_PATH",
        "SEMBLEND_TRTLLM_TOKEN_PREFIX_FAST_PATH",
        "SEMBLEND_TRTLLM_ENGINE_BLEND",
        "SEMBLEND_NATS_URL",
    ):
        monkeypatch.delenv(name, raising=False)


class _StubEmbedder:
    """Same unit vector for every text: cosine similarity 1.0.

    Isolation must not depend on the prompts being dissimilar — every prompt
    here is a perfect semantic match, so the only thing that can keep a donor
    away from a request is the namespace.
    """

    dimension = EMBED_DIM

    def embed(self, text: str):
        return np.ones(EMBED_DIM, dtype=np.float32) / np.sqrt(EMBED_DIM)


class _StubTokenizer:
    """Constant decode — the donor text never carries the isolation signal."""

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        del token_ids, skip_special_tokens
        return DONOR_TEXT


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


def _donor_node(pipeline, donor_id: str):
    return pipeline._donor_store.get_donor(donor_id)  # noqa: SLF001


# ----------------------------------------------------------------------
# Connector scheduler
# ----------------------------------------------------------------------


class _FakeRequest:
    """The attributes the scheduler reads off a TRT-LLM request."""

    def __init__(self, request_id: int, token_ids, prompt: str, cache_salt=None) -> None:
        self.request_id = request_id
        self.prompt = prompt
        self.cache_salt = cache_salt
        self.block_hashes = ()
        self._token_ids = list(token_ids)

    def get_tokens(self, beam: int):
        del beam
        return list(self._token_ids)


def _scheduler():
    from semblend.integration.trtllm.connector import SemBlendKvConnectorScheduler

    scheduler = SemBlendKvConnectorScheduler(llm_args=None)
    scheduler._provider._pipeline = _make_pipeline()  # noqa: SLF001
    return scheduler


def _finish(scheduler, salt, request_id: int = 1, token_ids=None):
    request = _FakeRequest(request_id, token_ids or DONOR_TOKENS, DONOR_TEXT, salt)
    scheduler.request_finished(request, list(range(20)))
    return request


def _lookup(scheduler, salt, request_id: int = 7, token_ids=None):
    request = _FakeRequest(request_id, token_ids or TARGET_TOKENS, TARGET_TEXT, salt)
    _, _, result = scheduler.get_num_new_matched_tokens_with_metadata(request, 0)
    return result


class TestSchedulerDonorRegistration:
    """request_finished must key the donor by the finishing request's salt."""

    def test_salted_donor_is_registered_under_the_salted_namespace(self):
        from semblend.integration.trtllm.namespace import bind_cache_salt, namespace_key

        scheduler = _scheduler()
        _finish(scheduler, SALT_A)

        node = _donor_node(scheduler._provider._pipeline, "1")  # noqa: SLF001

        assert node is not None
        assert node.extra_key == namespace_key(bind_cache_salt(scheduler._namespace, SALT_A))
        assert node.extra_key != namespace_key(scheduler._namespace)

    def test_unsalted_donor_is_registered_under_the_sentinel_namespace(self):
        from semblend.integration.trtllm.namespace import namespace_key

        scheduler = _scheduler()
        _finish(scheduler, None)

        node = _donor_node(scheduler._provider._pipeline, "1")  # noqa: SLF001

        assert node.extra_key == namespace_key(scheduler._namespace)

    def test_raw_salt_never_reaches_the_donor_record_or_the_event(self):
        scheduler = _scheduler()
        _finish(scheduler, "acme-corp")

        node = _donor_node(scheduler._provider._pipeline, "1")  # noqa: SLF001

        assert "acme-corp" not in (node.extra_key or "")
        assert "acme-corp" not in json.dumps(scheduler._events[0].data.to_dict())  # noqa: SLF001

    def test_published_event_carries_the_hashed_salt(self):
        from semblend.integration.trtllm.namespace import (
            CACHE_SALT_EXTRA_FIELD,
            cache_salt_namespace,
        )

        scheduler = _scheduler()
        _finish(scheduler, SALT_A)

        event = scheduler._events[0].data  # noqa: SLF001

        assert event.namespace.extra[CACHE_SALT_EXTRA_FIELD] == cache_salt_namespace(SALT_A)


class TestSchedulerCrossTenantReuse:
    """The registration side and the lookup side must agree on the key."""

    def test_salted_donor_is_not_served_to_an_unsalted_request(self):
        scheduler = _scheduler()
        _finish(scheduler, SALT_A)

        result = _lookup(scheduler, None)

        assert result.found is False
        assert scheduler._provider.get_stats()["hits"] == 0  # noqa: SLF001
        assert scheduler._provider.get_stats()["cross_namespace_rejected"] == 1  # noqa: SLF001

    def test_salted_donor_is_not_served_to_another_salt(self):
        scheduler = _scheduler()
        _finish(scheduler, SALT_A)

        result = _lookup(scheduler, SALT_B)

        assert result.found is False
        assert scheduler._provider.get_stats()["cross_namespace_rejected"] == 1  # noqa: SLF001

    def test_unsalted_donor_is_not_served_to_a_salted_request(self):
        scheduler = _scheduler()
        _finish(scheduler, None)

        result = _lookup(scheduler, SALT_A)

        assert result.found is False

    def test_same_salt_request_reuses_the_donor(self):
        scheduler = _scheduler()
        _finish(scheduler, SALT_A)

        result = _lookup(scheduler, SALT_A)

        assert result.found is True
        assert result.plan.donor_ids == ("1",)
        assert scheduler._provider.get_stats()["cross_namespace_rejected"] == 0  # noqa: SLF001

    def test_unsalted_deployment_still_reuses(self):
        scheduler = _scheduler()
        _finish(scheduler, None)

        result = _lookup(scheduler, None)

        assert result.found is True
        assert result.plan.donor_ids == ("1",)

    def test_every_foreign_donor_is_counted(self):
        scheduler = _scheduler()
        _finish(scheduler, SALT_A, request_id=1)
        _finish(scheduler, SALT_A, request_id=2, token_ids=DONOR_TOKENS[:-1])

        _lookup(scheduler, SALT_B)

        assert scheduler._provider.get_stats()["cross_namespace_rejected"] == 2  # noqa: SLF001


# ----------------------------------------------------------------------
# PyTorch backend (semblend-trtllm launcher hook path)
# ----------------------------------------------------------------------


def _backend():
    from semblend.integration.trtllm.pytorch_backend import TRTLLMPyTorchBackend

    backend = TRTLLMPyTorchBackend(
        kv_cache_manager=None,
        model_config={
            "model_name": MODEL,
            "tokens_per_block": BLOCK_SIZE,
            "num_layers": 4,
            "num_kv_heads": 2,
            "head_dim": 8,
        },
        tokenizer=_StubTokenizer(),
    )
    backend._pipeline = _make_pipeline()  # noqa: SLF001
    return backend


def _register(backend, salt, request_id: str = "donor", token_ids=None):
    backend.register_donor(
        request_id,
        list(token_ids or DONOR_TOKENS),
        {},
        cache_salt=salt,
    )


def _find(backend, salt, token_ids=None):
    return backend.find_semantic_donor(
        list(token_ids or TARGET_TOKENS),
        TARGET_TEXT,
        cache_salt=salt,
    )


class TestBackendRegistration:
    """register_donor must key the donor, never leave it fail-open."""

    def test_salted_donor_is_registered_under_the_salted_namespace(self):
        from semblend.integration.trtllm.namespace import bind_cache_salt, namespace_key

        backend = _backend()
        _register(backend, SALT_A)

        node = _donor_node(backend._pipeline, "donor")  # noqa: SLF001

        assert node.extra_key == namespace_key(bind_cache_salt(backend._namespace, SALT_A))

    def test_unsalted_donor_is_keyed_with_the_sentinel_not_none(self):
        """An absent salt must still produce a key: extra_key=None is visible
        to every namespace in semblend_core's donor filter."""
        from semblend.integration.trtllm.namespace import namespace_key

        backend = _backend()
        _register(backend, None)

        node = _donor_node(backend._pipeline, "donor")  # noqa: SLF001

        assert node.extra_key == namespace_key(backend._namespace)
        assert node.extra_key is not None

    def test_salt_arrives_through_kv_metadata(self):
        from semblend.integration.trtllm.namespace import bind_cache_salt, namespace_key

        backend = _backend()
        backend.register_donor("donor", list(DONOR_TOKENS), {"cache_salt": SALT_A})

        node = _donor_node(backend._pipeline, "donor")  # noqa: SLF001

        assert node.extra_key == namespace_key(bind_cache_salt(backend._namespace, SALT_A))

    def test_explicit_salt_wins_over_metadata(self):
        from semblend.integration.trtllm.namespace import bind_cache_salt, namespace_key

        backend = _backend()
        backend.register_donor(
            "donor",
            list(DONOR_TOKENS),
            {"cache_salt": SALT_B},
            cache_salt=SALT_A,
        )

        node = _donor_node(backend._pipeline, "donor")  # noqa: SLF001

        assert node.extra_key == namespace_key(bind_cache_salt(backend._namespace, SALT_A))

    def test_salt_is_not_stored_verbatim(self):
        backend = _backend()
        _register(backend, "acme-corp")

        node = _donor_node(backend._pipeline, "donor")  # noqa: SLF001

        assert "acme-corp" not in (node.extra_key or "")


class TestBackendLookup:
    """find_semantic_donor must filter on the same key it registers under."""

    def test_cross_salt_request_gets_no_donor(self):
        backend = _backend()
        _register(backend, SALT_A)

        assert _find(backend, SALT_B) is None
        assert backend.get_stats()["semantic_hits"] == 0
        assert backend.get_stats()["cross_namespace_rejected"] == 1

    def test_same_salt_request_reuses_donor(self):
        backend = _backend()
        _register(backend, SALT_A)

        donor = _find(backend, SALT_A)

        assert donor is not None
        assert donor["donor_id"] == "donor"
        assert backend.get_stats()["cross_namespace_rejected"] == 0

    def test_unsalted_request_gets_no_salted_donor(self):
        backend = _backend()
        _register(backend, SALT_A)

        assert _find(backend, None) is None
        assert backend.get_stats()["cross_namespace_rejected"] == 1

    def test_salted_request_gets_no_unsalted_donor(self):
        backend = _backend()
        _register(backend, None)

        assert _find(backend, SALT_A) is None
        assert backend.get_stats()["cross_namespace_rejected"] == 1

    def test_unsalted_deployment_still_reuses(self):
        backend = _backend()
        _register(backend, None)

        donor = _find(backend, None)

        assert donor is not None
        assert donor["donor_id"] == "donor"

    def test_launcher_hook_call_shape_still_works(self):
        """model_engine_hook calls find_semantic_donor(token_ids) positionally;
        that caller has no salt, so it may only see unsalted donors."""
        backend = _backend()
        _register(backend, None)

        assert backend.find_semantic_donor(list(TARGET_TOKENS)) is not None

    def test_launcher_hook_call_shape_sees_no_salted_donor(self):
        """Both call shapes here predate the fix — a three-positional-argument
        registration carrying the salt in its metadata, and the hook's
        two-argument lookup — so this is the leak itself, not a signature."""
        backend = _backend()
        backend.register_donor("donor", list(DONOR_TOKENS), {"cache_salt": SALT_A})

        assert backend.find_semantic_donor(list(TARGET_TOKENS), TARGET_TEXT) is None
        assert backend.get_stats()["cross_namespace_rejected"] == 1

    def test_high_similarity_does_not_override_isolation(self):
        """A one-token-apart donor from another tenant is still refused."""
        backend = _backend()
        _register(backend, SALT_A)

        assert _find(backend, SALT_B, token_ids=DONOR_TOKENS[:-1] + [4242]) is None

    def test_every_foreign_donor_is_counted(self):
        backend = _backend()
        _register(backend, SALT_A, request_id="d1")
        _register(backend, SALT_A, request_id="d2", token_ids=DONOR_TOKENS[:-1])

        _find(backend, SALT_B)

        assert backend.get_stats()["cross_namespace_rejected"] == 2

    def test_miss_with_no_donors_counts_nothing(self):
        backend = _backend()

        assert _find(backend, SALT_A) is None
        assert backend.get_stats()["cross_namespace_rejected"] == 0


class TestBackendPostCheckFailsClosed:
    """The backend's own gate, for when the store's filter is bypassed."""

    def test_donor_from_another_namespace_is_rejected(self):
        backend = _backend()
        _register(backend, SALT_A)

        allowed = backend._namespace_allows(  # noqa: SLF001
            "donor", backend._request_namespace(SALT_B)
        )

        assert allowed is False
        assert backend.get_stats()["cross_namespace_rejected"] == 1

    def test_donor_from_same_namespace_is_allowed(self):
        backend = _backend()
        _register(backend, SALT_A)

        allowed = backend._namespace_allows(  # noqa: SLF001
            "donor", backend._request_namespace(SALT_A)
        )

        assert allowed is True
        assert backend.get_stats()["cross_namespace_rejected"] == 0

    def test_unknown_donor_is_rejected(self):
        """Fail closed: a donor whose namespace cannot be read is refused."""
        backend = _backend()

        allowed = backend._namespace_allows(  # noqa: SLF001
            "never-registered", backend._request_namespace(SALT_A)
        )

        assert allowed is False
        assert backend.get_stats()["cross_namespace_rejected"] == 1

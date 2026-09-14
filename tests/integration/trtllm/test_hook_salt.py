"""Cache-salt wiring on the ``semblend-trtllm`` launcher hook path.

``model_engine_hook`` is the TensorRT-LLM path that drives
``TRTLLMPyTorchBackend`` directly, and semblend_core's donor filter is
fail-open on a missing key: a lookup that passes no namespace sees every
donor in the store. So both of the hook's lookup sites (token substitution
and the radix fallback) and its donor registration have to carry the
request's own cache_salt, and the console-script wiring has to hook up all
three or the shipped launcher isolates on one side of a pair only.

Exercised against the real semblend_core pipeline and donor store, with an
embedder that returns the same vector for every text: every prompt here is a
perfect semantic match, so the only thing that can keep a donor away from a
request is the namespace.

Also covers the operator signal: a process that registers donor after donor
without a salt ever arriving warns exactly once, naming the component and
where it reads the salt from.
"""

from __future__ import annotations

import argparse
import logging
from types import SimpleNamespace

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

SALT_A = "tenant-a"
SALT_B = "tenant-b"

WARN_AFTER_ENV = "SEMBLEND_UNSALTED_DONOR_WARN_AFTER"


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
        "SEMBLEND_TRTLLM_APPROACH",
        "SEMBLEND_TRTLLM_ROPE_BASE",
        "SEMBLEND_TRTLLM_AUDIT_PATH",
        WARN_AFTER_ENV,
    ):
        monkeypatch.delenv(name, raising=False)


# ----------------------------------------------------------------------
# Pipeline doubles
# ----------------------------------------------------------------------


class _StubEmbedder:
    """Same unit vector for every text: cosine similarity 1.0."""

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


def _donor_node(backend, donor_id: str):
    return backend._pipeline._donor_store.get_donor(donor_id)  # noqa: SLF001


def _salted_key(backend, salt):
    from semblend.integration.trtllm.namespace import bind_cache_salt, namespace_key

    return namespace_key(bind_cache_salt(backend._namespace, salt))  # noqa: SLF001


# ----------------------------------------------------------------------
# Engine doubles
# ----------------------------------------------------------------------


class _FakeRequest:
    """The attributes the hook reads off a TensorRT-LLM request."""

    def __init__(self, request_id: str, token_ids, cache_salt=None) -> None:
        self.request_id = request_id
        self.token_ids = list(token_ids)
        self.cache_salt = cache_salt


class _FakeEngine:
    """Engine exposing the enqueue and completion APIs the hook patches."""

    def __init__(self) -> None:
        self.enqueued: list = []
        self.finished: list = []

    def enqueue_request(self, request, *args, **kwargs):
        del args, kwargs
        self.enqueued.append(request)
        return "enqueued"

    def finish_request(self, request, *args, **kwargs):
        del args, kwargs
        self.finished.append(request)
        return "finished"


class _FakeTree:
    """Radix tree that only matches the donor's exact token sequence."""

    def __init__(self, donor_tokens) -> None:
        self._donor = list(donor_tokens)
        self.keys: list = []

    def match_prefix(self, key, *args, **kwargs):
        del args, kwargs
        self.keys.append(key)
        return len(self._donor) if list(key) == self._donor else 0


class _FakeRadixEngine:
    """Engine with no enqueue API, so the hook takes the radix-patch path."""

    def __init__(self, tree) -> None:
        self.kv_cache_manager = SimpleNamespace(radix_tree=tree)
        self.finished: list = []

    def finish_request(self, request, *args, **kwargs):
        del args, kwargs
        self.finished.append(request)
        return "finished"


class _SaltedKey(list):
    """Token sequence that also carries the isolation fields, as newer
    TensorRT-LLM/SGLang-style key wrappers do."""

    def __init__(self, token_ids, cache_salt=None, req=None) -> None:
        super().__init__(token_ids)
        self.cache_salt = cache_salt
        self.req = req


def _hook(engine=None, backend=None):
    from semblend.integration.trtllm.model_engine_hook import SemBlendModelEngineHook

    engine = _FakeEngine() if engine is None else engine
    backend = _backend() if backend is None else backend
    hook = SemBlendModelEngineHook(engine=engine, backend=backend)
    hook.wrap()
    return hook, engine, backend


def _enqueue(engine, salt, request_id: str = "target", token_ids=None):
    request = _FakeRequest(request_id, token_ids or TARGET_TOKENS, salt)
    engine.enqueue_request(request)
    return request


# ----------------------------------------------------------------------
# Token substitution (approach B)
# ----------------------------------------------------------------------


class TestHookLookupSalt:
    """The intercepted request's salt must reach find_semantic_donor."""

    def test_two_salted_tenants_do_not_share_a_donor(self):
        hook, engine, backend = _hook()
        engine.finish_request(_FakeRequest("donor", DONOR_TOKENS, SALT_A))

        request = _enqueue(engine, SALT_B)

        assert request.token_ids == list(TARGET_TOKENS)
        assert hook.get_stats()["substitutions_applied"] == 0
        assert backend.get_stats()["cross_namespace_rejected"] == 1

    def test_same_salt_request_reuses_the_donor(self):
        hook, engine, backend = _hook()
        engine.finish_request(_FakeRequest("donor", DONOR_TOKENS, SALT_A))

        request = _enqueue(engine, SALT_A)

        assert request.token_ids == list(DONOR_TOKENS)
        assert hook.get_stats()["substitutions_applied"] == 1
        assert backend.get_stats()["cross_namespace_rejected"] == 0

    def test_unsalted_request_gets_no_salted_donor(self):
        hook, engine, _ = _hook()
        engine.finish_request(_FakeRequest("donor", DONOR_TOKENS, SALT_A))

        request = _enqueue(engine, None)

        assert request.token_ids == list(TARGET_TOKENS)
        assert hook.get_stats()["substitutions_applied"] == 0

    def test_salted_request_gets_no_unsalted_donor(self):
        """The leak itself: the donor is registered straight on the backend,
        so this holds whether or not the hook wires registration. A lookup
        that passes no salt keys into the sentinel namespace and consumes
        whatever an unsalted caller left there."""
        hook, engine, backend = _hook()
        backend.register_donor("donor", list(DONOR_TOKENS), {})

        request = _enqueue(engine, SALT_A)

        assert request.token_ids == list(TARGET_TOKENS)
        assert hook.get_stats()["substitutions_applied"] == 0

    def test_unsalted_deployment_still_substitutes(self):
        """Single-tenant reuse must not regress: two unsalted requests share."""
        hook, engine, _ = _hook()
        engine.finish_request(_FakeRequest("donor", DONOR_TOKENS, None))

        request = _enqueue(engine, None)

        assert request.token_ids == list(DONOR_TOKENS)
        assert hook.get_stats()["substitutions_applied"] == 1

    def test_enqueue_result_is_passed_through(self):
        _, engine, _ = _hook()

        assert engine.enqueue_request(_FakeRequest("t", TARGET_TOKENS, SALT_A)) == "enqueued"


# ----------------------------------------------------------------------
# Donor registration
# ----------------------------------------------------------------------


class TestHookDonorRegistration:
    """A finished request must become a donor under its OWN salt."""

    def test_finished_request_is_keyed_by_its_salt(self):
        hook, engine, backend = _hook()

        engine.finish_request(_FakeRequest("donor", DONOR_TOKENS, SALT_A))

        node = _donor_node(backend, "donor")
        assert node is not None
        assert node.extra_key == _salted_key(backend, SALT_A)
        assert node.extra_key != _salted_key(backend, None)
        assert hook.get_stats()["donors_registered"] == 1

    def test_unsalted_registration_is_keyed_with_the_sentinel_not_none(self):
        """extra_key=None is visible to every namespace in the donor filter."""
        _, engine, backend = _hook()

        engine.finish_request(_FakeRequest("donor", DONOR_TOKENS, None))

        node = _donor_node(backend, "donor")
        assert node.extra_key == _salted_key(backend, None)
        assert node.extra_key is not None

    def test_salt_is_not_stored_verbatim(self):
        _, engine, backend = _hook()

        engine.finish_request(_FakeRequest("donor", DONOR_TOKENS, "acme-corp"))

        assert "acme-corp" not in (_donor_node(backend, "donor").extra_key or "")

    def test_completion_result_is_passed_through(self):
        _, engine, _ = _hook()

        assert engine.finish_request(_FakeRequest("donor", DONOR_TOKENS, SALT_A)) == "finished"
        assert len(engine.finished) == 1

    def test_unwrap_restores_the_completion_api(self):
        hook, engine, backend = _hook()

        hook.unwrap()
        engine.finish_request(_FakeRequest("donor", DONOR_TOKENS, SALT_A))

        assert _donor_node(backend, "donor") is None


# ----------------------------------------------------------------------
# Radix fallback (approach A)
# ----------------------------------------------------------------------


def _radix_hook():
    tree = _FakeTree(DONOR_TOKENS)
    hook, engine, backend = _hook(engine=_FakeRadixEngine(tree))
    assert hook.active_approach == "radix_patch"
    return hook, engine, backend, tree


class TestRadixFallbackSalt:
    """The salt must be derived from whatever match_prefix is handed."""

    def test_cross_salt_key_gets_no_donor(self):
        hook, engine, backend, tree = _radix_hook()
        engine.finish_request(_FakeRequest("donor", DONOR_TOKENS, SALT_A))

        assert tree.match_prefix(_SaltedKey(TARGET_TOKENS, SALT_B)) == 0
        assert hook.get_stats()["substitutions_applied"] == 0
        assert backend.get_stats()["cross_namespace_rejected"] == 1

    def test_same_salt_key_reaches_the_donor(self):
        hook, engine, _, tree = _radix_hook()
        engine.finish_request(_FakeRequest("donor", DONOR_TOKENS, SALT_A))

        assert tree.match_prefix(_SaltedKey(TARGET_TOKENS, SALT_A)) == len(DONOR_TOKENS)
        assert hook.get_stats()["substitutions_applied"] == 1

    def test_salt_on_the_request_behind_the_key(self):
        """An outer key without the field must not mask the req that has it."""
        hook, engine, _, tree = _radix_hook()
        engine.finish_request(_FakeRequest("donor", DONOR_TOKENS, SALT_A))

        key = _SaltedKey(TARGET_TOKENS, None, req=_FakeRequest("t", TARGET_TOKENS, SALT_A))

        assert tree.match_prefix(key) == len(DONOR_TOKENS)
        assert hook.get_stats()["substitutions_applied"] == 1

    def test_request_behind_the_key_cannot_cross_salts(self):
        hook, engine, _, tree = _radix_hook()
        engine.finish_request(_FakeRequest("donor", DONOR_TOKENS, SALT_A))

        key = _SaltedKey(TARGET_TOKENS, None, req=_FakeRequest("t", TARGET_TOKENS, SALT_B))

        assert tree.match_prefix(key) == 0
        assert hook.get_stats()["substitutions_applied"] == 0

    def test_salted_key_gets_no_unsalted_donor(self):
        """Same leak on the fallback path, with the donor registered straight
        on the backend: a match_prefix whose salt is never read keys into the
        sentinel namespace and reaches another caller's KV."""
        hook, _, backend, tree = _radix_hook()
        backend.register_donor("donor", list(DONOR_TOKENS), {})

        assert tree.match_prefix(_SaltedKey(TARGET_TOKENS, SALT_A)) == 0
        assert hook.get_stats()["substitutions_applied"] == 0

    def test_bare_token_list_sees_no_salted_donor(self):
        """A key carrying nothing lands in the sentinel namespace."""
        hook, engine, _, tree = _radix_hook()
        engine.finish_request(_FakeRequest("donor", DONOR_TOKENS, SALT_A))

        assert tree.match_prefix(list(TARGET_TOKENS)) == 0
        assert hook.get_stats()["substitutions_applied"] == 0

    def test_unsalted_deployment_still_falls_back(self):
        hook, engine, _, tree = _radix_hook()
        engine.finish_request(_FakeRequest("donor", DONOR_TOKENS, None))

        assert tree.match_prefix(list(TARGET_TOKENS)) == len(DONOR_TOKENS)
        assert hook.get_stats()["substitutions_applied"] == 1


# ----------------------------------------------------------------------
# Console-script wiring (semblend-trtllm)
# ----------------------------------------------------------------------


class _FakeLauncherEngine(_FakeEngine):
    """What _attach_semblend discovers: an engine owning a KV manager."""

    def __init__(self) -> None:
        super().__init__()
        self.kv_cache_manager = SimpleNamespace()


def _attached():
    from semblend.integration.trtllm.launch_semblend_trtllm import _attach_semblend

    engine = _FakeLauncherEngine()
    args = argparse.Namespace(tokens_per_block=BLOCK_SIZE, model=MODEL)
    hook = _attach_semblend(SimpleNamespace(_engine=engine), args)

    hook.backend._tokenizer = _StubTokenizer()  # noqa: SLF001
    hook.backend._pipeline = _make_pipeline()  # noqa: SLF001
    return hook, engine


class TestConsoleScriptWiring:
    """The shipped launcher must isolate on both sides, not just at lookup."""

    def test_launcher_registers_donors_under_the_salt(self):
        hook, engine = _attached()

        engine.finish_request(_FakeRequest("donor", DONOR_TOKENS, SALT_A))

        node = _donor_node(hook.backend, "donor")
        assert node is not None
        assert node.extra_key == _salted_key(hook.backend, SALT_A)

    def test_launcher_wiring_isolates_end_to_end(self):
        hook, engine = _attached()
        engine.finish_request(_FakeRequest("donor", DONOR_TOKENS, SALT_A))

        other = _enqueue(engine, SALT_B, request_id="b")
        same = _enqueue(engine, SALT_A, request_id="a")

        assert other.token_ids == list(TARGET_TOKENS)
        assert same.token_ids == list(DONOR_TOKENS)
        assert hook.get_stats()["substitutions_applied"] == 1


# ----------------------------------------------------------------------
# Operator signal: a process where no salt ever arrives
# ----------------------------------------------------------------------


def _isolation_warnings(caplog):
    return [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING and "cache_salt" in record.getMessage()
    ]


def _register_unsalted(register, count: int) -> None:
    for index in range(count):
        register(f"donor-{index}")


class TestUnsaltedDonorWarning:
    """One warning per component, naming it and where it reads the salt."""

    def test_backend_warns_exactly_once(self, monkeypatch, caplog):
        monkeypatch.setenv(WARN_AFTER_ENV, "2")
        backend = _backend()

        with caplog.at_level(logging.WARNING):
            _register_unsalted(
                lambda donor_id: backend.register_donor(donor_id, list(DONOR_TOKENS), {}),
                5,
            )

        warnings = _isolation_warnings(caplog)
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "TRTLLMPyTorchBackend" in message
        assert "kv_metadata['cache_salt']" in message
        assert WARN_AFTER_ENV in message

    def test_backend_exposes_the_signal_as_a_stat(self, monkeypatch):
        monkeypatch.setenv(WARN_AFTER_ENV, "2")
        backend = _backend()

        _register_unsalted(
            lambda donor_id: backend.register_donor(donor_id, list(DONOR_TOKENS), {}),
            3,
        )

        stats = backend.get_stats()
        assert stats["unsalted_donors_registered"] == 3
        assert stats["salted_donors_registered"] == 0
        assert stats["unsalted_namespace_warned"] is True

    def test_backend_stays_quiet_when_any_request_carries_a_salt(self, monkeypatch, caplog):
        monkeypatch.setenv(WARN_AFTER_ENV, "2")
        backend = _backend()

        with caplog.at_level(logging.WARNING):
            backend.register_donor("salted", list(DONOR_TOKENS), {}, cache_salt=SALT_A)
            _register_unsalted(
                lambda donor_id: backend.register_donor(donor_id, list(DONOR_TOKENS), {}),
                5,
            )

        assert _isolation_warnings(caplog) == []
        assert backend.get_stats()["unsalted_namespace_warned"] is False

    def test_backend_stays_quiet_below_the_threshold(self, caplog):
        backend = _backend()

        with caplog.at_level(logging.WARNING):
            _register_unsalted(
                lambda donor_id: backend.register_donor(donor_id, list(DONOR_TOKENS), {}),
                3,
            )

        assert _isolation_warnings(caplog) == []

    def test_connector_provider_warns_exactly_once(self, monkeypatch, caplog):
        from semblend.integration.trtllm.namespace import build_cache_namespace
        from semblend.integration.trtllm.semblend_provider import SemBlendTensorRTProvider

        monkeypatch.setenv(WARN_AFTER_ENV, "2")
        provider = SemBlendTensorRTProvider(model_name=MODEL, chunk_size=BLOCK_SIZE)
        provider._pipeline = _make_pipeline()  # noqa: SLF001
        namespace = build_cache_namespace(model=MODEL, block_size=BLOCK_SIZE)

        with caplog.at_level(logging.WARNING):
            _register_unsalted(
                lambda donor_id: provider.register_completed(
                    request_id=donor_id,
                    token_ids=list(DONOR_TOKENS),
                    prompt_text=DONOR_TEXT,
                    namespace=namespace,
                    block_ids=[1, 2],
                ),
                5,
            )

        warnings = _isolation_warnings(caplog)
        assert len(warnings) == 1
        assert "SemBlendTensorRTProvider" in warnings[0].getMessage()
        assert "kv_connector_config" in warnings[0].getMessage()
        assert provider.get_stats()["unsalted_namespace_warned"] is True

    def test_connector_provider_stays_quiet_when_a_salt_arrives(self, monkeypatch, caplog):
        from semblend.integration.trtllm.namespace import build_cache_namespace
        from semblend.integration.trtllm.semblend_provider import SemBlendTensorRTProvider

        monkeypatch.setenv(WARN_AFTER_ENV, "2")
        provider = SemBlendTensorRTProvider(model_name=MODEL, chunk_size=BLOCK_SIZE)
        provider._pipeline = _make_pipeline()  # noqa: SLF001
        namespace = build_cache_namespace(model=MODEL, block_size=BLOCK_SIZE)

        with caplog.at_level(logging.WARNING):
            provider.register_completed(
                request_id="salted",
                token_ids=list(DONOR_TOKENS),
                prompt_text=DONOR_TEXT,
                namespace=namespace,
                block_ids=[1, 2],
                cache_salt=SALT_A,
            )
            _register_unsalted(
                lambda donor_id: provider.register_completed(
                    request_id=donor_id,
                    token_ids=list(DONOR_TOKENS),
                    prompt_text=DONOR_TEXT,
                    namespace=namespace,
                    block_ids=[1, 2],
                ),
                5,
            )

        assert _isolation_warnings(caplog) == []
        assert provider.get_stats()["salted_donors_registered"] == 1

    def test_lookup_provider_warns_exactly_once(self, monkeypatch, caplog):
        from semblend.integration.trtllm.semblend_provider import SemBlendProvider

        monkeypatch.setenv(WARN_AFTER_ENV, "2")
        provider = SemBlendProvider(model_name=MODEL, chunk_size=BLOCK_SIZE)
        provider._pipeline = _make_pipeline()  # noqa: SLF001

        with caplog.at_level(logging.WARNING):
            _register_unsalted(
                lambda donor_id: provider.register_completed(
                    donor_id,
                    list(DONOR_TOKENS),
                    DONOR_TEXT,
                ),
                5,
            )

        warnings = _isolation_warnings(caplog)
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "SemBlendProvider" in message
        assert "find_semantic_match()" in message
        assert provider.get_stats()["unsalted_donors_registered"] == 5

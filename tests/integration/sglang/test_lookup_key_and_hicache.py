"""Isolated donor lookups and fail-closed HiCache lookups for SGLang.

Two holes this covers, both of which let the core donor filter run
fail-open (it treats an absent key as "every donor is visible"):

1. The RadixCache subclass looked its chosen donor up in the tree with a
   bare token list, dropping the RadixKey — and with it the extra_key —
   that the request arrived with. On builds whose match_prefix expects a
   RadixKey that raises, so the semantic path is inert; on builds that
   partition the tree by extra_key it queries outside the requesting
   tenant's partition. The donor lookup now rebuilds the request's own key
   shape around the donor's tokens, and declines when it cannot.

2. The HiCacheStorage backend derived its namespace only from extra_info.
   On an SGLang build whose HiCacheStorage never passes extra_info, a
   salted request's entry can land under the no-extra-key sentinel and
   then be read by any other tenant. The backend now probes the installed
   signature once and, on such a build, declines unnamespaced lookups as
   soon as it has seen that salts are in use.

SGLang is not importable on CPU CI (its radix cache pulls in triton), so
the key, result and storage shapes are reproduced here from the builds
these backends target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, NamedTuple, Optional

import numpy as np
import pytest

from semblend.integration.sglang import hicache_backend, namespace, provider, radix_backend
from semblend.integration.sglang.hicache_backend import (
    SemBlendHiCacheStorage,
    _hicache_lookup_carries_extra_info,
)
from semblend.integration.sglang.namespace import NO_EXTRA_KEY_NAMESPACE, isolation_namespace

TENANT_A = "tenant-a-secret-salt"
TENANT_B = "tenant-b-secret-salt"

EMBED_DIM = 8
# Same unit vector for every prompt, so cosine similarity is always 1.0:
# nothing here may depend on prompts being dissimilar.
_UNIT_EMBEDDING = np.ones(EMBED_DIM, dtype=np.float32) / np.sqrt(EMBED_DIM)

# Donor and near-identical target. They differ inside the first 256 tokens
# because that window is the donor store's self-exclusion key: a target
# sharing it would be treated as the donor itself.
DONOR_TOKENS = list(range(1000, 1600))
TARGET_TOKENS = DONOR_TOKENS[:100] + [7777, 7778, 7779, 7780] + DONOR_TOKENS[104:]


# ---------------------------------------------------------------------
# SGLang key / result / request shapes
# ---------------------------------------------------------------------


@dataclass
class _RadixKey:
    """sglang.srt.mem_cache.radix_cache.RadixKey — a dataclass upstream."""

    token_ids: List[int]
    extra_key: Optional[str] = None

    def __len__(self) -> int:
        return len(self.token_ids)


class _MatchPrefixParams:
    """sglang.srt.mem_cache.base_prefix_cache.MatchPrefixParams (plain object)."""

    def __init__(self, key: Any, req: Any = None) -> None:
        self.key = key
        self.req = req


class _Node:
    """Stand-in for a radix TreeNode."""


class _MatchResult(NamedTuple):
    """sglang.srt.mem_cache.base_prefix_cache.MatchResult — a NamedTuple."""

    device_indices: List[int]
    last_device_node: Any
    last_host_node: Any
    host_hit_length: int = 0


class _ReadOnlyKey:
    """A key type that refuses attribute assignment and is not a dataclass.

    Nothing in SGLang looks like this today; it stands for any future key
    shape the rebuild cannot handle, which must decline rather than fall
    back to an unisolated bare-token-list lookup.
    """

    def __init__(self, token_ids: List[int]) -> None:
        object.__setattr__(self, "token_ids", list(token_ids))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("read-only key")

    def __len__(self) -> int:
        return len(self.token_ids)


class _ObjectMatchResult:
    """A non-tuple result object: the shape the prefix cap cannot cut."""

    def __init__(self, device_indices: List[int]) -> None:
        self.device_indices = device_indices


@dataclass
class _Req:
    """The subset of sglang's Req the donor path reads."""

    rid: str
    origin_input_ids: List[int] = field(default_factory=list)
    extra_key: Optional[str] = None


# ---------------------------------------------------------------------
# Fake radix caches
# ---------------------------------------------------------------------


class _ForkRadixCache:
    """A tree that only accepts a RadixKey and partitions on extra_key.

    This is the fork / recent-upstream contract: match_prefix is handed a
    RadixKey, and entries cached under one extra_key are a different
    partition from entries cached under another. A bare token list raises,
    exactly as it does there.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.cached: list[tuple[Optional[str], tuple[int, ...]]] = []
        self.seen_keys: list[Any] = []

    def match_prefix(self, key: Any, **kwargs: Any) -> _MatchResult:
        inner = getattr(key, "key", key)
        if not isinstance(inner, _RadixKey):
            raise TypeError(f"match_prefix expects RadixKey, got {type(inner).__name__}")
        self.seen_keys.append(inner)
        node = _Node()
        if (inner.extra_key, tuple(inner.token_ids)) in self.cached:
            return _MatchResult(list(range(len(inner.token_ids))), node, node)
        return _MatchResult([], node, node)

    def cache_finished_req(self, req: Any, *args: Any, **kwargs: Any) -> None:
        self.cached.append((req.extra_key, tuple(req.origin_input_ids)))


class _TolerantRadixCache:
    """A tree that accepts a bare token list, so the cap path is reachable.

    Used to exercise the prefix cap on its own, without the key
    reconstruction also being under test.
    """

    result_cls: Any = _MatchResult
    host_hit_length: int = 0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.cached: list[tuple[int, ...]] = []

    @staticmethod
    def _token_ids(key: Any) -> list[int]:
        if isinstance(key, (list, tuple)):
            return list(key)
        return list(key.token_ids)

    def match_prefix(self, key: Any, **kwargs: Any) -> Any:
        token_ids = self._token_ids(key)
        hit = tuple(token_ids) in self.cached
        indices = list(range(len(token_ids))) if hit else []
        if self.result_cls is _ObjectMatchResult:
            return _ObjectMatchResult(indices)
        node = _Node()
        return _MatchResult(indices, node, node, self.host_hit_length if hit else 0)

    def cache_finished_req(self, req: Any, *args: Any, **kwargs: Any) -> None:
        self.cached.append(tuple(req.origin_input_ids))


def _make_cache(base_cls: type) -> Any:
    """A SemBlendRadixCache over `base_cls` with embedding stubbed out.

    Embedding for real needs a tokenizer and a MiniLM model; the donor path
    under test only needs a vector, and a constant one makes every prompt a
    perfect match.
    """
    cls = radix_backend.get_semblend_radix_cache_class(base_cls)
    cache = cls()
    cache._embed_tokens = lambda token_ids: _UNIT_EMBEDDING  # noqa: SLF001 — test injection
    return cache


@pytest.fixture(autouse=True)
def deterministic_env(monkeypatch):
    monkeypatch.setenv("SEMBLEND_ENABLED", "1")
    monkeypatch.setenv("SEMBLEND_MAX_DONORS", "16")
    monkeypatch.setenv("SEMBLEND_MIN_SIMILARITY", "0.60")


# ---------------------------------------------------------------------
# Donor lookup key reconstruction
# ---------------------------------------------------------------------


class TestDonorLookupKey:
    def test_bare_list_key_stays_a_list(self):
        """Older SGLang keys carry no isolation value; the list is the key."""
        rebuilt = radix_backend._donor_lookup_key([1, 2, 3], DONOR_TOKENS)

        assert rebuilt == DONOR_TOKENS

    def test_radix_key_keeps_its_extra_key(self):
        key = _RadixKey(TARGET_TOKENS, TENANT_A)

        rebuilt = radix_backend._donor_lookup_key(key, DONOR_TOKENS)

        assert isinstance(rebuilt, _RadixKey)
        assert rebuilt.token_ids == DONOR_TOKENS
        assert rebuilt.extra_key == TENANT_A

    def test_request_key_is_not_mutated(self):
        """SGLang hands us the live key of the request being scheduled."""
        key = _RadixKey(list(TARGET_TOKENS), TENANT_A)

        radix_backend._donor_lookup_key(key, DONOR_TOKENS)

        assert key.token_ids == TARGET_TOKENS

    def test_params_wrapper_is_rebuilt_around_its_inner_key(self):
        req = _Req("target", TARGET_TOKENS, TENANT_A)
        key = _MatchPrefixParams(_RadixKey(TARGET_TOKENS, TENANT_A), req)

        rebuilt = radix_backend._donor_lookup_key(key, DONOR_TOKENS)

        assert isinstance(rebuilt, _MatchPrefixParams)
        assert rebuilt.key.token_ids == DONOR_TOKENS
        assert rebuilt.key.extra_key == TENANT_A
        # Some builds read the isolation value off the request instead.
        assert rebuilt.req is req
        assert key.key.token_ids == TARGET_TOKENS

    def test_unrecognized_shape_declines(self):
        class _Opaque:
            pass

        assert radix_backend._donor_lookup_key(_Opaque(), DONOR_TOKENS) is None


# ---------------------------------------------------------------------
# Donor lookup against a fork-shaped tree
# ---------------------------------------------------------------------


class TestDonorLookupOnForkApi:
    def _cache_with_donor(self, extra_key: Optional[str]):
        cache = _make_cache(_ForkRadixCache)
        cache.cache_finished_req(_Req("donor", DONOR_TOKENS, extra_key))
        return cache

    def test_same_tenant_donor_is_served(self):
        """A bare-list donor lookup raises here, so the path was inert."""
        cache = self._cache_with_donor(TENANT_A)

        cache.match_prefix(_RadixKey(TARGET_TOKENS, TENANT_A))

        assert cache.get_semblend_stats()["semantic_hits"] == 1

    def test_donor_lookup_presents_the_requests_extra_key(self):
        cache = self._cache_with_donor(TENANT_A)

        cache.match_prefix(_RadixKey(TARGET_TOKENS, TENANT_A))

        donor_lookup = cache.seen_keys[-1]
        assert donor_lookup.token_ids == DONOR_TOKENS
        assert donor_lookup.extra_key == TENANT_A

    def test_params_wrapper_reaches_the_donor_too(self):
        cache = self._cache_with_donor(TENANT_A)
        req = _Req("target", TARGET_TOKENS, TENANT_A)

        cache.match_prefix(_MatchPrefixParams(_RadixKey(TARGET_TOKENS, TENANT_A), req))

        assert cache.get_semblend_stats()["semantic_hits"] == 1
        assert cache.seen_keys[-1].extra_key == TENANT_A

    def test_cross_tenant_request_gets_nothing(self):
        cache = self._cache_with_donor(TENANT_A)

        cache.match_prefix(_RadixKey(TARGET_TOKENS, TENANT_B))

        stats = cache.get_semblend_stats()
        assert stats["semantic_hits"] == 0
        assert stats["cross_namespace_rejected"] == 1

    def test_unkeyed_request_gets_no_keyed_donor(self):
        cache = self._cache_with_donor(TENANT_A)

        cache.match_prefix(_RadixKey(TARGET_TOKENS, None))

        assert cache.get_semblend_stats()["semantic_hits"] == 0

    def test_result_keeps_the_engine_result_type(self):
        """A plain 2-tuple would drop the node fields the scheduler reads."""
        cache = self._cache_with_donor(TENANT_A)

        result = cache.match_prefix(_RadixKey(TARGET_TOKENS, TENANT_A))

        assert isinstance(result, _MatchResult)
        assert result.last_host_node is not None

    def test_reused_prefix_leaves_a_token_to_prefill(self):
        cache = self._cache_with_donor(TENANT_A)

        result = cache.match_prefix(_RadixKey(TARGET_TOKENS, TENANT_A))

        assert len(result.device_indices) == len(TARGET_TOKENS) - 1


# ---------------------------------------------------------------------
# Prefix cap
# ---------------------------------------------------------------------


class TestPrefixCap:
    def _cache(self, result_cls: Any = _MatchResult, host_hit_length: int = 0):
        base = type(
            "_ConfiguredTolerantRadixCache",
            (_TolerantRadixCache,),
            {"result_cls": result_cls, "host_hit_length": host_hit_length},
        )
        cache = _make_cache(base)
        cache.cache_finished_req(_Req("donor", DONOR_TOKENS, None))
        return cache

    def test_capped_result_keeps_its_namedtuple_type(self):
        cache = self._cache()

        result = cache.match_prefix(list(TARGET_TOKENS))

        assert isinstance(result, _MatchResult)
        assert len(result.device_indices) == len(TARGET_TOKENS) - 1
        assert result.last_device_node is not None
        assert cache.get_semblend_stats()["semantic_hits"] == 1

    def test_uncappable_result_is_declined_not_served(self):
        """An uncapped prefix leaves the scheduler no token to prefill."""
        cache = self._cache(result_cls=_ObjectMatchResult)

        result = cache.match_prefix(list(TARGET_TOKENS))

        stats = cache.get_semblend_stats()
        assert stats["semantic_hits"] == 0
        assert stats["uncapped_prefix_declined"] == 1
        assert not getattr(result, "device_indices", [])

    def test_host_side_hit_length_blocks_the_cap(self):
        """Cutting device indices under a host hit length desyncs the two."""
        cache = self._cache(host_hit_length=8)

        cache.match_prefix(list(TARGET_TOKENS))

        assert cache.get_semblend_stats()["uncapped_prefix_declined"] == 1

    def test_unrebuildable_key_declines_instead_of_dropping_isolation(self):
        """A key we cannot copy is a decline, not a bare-list lookup."""
        cache = self._cache()

        cache.match_prefix(_ReadOnlyKey(TARGET_TOKENS))

        stats = cache.get_semblend_stats()
        assert stats["donor_key_unavailable"] == 1
        assert stats["semantic_hits"] == 0


# ---------------------------------------------------------------------
# HiCacheStorage: probing the installed signature
# ---------------------------------------------------------------------


class _OldHiCacheStorage:
    """SGLang builds whose lookups take hash keys only."""

    def get(self, key, target_location=None, target_sizes=None): ...

    def exists(self, key): ...

    def batch_exists(self, keys): ...


class _NewHiCacheStorage:
    """Builds that pass the per-request extra_info through to lookups."""

    def get(self, key, target_location=None, target_sizes=None, extra_info=None): ...

    def exists(self, key, extra_info=None): ...

    def batch_exists(self, keys, extra_info=None): ...


class TestHiCacheProbe:
    def test_build_without_extra_info_is_detected(self, monkeypatch):
        monkeypatch.setattr(
            hicache_backend, "_get_hicache_storage_base", lambda: _OldHiCacheStorage
        )

        assert _hicache_lookup_carries_extra_info() is False

    def test_build_with_extra_info_is_detected(self, monkeypatch):
        monkeypatch.setattr(
            hicache_backend, "_get_hicache_storage_base", lambda: _NewHiCacheStorage
        )

        assert _hicache_lookup_carries_extra_info() is True

    def test_probe_result_is_captured_at_construction(self, monkeypatch):
        monkeypatch.setattr(
            hicache_backend, "_get_hicache_storage_base", lambda: _OldHiCacheStorage
        )
        storage = SemBlendHiCacheStorage(storage_config=None)

        # Probing per call would let a later import change the answer
        # mid-flight; it is a property of the installed package.
        assert storage._extra_info_supported is False


# ---------------------------------------------------------------------
# HiCacheStorage: fail closed when the build cannot carry a namespace
# ---------------------------------------------------------------------


class _StubTokenEmbedder:
    def embed_tokens(self, token_ids: list[int]) -> np.ndarray:
        vec = np.zeros(EMBED_DIM, dtype=np.float32)
        vec[sum(token_ids) % EMBED_DIM] = 1.0
        return vec


def _storage(monkeypatch, base: type) -> SemBlendHiCacheStorage:
    monkeypatch.setattr(hicache_backend, "_get_hicache_storage_base", lambda: base)
    storage = SemBlendHiCacheStorage(storage_config=None)
    storage._embedder = _StubTokenEmbedder()  # noqa: SLF001 — test injection
    return storage


@pytest.fixture
def old_build(monkeypatch) -> SemBlendHiCacheStorage:
    return _storage(monkeypatch, _OldHiCacheStorage)


@pytest.fixture
def new_build(monkeypatch) -> SemBlendHiCacheStorage:
    return _storage(monkeypatch, _NewHiCacheStorage)


class TestHiCacheFailsClosedOnOldBuilds:
    def test_unnamespaced_lookup_is_declined_once_salts_are_in_use(self, old_build):
        """The entry below could be a salted request whose key was dropped.

        On this build nothing reaches the backend from the lookup side, so
        an entry stored without a namespace cannot be shown to be unsalted
        — and this deployment demonstrably uses salts.
        """
        old_build.register_token_ids("hash-a", [11, 22, 33], extra_key=TENANT_A)
        old_build.set("hash-a", value="kv-a")
        old_build.set("hash-b", value="kv-b")

        assert old_build.get("hash-b") is None
        assert old_build.exists("hash-b") is False
        assert old_build.batch_exists(["hash-b"]) == 0
        assert old_build.get_stats()["isolation_declined"] == 3

    def test_decline_is_logged_once_with_a_reason(self, old_build, caplog):
        old_build.register_token_ids("hash-a", [11, 22, 33], extra_key=TENANT_A)
        old_build.set("hash-a", value="kv-a")
        old_build.set("hash-b", value="kv-b")

        with caplog.at_level("WARNING", logger="semblend.sglang"):
            old_build.get("hash-b")
            old_build.get("hash-b")

        warnings = [record for record in caplog.records if record.levelname == "WARNING"]
        assert len(warnings) == 1
        assert "extra_info" in warnings[0].getMessage()

    def test_salted_lookups_still_work(self, old_build):
        """A caller that can name its tenant is never declined."""
        old_build.set("hash-a", value="kv-a", extra_info={"extra_key": TENANT_A})

        assert old_build.get("hash-a", extra_info={"extra_key": TENANT_A}) == "kv-a"
        assert old_build.get("hash-a", extra_info={"extra_key": TENANT_B}) is None
        assert old_build.get_stats()["isolation_declined"] == 0

    def test_never_salted_deployment_is_unchanged(self, old_build):
        """Single tenant: no isolation value has ever reached this process."""
        old_build.set("hash-a", value="kv-a")

        assert old_build.get("hash-a") == "kv-a"
        assert old_build.exists("hash-a") is True
        assert old_build.batch_exists(["hash-a"]) == 1
        assert old_build.get_stats()["isolation_declined"] == 0

    def test_declines_are_not_counted_as_cross_namespace_rejections(self, old_build):
        old_build.set("hash-a", value="kv-a", extra_info={"extra_key": TENANT_A})
        old_build.set("hash-b", value="kv-b")

        old_build.get("hash-b")

        stats = old_build.get_stats()
        assert stats["isolation_declined"] == 1
        assert stats["cross_namespace_rejected"] == 0


class TestHiCacheOnBuildsThatCarryExtraInfo:
    def test_unkeyed_requests_still_share_unkeyed_entries(self, new_build):
        """Here an absent extra_info really does mean "no salt"."""
        new_build.set("hash-a", value="kv-a", extra_info={"extra_key": TENANT_A})
        new_build.set("hash-b", value="kv-b")

        assert new_build.get("hash-b") == "kv-b"
        assert new_build.get_stats()["isolation_declined"] == 0

    def test_salted_entry_stays_invisible_to_an_unkeyed_lookup(self, new_build):
        new_build.set("hash-a", value="kv-a", extra_info={"extra_key": TENANT_A})

        assert new_build.get("hash-a") is None
        assert new_build.get_stats()["cross_namespace_rejected"] == 1


# ---------------------------------------------------------------------
# One namespace helper for every SGLang path
# ---------------------------------------------------------------------


class TestNamespaceModule:
    def test_every_path_uses_one_helper(self):
        assert radix_backend._isolation_namespace is isolation_namespace
        assert provider._isolation_namespace is isolation_namespace
        assert hicache_backend.isolation_namespace is isolation_namespace

    def test_every_path_uses_one_sentinel(self):
        assert radix_backend.NO_EXTRA_KEY_NAMESPACE == NO_EXTRA_KEY_NAMESPACE
        assert provider.NO_EXTRA_KEY_NAMESPACE == NO_EXTRA_KEY_NAMESPACE
        assert hicache_backend.NO_EXTRA_KEY_NAMESPACE == NO_EXTRA_KEY_NAMESPACE

    @pytest.mark.parametrize("module", [provider, hicache_backend])
    def test_other_paths_do_not_import_the_monkey_patch_module(self, module):
        """The upstream-PR adapter and the plugin backend stand alone."""
        source = Path(module.__file__).read_text()

        assert "radix_backend" not in source

    def test_namespace_module_mirrors_the_trtllm_one(self):
        from semblend.integration.trtllm.namespace import cache_salt_namespace

        # Same scheme on both engines: absent maps to a sentinel, present
        # maps to a hash, and the two can never collide.
        assert namespace.isolation_namespace(None) == NO_EXTRA_KEY_NAMESPACE
        assert cache_salt_namespace(None) != isolation_namespace(TENANT_A)
        assert isolation_namespace(TENANT_A) != isolation_namespace(TENANT_B)
        assert TENANT_A not in isolation_namespace(TENANT_A)

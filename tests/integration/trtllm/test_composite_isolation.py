"""Composite (multi-donor) plans are dropped whole when a donor is foreign.

A composite plan assembles one request's KV from several donors at once. If
the plan is trimmed donor by donor instead of dropped, a request either gets
a plan that still mixes two tenants' KV or one that silently shrank to a
subset it never asked for. Neither is acceptable, so a single foreign donor
rejects the entire result.

Both TensorRT-LLM providers are covered: ``SemBlendTensorRTProvider`` (the
connector's provider, driven through ``SemanticKvLookupRequest``) and the
exported ``SemBlendProvider`` (TRT-LLM's ``SemanticCacheLookupProvider``
contract, which carries the salt as a keyword). Donors are registered through
the real semblend_core pipeline and donor store, so the namespace each donor
is recorded under is the real one; only the pipeline's match result is
supplied by the test, because a composite match is what is under test.
"""

from __future__ import annotations

import numpy as np
import pytest

from semblend_core.multi_donor_types import (
    ChunkAssignment,
    CompositeKVPlan,
    MatchType,
    MultiDonorPositionMapping,
    MultiDonorSlotAction,
)
from semblend_core.pipeline import PipelineResult, PositionMapping

EMBED_DIM = 8
BLOCK_SIZE = 32
CHUNK_LEN = 96

DONOR_TOKENS = list(range(1000, 1600))
TARGET_TOKENS = list(range(1000, 1600))

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

    Isolation must not depend on the prompts being dissimilar.
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
# Composite result construction
# ----------------------------------------------------------------------


def _spans(*donor_ids):
    """Contiguous, non-overlapping target spans, one per donor."""
    return [
        (donor_id, idx * CHUNK_LEN, (idx + 1) * CHUNK_LEN) for idx, donor_id in enumerate(donor_ids)
    ]


def _composite_result(spans, *, composite_only_donors=()):
    """A multi-donor PipelineResult shaped like the pipeline's own.

    `composite_only_donors` are named by the composite plan's donor set but
    contribute no slot action, which is how a donor can reach a plan without
    appearing in the per-slot stream.
    """
    slot_actions = []
    multi_slot_actions = []
    position_map = PositionMapping()
    chunk_assignments = []

    for chunk_idx, (donor_id, start, end) in enumerate(spans):
        for pos in range(start, end):
            slot_actions.append(
                {
                    "action": "copy_from_donor",
                    "targetPos": pos,
                    "donorPos": pos,
                    "donorId": donor_id,
                }
            )
            multi_slot_actions.append(
                MultiDonorSlotAction(
                    action="copy_from_donor",
                    target_pos=pos,
                    donor_pos=pos,
                    donor_id=donor_id,
                )
            )
            position_map.donor_positions.append(pos)
            position_map.target_positions.append(pos)
        chunk_assignments.append(
            ChunkAssignment(
                target_chunk_idx=chunk_idx,
                donor_id=donor_id,
                donor_chunk_idx=chunk_idx,
                match_type=MatchType.FUZZY,
                confidence=0.9,
            )
        )

    donor_ids = list(dict.fromkeys(donor_id for donor_id, _, _ in spans))
    composite_donor_ids = list(dict.fromkeys([*donor_ids, *composite_only_donors]))
    composite = CompositeKVPlan(
        donor_ids=tuple(composite_donor_ids),
        chunk_assignments=tuple(chunk_assignments),
        slot_actions=tuple(multi_slot_actions),
        position_map=MultiDonorPositionMapping(
            donor_ids=tuple(action.donor_id for action in multi_slot_actions),
            donor_positions=tuple(action.donor_pos for action in multi_slot_actions),
            target_positions=tuple(action.target_pos for action in multi_slot_actions),
        ),
        total_reuse_ratio=1.0,
        donors_per_composite=len(composite_donor_ids),
    )

    return PipelineResult(
        found=True,
        donor_id=donor_ids[0],
        similarity=1.0,
        reuse_ratio=1.0,
        donor_tokens=list(DONOR_TOKENS),
        slot_actions=slot_actions,
        layer_deviations=[],
        position_map=position_map,
        confidence_tier="verified_reuse",
        donor_ids=list(donor_ids),
        composite_plan=composite,
        multi_donor_position_map=composite.position_map,
    )


# ----------------------------------------------------------------------
# SemBlendTensorRTProvider helpers
# ----------------------------------------------------------------------


def _tensorrt_provider():
    from semblend.integration.trtllm.semblend_provider import SemBlendTensorRTProvider

    provider = SemBlendTensorRTProvider(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        chunk_size=BLOCK_SIZE,
        min_match_length=1,
        allow_segmented=True,
    )
    provider._pipeline = _make_pipeline()  # noqa: SLF001
    return provider


def _register_tensorrt(provider, salt, request_id):
    return provider.register_completed(
        request_id=request_id,
        token_ids=list(DONOR_TOKENS),
        prompt_text=DONOR_TEXT,
        namespace=_namespace(),
        block_ids=list(range(20)),
        cache_salt=salt,
    )


def _lookup_tensorrt(provider, salt, result):
    from semblend.integration.trtllm.upstream_interface import SemanticKvLookupRequest

    provider._pipeline.find_donor = lambda **kwargs: result  # noqa: SLF001
    return provider.lookup(
        SemanticKvLookupRequest(
            request_id=7,
            token_ids=tuple(TARGET_TOKENS),
            prompt_text=TARGET_TEXT,
            namespace=_namespace(),
            cache_salt=salt,
            allow_segmented=True,
            max_segments=8,
        )
    )


# ----------------------------------------------------------------------
# SemBlendProvider helpers
# ----------------------------------------------------------------------


def _legacy_provider():
    from semblend.integration.trtllm.semblend_provider import SemBlendProvider

    provider = SemBlendProvider(model_name="Qwen/Qwen2.5-7B-Instruct", chunk_size=BLOCK_SIZE)
    provider._pipeline = _make_pipeline()  # noqa: SLF001
    return provider


def _register_legacy(provider, salt, request_id):
    provider.register_completed(request_id, list(DONOR_TOKENS), DONOR_TEXT, cache_salt=salt)


def _lookup_legacy(provider, salt, result):
    provider._pipeline.find_donor = lambda **kwargs: result  # noqa: SLF001
    return provider.find_semantic_match(list(TARGET_TOKENS), TARGET_TEXT, cache_salt=salt)


class TestResultDonorIds:
    """The gate has to see every donor the result would draw KV from."""

    def test_collects_primary_slot_and_composite_donors(self):
        from semblend.integration.trtllm.semblend_provider import _result_donor_ids

        result = _composite_result(_spans("d1", "d2"), composite_only_donors=("d3",))

        assert set(_result_donor_ids(result)) == {"d1", "d2", "d3"}

    def test_donor_ids_are_deduplicated(self):
        from semblend.integration.trtllm.semblend_provider import _result_donor_ids

        result = _composite_result(_spans("d1", "d2"))

        assert _result_donor_ids(result) == ("d1", "d2")

    def test_recomputed_chunks_name_no_donor(self):
        """Recompute carries no donor KV, so it cannot carry a foreign tenant."""
        from semblend.integration.trtllm.semblend_provider import _result_donor_ids

        result = _composite_result(_spans("d1"))
        result.slot_actions = [{"action": "recompute", "targetPos": 4, "donorId": "d9"}]
        result.composite_plan = CompositeKVPlan(
            chunk_assignments=(
                ChunkAssignment(target_chunk_idx=0, donor_id="d9", match_type=MatchType.RECOMPUTE),
            ),
            slot_actions=(MultiDonorSlotAction(action="recompute", target_pos=4, donor_id="d9"),),
        )

        assert _result_donor_ids(result) == ("d1",)


class TestTensorRTCompositeIsolation:
    """SemBlendTensorRTProvider: a composite plan is all-or-nothing."""

    def test_all_native_composite_is_served(self):
        provider = _tensorrt_provider()
        _register_tensorrt(provider, SALT_A, "d1")
        _register_tensorrt(provider, SALT_A, "d2")

        result = _lookup_tensorrt(provider, SALT_A, _composite_result(_spans("d1", "d2")))

        assert result.found is True
        assert set(result.plan.donor_ids) == {"d1", "d2"}
        assert provider.get_stats()["cross_namespace_rejected"] == 0

    def test_one_foreign_donor_rejects_the_whole_composite(self):
        provider = _tensorrt_provider()
        _register_tensorrt(provider, SALT_A, "d1")
        _register_tensorrt(provider, SALT_A, "d2")
        _register_tensorrt(provider, SALT_B, "foreign")

        result = _lookup_tensorrt(
            provider, SALT_A, _composite_result(_spans("d1", "d2", "foreign"))
        )

        assert result.found is False
        assert result.plan is None
        assert result.rejection_reason == "cross_namespace"

    def test_native_donors_are_not_trimmed_into_a_partial_plan(self):
        """The same two donors served on their own must not survive here.

        Trimming the foreign donor per segment would leave a plan built from
        d1 and d2 — shorter than the one the pipeline proposed, and never
        requested.
        """
        provider = _tensorrt_provider()
        _register_tensorrt(provider, SALT_A, "d1")
        _register_tensorrt(provider, SALT_A, "d2")
        _register_tensorrt(provider, SALT_B, "foreign")

        result = _lookup_tensorrt(
            provider, SALT_A, _composite_result(_spans("d1", "d2", "foreign"))
        )

        assert result.plan is None
        assert provider.get_stats()["hits"] == 0
        assert provider.get_stats()["misses"] == 1

    def test_foreign_donor_named_only_by_the_composite_plan_is_rejected(self):
        """A donor can reach the plan without owning a single slot action."""
        provider = _tensorrt_provider()
        _register_tensorrt(provider, SALT_A, "d1")
        _register_tensorrt(provider, SALT_A, "d2")
        _register_tensorrt(provider, SALT_B, "foreign")

        result = _lookup_tensorrt(
            provider,
            SALT_A,
            _composite_result(_spans("d1", "d2"), composite_only_donors=("foreign",)),
        )

        assert result.found is False
        assert provider.get_stats()["cross_namespace_rejected"] == 1

    def test_every_foreign_donor_in_a_composite_is_counted(self):
        provider = _tensorrt_provider()
        _register_tensorrt(provider, SALT_A, "d1")
        _register_tensorrt(provider, SALT_B, "foreign-1")
        _register_tensorrt(provider, SALT_B, "foreign-2")

        result = _lookup_tensorrt(
            provider, SALT_A, _composite_result(_spans("d1", "foreign-1", "foreign-2"))
        )

        assert result.found is False
        assert provider.get_stats()["cross_namespace_rejected"] == 2

    def test_unresolvable_donor_in_a_composite_fails_closed(self):
        """A donor no store can attribute could belong to anyone."""
        provider = _tensorrt_provider()
        _register_tensorrt(provider, SALT_A, "d1")

        result = _lookup_tensorrt(
            provider, SALT_A, _composite_result(_spans("d1", "never-registered"))
        )

        assert result.found is False
        assert provider.get_stats()["cross_namespace_rejected"] == 1

    def test_unsalted_deployment_still_serves_composites(self):
        provider = _tensorrt_provider()
        _register_tensorrt(provider, None, "d1")
        _register_tensorrt(provider, None, "d2")

        result = _lookup_tensorrt(provider, None, _composite_result(_spans("d1", "d2")))

        assert result.found is True
        assert set(result.plan.donor_ids) == {"d1", "d2"}


class TestLegacyProviderCompositeIsolation:
    """SemBlendProvider: the same rule on the upstream lookup contract."""

    def test_all_native_composite_is_served(self):
        provider = _legacy_provider()
        _register_legacy(provider, SALT_A, "d1")
        _register_legacy(provider, SALT_A, "d2")

        match = _lookup_legacy(provider, SALT_A, _composite_result(_spans("d1", "d2")))

        assert match is not None
        assert match.donor_id == "d1"
        assert provider.get_stats()["cross_namespace_rejected"] == 0

    def test_one_foreign_donor_rejects_the_whole_composite(self):
        provider = _legacy_provider()
        _register_legacy(provider, SALT_A, "d1")
        _register_legacy(provider, SALT_A, "d2")
        _register_legacy(provider, SALT_B, "foreign")

        match = _lookup_legacy(provider, SALT_A, _composite_result(_spans("d1", "d2", "foreign")))

        assert match is None
        assert provider.get_stats()["hits"] == 0
        assert provider.get_stats()["misses"] == 1
        assert provider.get_stats()["cross_namespace_rejected"] == 1

    def test_foreign_donor_named_only_by_the_composite_plan_is_rejected(self):
        provider = _legacy_provider()
        _register_legacy(provider, SALT_A, "d1")
        _register_legacy(provider, SALT_B, "foreign")

        match = _lookup_legacy(
            provider,
            SALT_A,
            _composite_result(_spans("d1"), composite_only_donors=("foreign",)),
        )

        assert match is None
        assert provider.get_stats()["cross_namespace_rejected"] == 1

    def test_every_foreign_donor_in_a_composite_is_counted(self):
        provider = _legacy_provider()
        _register_legacy(provider, SALT_A, "d1")
        _register_legacy(provider, SALT_B, "foreign-1")
        _register_legacy(provider, SALT_B, "foreign-2")

        match = _lookup_legacy(
            provider, SALT_A, _composite_result(_spans("d1", "foreign-1", "foreign-2"))
        )

        assert match is None
        assert provider.get_stats()["cross_namespace_rejected"] == 2

    def test_unresolvable_donor_in_a_composite_fails_closed(self):
        provider = _legacy_provider()
        _register_legacy(provider, SALT_A, "d1")

        match = _lookup_legacy(
            provider, SALT_A, _composite_result(_spans("d1", "never-registered"))
        )

        assert match is None
        assert provider.get_stats()["cross_namespace_rejected"] == 1

    def test_unsalted_deployment_still_serves_composites(self):
        provider = _legacy_provider()
        _register_legacy(provider, None, "d1")
        _register_legacy(provider, None, "d2")

        match = _lookup_legacy(provider, None, _composite_result(_spans("d1", "d2")))

        assert match is not None
        assert match.donor_id == "d1"

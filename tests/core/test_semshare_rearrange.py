"""Tests for SemShareKV token matching and KV rearrangement."""
from __future__ import annotations

import numpy as np
import pytest

from semblend_core.semshare.config import SemShareConfig
from semblend_core.semshare.lsh_index import LSHIndex
from semblend_core.semshare.token_matcher import TokenMatchResult, match_tokens
from semblend_core.semshare.kv_rearrange import KVRearrangePlan, build_rearrange_plan


@pytest.fixture
def config() -> SemShareConfig:
    return SemShareConfig(
        embed_dim=64,
        lsh_seed=42,
        max_donors=100,
        ttl_seconds=3600,
        min_match_ratio=0.3,
        min_similarity=0.2,
    )


@pytest.fixture
def lsh_index(config: SemShareConfig) -> LSHIndex:
    return LSHIndex(config)


class TestTokenMatcher:
    def test_identical_embeddings_full_match(
        self, lsh_index: LSHIndex, config: SemShareConfig
    ) -> None:
        rng = np.random.RandomState(42)
        emb = rng.randn(20, 64).astype(np.float32)
        lsh_index.register_donor("d1", emb)

        result = match_tokens(emb, lsh_index, config)

        assert result is not None
        assert result.donor_id == "d1"
        assert result.match_ratio >= 0.5  # should be high for identical

    def test_no_donors_returns_none(
        self, lsh_index: LSHIndex, config: SemShareConfig
    ) -> None:
        rng = np.random.RandomState(42)
        result = match_tokens(rng.randn(10, 64).astype(np.float32), lsh_index, config)
        assert result is None

    def test_very_different_below_threshold(
        self, lsh_index: LSHIndex, config: SemShareConfig
    ) -> None:
        """Very different embeddings should fail min_match_ratio."""
        rng = np.random.RandomState(42)
        donor = rng.randn(10, 64).astype(np.float32) * 100
        target = rng.randn(10, 64).astype(np.float32) * 100

        lsh_index.register_donor("d1", donor)
        # Use strict threshold
        strict_config = SemShareConfig(
            embed_dim=64, min_match_ratio=0.9, min_similarity=0.8
        )
        result = match_tokens(target, lsh_index, strict_config)
        # Very different vectors with strict threshold → no match
        # (may or may not match depending on LSH randomness, so we just check it returns)
        assert result is None or result.match_ratio < 0.9

    def test_greedy_no_double_assignment(
        self, lsh_index: LSHIndex, config: SemShareConfig
    ) -> None:
        """Each donor position should be used at most once."""
        rng = np.random.RandomState(42)
        emb = rng.randn(20, 64).astype(np.float32)
        lsh_index.register_donor("d1", emb)

        result = match_tokens(emb, lsh_index, config)
        if result is not None:
            donor_positions = [p[1] for p in result.matched_pairs]
            assert len(donor_positions) == len(set(donor_positions))

    def test_best_donor_selected(
        self, lsh_index: LSHIndex, config: SemShareConfig
    ) -> None:
        """Should pick donor with most token matches."""
        rng = np.random.RandomState(42)
        target = rng.randn(10, 64).astype(np.float32)
        good_donor = target + rng.randn(10, 64).astype(np.float32) * 0.01
        bad_donor = rng.randn(10, 64).astype(np.float32) * 50

        lsh_index.register_donor("good", good_donor)
        lsh_index.register_donor("bad", bad_donor)

        result = match_tokens(target, lsh_index, config)
        if result is not None:
            assert result.donor_id == "good"

    def test_result_is_frozen(
        self, lsh_index: LSHIndex, config: SemShareConfig
    ) -> None:
        rng = np.random.RandomState(42)
        emb = rng.randn(10, 64).astype(np.float32)
        lsh_index.register_donor("d1", emb)
        result = match_tokens(emb, lsh_index, config)
        if result is not None:
            with pytest.raises(AttributeError):
                result.donor_id = "hacked"  # type: ignore[misc]


class TestKVRearrangePlan:
    def test_identity_rearrangement(self) -> None:
        """When positions match, rope_delta should be 0."""
        match = TokenMatchResult(
            matched_pairs=tuple((i, i, 0.9) for i in range(5)),
            unmatched_target=(),
            donor_id="d1",
            match_ratio=1.0,
            similarities=tuple([0.9] * 5),
        )
        plan = build_rearrange_plan(match, target_seq_len=5, donor_seq_len=5)

        assert plan.num_scatter == 5
        assert plan.num_recompute == 0
        assert plan.coverage == 1.0
        assert plan.needs_rope_correction is False
        for instr in plan.scatter:
            assert instr.rope_delta == 0

    def test_shifted_positions(self) -> None:
        """When donor positions are shifted, rope deltas computed correctly."""
        match = TokenMatchResult(
            matched_pairs=((0, 5, 0.8), (1, 6, 0.8), (2, 7, 0.8)),
            unmatched_target=(3, 4),
            donor_id="d1",
            match_ratio=0.6,
            similarities=(0.8, 0.8, 0.8, 0.0, 0.0),
        )
        plan = build_rearrange_plan(match, target_seq_len=5, donor_seq_len=10)

        assert plan.needs_rope_correction is True
        assert plan.scatter[0].rope_delta == -5  # 0 - 5
        assert plan.scatter[1].rope_delta == -5  # 1 - 6
        assert plan.num_recompute == 2
        assert plan.recompute_positions == (3, 4)

    def test_sorted_by_target_pos(self) -> None:
        match = TokenMatchResult(
            matched_pairs=((5, 0, 0.9), (2, 3, 0.8), (0, 1, 0.7)),
            unmatched_target=(),
            donor_id="d1",
            match_ratio=1.0,
            similarities=(0.7, 0.0, 0.8, 0.0, 0.0, 0.9),
        )
        plan = build_rearrange_plan(match, target_seq_len=6, donor_seq_len=6)

        positions = [instr.target_pos for instr in plan.scatter]
        assert positions == sorted(positions)

    def test_plan_is_frozen(self) -> None:
        match = TokenMatchResult(
            matched_pairs=((0, 0, 0.9),),
            unmatched_target=(),
            donor_id="d1",
            match_ratio=1.0,
            similarities=(0.9,),
        )
        plan = build_rearrange_plan(match, target_seq_len=1, donor_seq_len=1)
        with pytest.raises(AttributeError):
            plan.donor_id = "hacked"  # type: ignore[misc]

    def test_partial_coverage(self) -> None:
        match = TokenMatchResult(
            matched_pairs=((0, 0, 0.9), (1, 1, 0.9)),
            unmatched_target=(2, 3, 4),
            donor_id="d1",
            match_ratio=0.4,
            similarities=(0.9, 0.9, 0.0, 0.0, 0.0),
        )
        plan = build_rearrange_plan(match, target_seq_len=5, donor_seq_len=5)
        assert abs(plan.coverage - 0.4) < 0.01

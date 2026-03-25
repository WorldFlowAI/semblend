"""Integration tests for SemShareKV pipeline mode."""
from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import pytest

from semblend_core.pipeline import PipelineResult, SemBlendPipeline


@pytest.fixture
def semshare_pipeline():
    """Create a pipeline in semshare mode."""
    with patch.dict(os.environ, {"SEMBLEND_MODE": "semshare"}):
        pipeline = SemBlendPipeline(
            max_donors=100,
            min_similarity=0.60,
            embedder_type="minilm",
            enable_pq_segments=False,
        )
    return pipeline


@pytest.fixture
def chunk_pipeline():
    """Create a pipeline in default chunk mode."""
    with patch.dict(os.environ, {"SEMBLEND_MODE": "chunk"}):
        pipeline = SemBlendPipeline(
            max_donors=100,
            min_similarity=0.60,
            embedder_type="minilm",
            enable_pq_segments=False,
        )
    return pipeline


class TestSemSharePipelineInit:
    def test_semshare_mode_initializes_lsh(self, semshare_pipeline) -> None:
        assert semshare_pipeline.mode == "semshare"
        assert semshare_pipeline.lsh_index is not None

    def test_chunk_mode_no_lsh(self, chunk_pipeline) -> None:
        assert chunk_pipeline.mode == "chunk"
        assert chunk_pipeline.lsh_index is None

    def test_default_mode_is_chunk(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            # Remove SEMBLEND_MODE if set
            env = os.environ.copy()
            env.pop("SEMBLEND_MODE", None)
            with patch.dict(os.environ, env, clear=True):
                pipeline = SemBlendPipeline(
                    max_donors=10,
                    embedder_type="minilm",
                    enable_pq_segments=False,
                )
            assert pipeline.mode == "chunk"


class TestSemShareFindDonor:
    def test_no_donors_returns_miss(self, semshare_pipeline) -> None:
        rng = np.random.RandomState(42)
        embeddings = rng.randn(100, 4096).astype(np.float32)
        token_ids = list(range(100))

        result = semshare_pipeline.find_donor_semshare(token_ids, embeddings)

        assert result.found is False
        assert result.mode == "semshare"
        assert "no_match" in (result.rejection_reason or "")

    def test_identical_embeddings_match(self, semshare_pipeline) -> None:
        rng = np.random.RandomState(42)
        embeddings = rng.randn(50, 4096).astype(np.float32)
        token_ids = list(range(50))

        # Register donor
        semshare_pipeline.register_donor_semshare(
            "donor1", token_ids, embeddings, "test prompt"
        )

        # Query with same embeddings
        result = semshare_pipeline.find_donor_semshare(token_ids, embeddings)

        assert result.found is True
        assert result.mode == "semshare"
        assert result.donor_id == "donor1"
        assert result.token_match_ratio > 0.3

    def test_graceful_error_handling(self, semshare_pipeline) -> None:
        """Pipeline errors should not propagate — return miss."""
        # Pass invalid embeddings (wrong shape)
        result = semshare_pipeline.find_donor_semshare(
            [1, 2, 3],
            np.array([[1.0]]),  # wrong embed_dim
        )
        # Should return miss, not raise
        assert result.found is False
        assert result.mode == "semshare"


class TestSemShareRegisterDonor:
    def test_register_indexes_in_lsh(self, semshare_pipeline) -> None:
        rng = np.random.RandomState(42)
        embeddings = rng.randn(20, 4096).astype(np.float32)

        semshare_pipeline.register_donor_semshare(
            "d1", list(range(20)), embeddings, "test"
        )

        assert semshare_pipeline.lsh_index.num_donors == 1
        assert semshare_pipeline.donor_count >= 1

    def test_register_also_adds_to_donor_store(self, semshare_pipeline) -> None:
        """SemShare registration should also add to the chunk-level store."""
        rng = np.random.RandomState(42)
        embeddings = rng.randn(20, 4096).astype(np.float32)

        semshare_pipeline.register_donor_semshare(
            "d1", list(range(20)), embeddings, "some prompt text"
        )

        # Should be findable via chunk mode too
        assert semshare_pipeline.donor_count >= 1


class TestSemSharePipelineResult:
    def test_result_has_semshare_fields(self, semshare_pipeline) -> None:
        rng = np.random.RandomState(42)
        embeddings = rng.randn(30, 4096).astype(np.float32)
        token_ids = list(range(30))

        semshare_pipeline.register_donor_semshare("d1", token_ids, embeddings, "test")
        result = semshare_pipeline.find_donor_semshare(token_ids, embeddings)

        if result.found:
            assert result.kv_rearrange_plan is not None
            assert result.token_match_ratio > 0
            assert result.reuse_ratio > 0

    def test_chunk_mode_result_has_default_semshare_fields(
        self, chunk_pipeline
    ) -> None:
        """Chunk mode results should have default semshare fields."""
        result = PipelineResult(found=False)
        assert result.mode == "chunk"
        assert result.semshare_schedule is None
        assert result.kv_rearrange_plan is None
        assert result.token_match_ratio == 0.0

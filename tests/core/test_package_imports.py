"""Tests that the semblend package imports work correctly."""

from __future__ import annotations


class TestSemblendPackageImports:
    """Verify the semblend top-level package imports."""

    def test_version(self):
        import semblend

        assert semblend.__version__ == "0.3.1"

    def test_pipeline_import(self):
        from semblend import SemBlendPipeline

        assert SemBlendPipeline is not None

    def test_pipeline_result_import(self):
        from semblend import PipelineResult, PipelineTimings

        assert PipelineResult is not None
        assert PipelineTimings is not None

    def test_backend_import(self):
        from semblend import SemBlendBackend

        assert SemBlendBackend is not None

    def test_alignment_import(self):
        from semblend import (
            AlignmentResult,
            compute_alignment,
        )

        assert AlignmentResult is not None
        assert compute_alignment is not None

    def test_donor_store_import(self):
        from semblend import DonorStore

        assert DonorStore is not None

    def test_embedder_import(self):
        from semblend import EmbedderType, create_embedder

        assert EmbedderType is not None
        assert create_embedder is not None

    def test_bathtub_import(self):
        from semblend import (
            BathtubPreset,
            sigma,
        )

        assert BathtubPreset is not None
        assert sigma is not None

    def test_partial_attention_import(self):
        from semblend import (
            AttentionMode,
        )

        assert AttentionMode is not None

    def test_simhash_import(self):
        from semblend import compute_simhash

        assert compute_simhash is not None


class TestSemblendCoreImports:
    """Verify the semblend.core namespace imports."""

    def test_core_import(self):
        import semblend.core

        assert hasattr(semblend.core, "HAS_ROPE_CORRECTION")
        assert hasattr(semblend.core, "HAS_TRITON_KERNELS")

    def test_core_alignment(self):
        from semblend.core import compute_alignment

        assert compute_alignment is not None

    def test_core_donor_store(self):
        from semblend.core import DonorStore

        assert DonorStore is not None


class TestBackwardCompatImports:
    """Verify that old import paths still work."""

    def test_semblend_core_import(self):
        import semblend_core

        assert hasattr(semblend_core, "SemBlendPipeline")

    def test_semblend_core_alignment(self):
        from semblend_core import compute_alignment

        assert compute_alignment is not None

    def test_semblend_core_donor_store(self):
        from semblend_core import DonorStore

        assert DonorStore is not None

    def test_semblend_core_pipeline(self):
        from semblend_core.pipeline import SemBlendPipeline

        assert SemBlendPipeline is not None

    def test_semblend_core_embedder(self):
        from semblend_core.embedder import create_embedder

        assert create_embedder is not None


class TestCoreModuleDirect:
    """Test direct module access in semblend.core namespace."""

    def test_alignment_module(self):
        from semblend_core.alignment import compute_chunk_alignment

        assert compute_chunk_alignment is not None

    def test_bathtub_module(self):
        from semblend_core.bathtub import get_preset

        preset = get_preset("qwen")
        assert preset is not None
        assert preset.num_layers > 0

    def test_donor_store_creation(self):
        from semblend import DonorStore

        store = DonorStore(max_entries=10)
        assert store.size == 0

    def test_simhash_computation(self):
        from semblend import compute_simhash

        h = compute_simhash([1, 2, 3, 4, 5])
        assert isinstance(h, int)

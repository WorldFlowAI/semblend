"""Re-exports from semblend_core.pipeline plus a vLLM-specific factory.

Core pipeline logic lives in semblend_core. This module re-exports
everything and provides a vLLM-specific factory that uses the local
in-process donor store.
"""

from semblend_core.pipeline import (  # noqa: F401
    PipelineResult,
    PipelineTimings,
    PositionMapping,
    SemBlendPipeline,
    _order_invariant_text,
)


def create_vllm_pipeline(
    max_donors: int = 10_000,
    min_similarity: float = 0.60,
    min_reuse_ratio: float = 0.50,
    embedder_type: str | None = None,
    model_name: str | None = None,
    chunk_size: int | None = None,
) -> SemBlendPipeline:
    """Create a SemBlendPipeline with the local in-process donor store."""
    from semblend_core.donor_store import DonorStore
    from semblend_core.embedder import create_embedder

    embedder = create_embedder(embedder_type)
    donor_store = DonorStore(
        max_entries=max_donors,
        embedding_dim=embedder.dimension,
        min_similarity=min_similarity,
        chunk_size=chunk_size or 32,
    )

    return SemBlendPipeline(
        max_donors=max_donors,
        min_similarity=min_similarity,
        min_reuse_ratio=min_reuse_ratio,
        embedder_type=embedder_type,
        model_name=model_name,
        chunk_size=chunk_size,
        donor_store=donor_store,
    )

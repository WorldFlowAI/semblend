"""Re-exports from semblend_core.pipeline + vLLM-specific factory.

Core pipeline logic lives in semblend_core. This module re-exports
everything and provides a vLLM-specific factory that injects the
CAGRA GPU-accelerated donor store when available.
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
    """Create a SemBlendPipeline with vLLM-optimized donor store.

    Uses CAGRA GPU-accelerated ANN index when cuVS is available and
    SEMBLEND_USE_CAGRA=1 is set; falls back to numpy DonorStore otherwise.
    """
    from semblend_core.embedder import create_embedder
    from synapse_kv_connector.cagra_donor_store import make_donor_store

    embedder = create_embedder(embedder_type)
    donor_store = make_donor_store(
        max_entries=max_donors,
        embedding_dim=embedder.dimension,
        min_similarity=min_similarity,
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

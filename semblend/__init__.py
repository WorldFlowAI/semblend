"""SemBlend — Semantic KV cache reuse for LLM inference engines.

Extends exact-prefix caching (LMCache, vLLM prefix caching, SGLang RadixAttention)
with semantic donor discovery: finds and reuses KV tensors from semantically similar
prior requests, eliminating redundant prefill computation.

Quick start:
    pip install semblend

    # vLLM integration (dynamic connector loading):
    vllm serve model_name \\
      --kv-transfer-config '{"kv_connector": "SemBlendConnectorV1",
        "kv_connector_module_path": "semblend.integration.vllm.connector_v1",
        "kv_role": "kv_both"}'

    # SGLang integration (HiCache storage backend):
    python -m sglang.launch_server --model-path model_name \\
      --enable-hierarchical-cache \\
      --hicache-storage-backend dynamic \\
      --hicache-storage-backend-extra-config \\
        '{"module_path":"semblend.integration.sglang.hicache_backend",
          "class_name":"SemBlendHiCacheStorage"}'

Environment variables:
    SEMBLEND_ENABLED=1              Enable semantic donor search
    SEMBLEND_MIN_SIMILARITY=0.60    Minimum cosine similarity threshold
    SEMBLEND_EMBEDDER=minilm        Embedder type (minilm, jaccard, onnx_gpu)
    SEMBLEND_FUZZY_CHUNKS=1         Enable fuzzy chunk matching
"""

from __future__ import annotations

__version__ = "0.3.1"

# Re-export core public API from semblend_core
from semblend_core.alignment import (
    AlignmentResult,
    SlotAction,
    SlotActionType,
    compute_alignment,
    compute_batch_alignment,
    compute_chunk_alignment,
    estimate_reuse_ratio,
)
from semblend_core.backend import SemBlendBackend
from semblend_core.bathtub import (
    BathtubPreset,
    LayerDeviation,
    compute_layer_deviations,
    get_preset,
    sigma,
)
from semblend_core.donor_store import DonorMatch, DonorNode, DonorStore
from semblend_core.embedder import (
    EmbedderType,
    JaccardEmbedder,
    MiniLMEmbedder,
    OnnxGpuEmbedder,
    create_embedder,
)
from semblend_core.partial_attention import (
    AttentionMode,
    PartialAttentionPlan,
    build_attention_plan,
    compute_attention_mask,
    compute_donor_kv_indices,
)
from semblend_core.pipeline import (
    PipelineResult,
    PipelineTimings,
    PositionMapping,
    SemBlendPipeline,
)
from semblend_core.simhash import (
    bulk_hamming_distance,
    compute_simhash,
    hamming_distance,
    is_plausible_donor,
)

__all__ = [
    "__version__",
    # Pipeline
    "SemBlendPipeline",
    "PipelineResult",
    "PipelineTimings",
    "PositionMapping",
    # Backend ABC
    "SemBlendBackend",
    # Alignment
    "AlignmentResult",
    "SlotAction",
    "SlotActionType",
    "compute_alignment",
    "compute_batch_alignment",
    "compute_chunk_alignment",
    "estimate_reuse_ratio",
    # Donor store
    "DonorMatch",
    "DonorNode",
    "DonorStore",
    # Embedder
    "EmbedderType",
    "JaccardEmbedder",
    "MiniLMEmbedder",
    "OnnxGpuEmbedder",
    "create_embedder",
    # Bathtub
    "BathtubPreset",
    "LayerDeviation",
    "compute_layer_deviations",
    "get_preset",
    "sigma",
    # Partial attention
    "AttentionMode",
    "PartialAttentionPlan",
    "build_attention_plan",
    "compute_attention_mask",
    "compute_donor_kv_indices",
    # SimHash
    "bulk_hamming_distance",
    "compute_simhash",
    "hamming_distance",
    "is_plausible_donor",
]

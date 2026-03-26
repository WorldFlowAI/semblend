"""SemBlend Core — backend-agnostic semantic KV donor discovery.

Shared pipeline for semantic KV cache reuse across inference engines.
Backend-specific adapters (vLLM/LMCache, TRT-LLM) implement the
SemBlendBackend ABC and plug into this shared core.

Components:
  - pipeline.py     — 5-stage orchestrator (embed → lookup → align → bathtub → plan)
  - backend.py      — SemBlendBackend ABC for engine adapters
  - embedder.py     — MiniLM-L6-v2 ONNX GPU embedder (3ms)
  - donor_store.py  — Numpy cosine donor lookup (1-2ms)
  - alignment.py    — Chunk/token alignment (parameterized chunk_size, context gate)
  - bathtub.py      — Layer deviation scoring (per-model presets: Qwen, LLaMA)
  - partial_attention.py — Attention plan builder
  - rope_correction.py   — RoPE delta correction + NoPE two-step (MEPIC-style)
  - triton_kernels.py    — CUDA scatter kernels
  - simhash.py      — SimHash pre-filter
"""

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
from semblend_core.chunk_index import ChunkIndex, ChunkLocation, chunk_hash_sequence
from semblend_core.donor_store import DonorMatch, DonorNode, DonorStore
from semblend_core.embedder import (
    EmbedderType,
    JaccardEmbedder,
    MiniLMEmbedder,
    OnnxGpuEmbedder,
    create_embedder,
)
from semblend_core.multi_donor_alignment import compute_multi_donor_alignment
from semblend_core.multi_donor_types import (
    ChunkAssignment,
    CompositeKVPlan,
    MatchType,
    MultiDonorAlignmentResult,
    MultiDonorPositionMapping,
    MultiDonorSlotAction,
)
from semblend_core.partial_attention import (
    AttentionMode,
    PartialAttentionPlan,
    build_attention_plan,
    build_multi_donor_attention_plan,
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

# Optional GPU components
try:
    from semblend_core.rope_correction import (  # noqa: F401
        apply_rope_delta_inplace,
        nope_permute_paged_kv,
        permute_paged_kv_with_rope,
        rope_correct_k,
        rope_correct_scatter_paged,
    )

    HAS_ROPE_CORRECTION = True
except ImportError:
    HAS_ROPE_CORRECTION = False

try:
    from semblend_core.triton_kernels import (  # noqa: F401
        PartialPrefillResult,
        masked_qkv_projection,
        partial_prefill,
        partial_prefill_attention,
        scatter_donor_kv,
        scatter_donor_kv_paged,
    )

    HAS_TRITON_KERNELS = True
except ImportError:
    HAS_TRITON_KERNELS = False

__all__ = [
    # Backend ABC
    "SemBlendBackend",
    # Pipeline
    "PipelineResult",
    "PipelineTimings",
    "PositionMapping",
    "SemBlendPipeline",
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
    "build_multi_donor_attention_plan",
    "compute_attention_mask",
    "compute_donor_kv_indices",
    # ChunkIndex
    "ChunkIndex",
    "ChunkLocation",
    "chunk_hash_sequence",
    # Multi-donor
    "ChunkAssignment",
    "CompositeKVPlan",
    "MatchType",
    "MultiDonorAlignmentResult",
    "MultiDonorPositionMapping",
    "MultiDonorSlotAction",
    "compute_multi_donor_alignment",
    # SimHash
    "bulk_hamming_distance",
    "compute_simhash",
    "hamming_distance",
    "is_plausible_donor",
]

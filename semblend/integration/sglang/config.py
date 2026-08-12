"""Configuration for the SemBlend SGLang provider adapter.

Mirrors the new fields proposed for SGLang's FuzzyMatchConfig
(docs/sglang_semantic_provider_design.md § 7) so the adapter can be
constructed either from a parsed SGLang config or directly from a Python
dict without importing SGLang.

SemBlend is process-local: in-process MiniLM embedding and a numpy donor
store. There is no remote backend or service dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SemBlendProviderConfig:
    """Adapter config — subset of SGLang's FuzzyMatchConfig relevant to SemBlend.

    Defaults match the SemBlend paper (§ 3.1, Table 9) and the existing
    `SemBlendPipeline` defaults.
    """

    # Matching thresholds (paper § 3.1)
    min_similarity: float = 0.60
    min_reuse_ratio: float = 0.50
    min_match_length: int = 128

    # Donor store
    max_entries: int = 10_000
    block_size: int = 32

    # Embedding (process-local; uses MiniLM with optional GPU acceleration)
    embedder_type: str = "minilm"  # "minilm" | "onnx-gpu" | "e5"
    embedding_use_gpu: bool = True
    embedding_model_name: str = "all-MiniLM-L6-v2"

    # Bathtub (per-layer recomputation)
    enable_bathtub: bool = True
    model_arch: Optional[str] = None  # "llama" | "qwen2.5-7b" | None

    # Search
    top_k: int = 5

    # Quality gate (informational; monitored but not enforced here)
    quality_gate_ppl_threshold: float = 1.065

    # ----------------------------------------------------------------
    # Multi-segment emission (v0.4 line). Default OFF: with emission off
    # the adapter collapses to the single contiguous head run exactly as
    # 0.3.x did. When ON, every aligned segment that passes the
    # per-segment gate is emitted for the engine's segmented realizer.
    # ----------------------------------------------------------------
    multi_segment_emission: bool = False
    # Per-segment gate: fraction of positions in the aligned run whose
    # donor and target token ids are IDENTICAL. Reusing donor KV for
    # non-identical tokens is the direct damage mechanism, and
    # near-duplicate documents that differ only in entities (the
    # entity-swap hazard) produce runs with identity just below 1.0 —
    # a high default keeps them out.
    segment_min_token_identity: float = 0.98
    # Segments shorter than this are not worth a realization pass.
    segment_min_tokens: int = 16
    # Verified paraphrase serve: minimum embedding similarity before the
    # lexical fact gate is even consulted (the gate, not similarity, is
    # the acceptance decision; this floor just bounds gate volume).
    # Faithful full paraphrases of long documents measure ~0.84 cosine,
    # so the floor sits below that band.
    paraphrase_min_similarity: float = 0.80
    # Never realize donor KV into the first N target positions (attention
    # sink): foreign KV there poisons all downstream attention. 0 = off.
    sink_protect_tokens: int = 0
    # Donor-side counterpart: never realize KV computed at the DONOR's own
    # sink positions (phantom-sink import). 0 = off.
    donor_sink_protect_tokens: int = 0
    # Trim N tokens from BOTH ends of every gated run: run-boundary K/V
    # carries the strongest cross-context mismatch (boundary artifacts),
    # and trimmed positions are true-recomputed by the forward. 0 = off.
    segment_edge_trim_tokens: int = 0
    # Trim N tokens from the HEAD of every run: a run's first tokens carry
    # the donor's local attention onset regardless of absolute position
    # Combined with edge trim via max(). 0 = off.
    donor_run_head_trim_tokens: int = 0
    # Gated runs are concatenated into merged segments of up to this many
    # positions before emission. Position arrays are explicit, so merged
    # segments need no contiguity — this collapses hundreds of per-run
    # realization kernel launches into a few scatter copies (measured in live serving).
    segment_merge_max_positions: int = 4096

    # ----------------------------------------------------------------
    # Operating modes
    # ----------------------------------------------------------------

    # When True, the adapter still runs the full SemBlend pipeline
    # (embed → search → align → bathtub) and surfaces match metrics via
    # `QualitySignals`, but returns `cached_token_count=0` so SGLang's
    # RadixCache does NOT inject donor KV indices into match_prefix's
    # device_indices. Useful when the upstream RadixCache lacks the
    # donor inc_lock_ref protection (sglang-fuzzy-local @ ec4c41e):
    # discovery-only mode lets us measure hit rate, latency, and quality
    # (cold prefill happens normally) without tripping the leak detector.
    #
    # Set to False once the lock_ref fix is confirmed present.
    discovery_only: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "SemBlendProviderConfig":
        """Build from a plain dict, ignoring unknown keys.

        Allows a future SGLang-side wrapper to forward fields from its own
        FuzzyMatchConfig without worrying about version skew.
        """
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

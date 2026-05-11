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
    # match_block discovery (substring-based contiguous-run finder)
    # ----------------------------------------------------------------

    # Minimum token length for a discovered match_block to be surfaced to
    # SGLang. Blocks shorter than this don't beat the two-pass
    # forward_extend orchestration overhead and are dropped. Applies to
    # both the segments-path block and the substring-path block.
    match_block_min_length: int = 4

    # Maximum n-gram anchor size for the substring-path donor index.
    # Larger values reduce false positives (each n-gram is more
    # distinctive) but cannot find matches shorter than the anchor. The
    # effective anchor is clamped to the shorter of donor / target token
    # length, so this is an upper bound. 8 is a sensible default for
    # natural-language text (an 8-token phrase is rarely repeated).
    match_block_max_anchor: int = 8

    # Cap on candidates examined per n-gram anchor in the substring scan.
    # Prevents pathological inner-loop blow-up when a single n-gram
    # appears many times in the donor (heavily templated / repetitive
    # data). Natural-language workloads never reach this limit; raise it
    # for highly structured inputs where the true longest-match anchor
    # may be deep in the candidate list.
    match_block_max_candidates_per_anchor: int = 256

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

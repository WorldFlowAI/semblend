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
import os
from typing import Optional


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    return int(value)


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
    discovery_only: bool = _env_bool("SEMBLEND_DISCOVERY_ONLY", False)

    # Quality-first materialization safety. The current SGLang prefix-cache
    # contract can only express prefix-shaped hits. If SemBlend finds a reusable
    # span that starts later in the unmatched target suffix, collapsing it to the
    # prefix boundary can apply donor KV to wrapper/instruction/schema tokens.
    # Keep this enabled unless the engine supports true recompute holes or
    # segmented/non-prefix materialization.
    strict_prefix_boundary: bool = _env_bool("SEMBLEND_STRICT_PREFIX_BOUNDARY", True)

    # Optional extra guard: only materialize token-exact spans. Semantic search
    # can still discover candidate workers; approximate/fuzzy chunks are left
    # for future segmented recompute support instead of being injected as KV.
    exact_materialization_only: bool = _env_bool("SEMBLEND_EXACT_MATERIALIZATION_ONLY", True)

    # Experimental SGLang path: return the full segment plan instead of
    # collapsing the best run into the legacy contiguous-prefix contract.
    return_segments: bool = _env_bool("SEMBLEND_RETURN_SEGMENTS", False)

    @classmethod
    def from_dict(cls, d: dict) -> "SemBlendProviderConfig":
        """Build from a plain dict, ignoring unknown keys.

        Allows a future SGLang-side wrapper to forward fields from its own
        FuzzyMatchConfig without worrying about version skew.
        """
        known = {f for f in cls.__dataclass_fields__}
        values = {k: v for k, v in d.items() if k in known}
        block_size = _env_int("SEMBLEND_BLOCK_SIZE")
        if block_size is not None:
            values["block_size"] = block_size
        return cls(**values)

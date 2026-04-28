"""Configuration for the SemBlend SGLang provider adapter.

Mirrors the new fields proposed for SGLang's FuzzyMatchConfig
(docs/sglang_semantic_provider_design.md § 8) so the adapter can be
constructed either from a parsed SGLang config or directly from a Python
dict without importing SGLang.
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

    # Embedding
    embedder_type: str = "minilm"  # "minilm" | "onnx-gpu" | "e5" | "jaccard"
    embedding_use_gpu: bool = True
    embedding_model_name: str = "all-MiniLM-L6-v2"

    # Bathtub (per-layer recomputation)
    enable_bathtub: bool = True
    model_arch: Optional[str] = None  # "llama" | "qwen2.5-7b" | None

    # Search
    top_k: int = 5

    # Backend mode
    embedding_backend: str = "local"  # "local" | "gateway"
    gateway_url: Optional[str] = None
    gateway_timeout_ms: int = 3

    # Quality gate (informational; monitored but not enforced here)
    quality_gate_ppl_threshold: float = 1.065

    # ----------------------------------------------------------------
    # Operating modes (orthogonal to provider semantics)
    # ----------------------------------------------------------------

    # When True, the adapter still runs the full SemBlend pipeline
    # (embed → search → align → bathtub) and surfaces match metrics via
    # `QualitySignals`, but returns `cached_token_count=0` so SGLang's
    # RadixCache does NOT inject donor KV indices into match_prefix's
    # device_indices. Useful when the upstream RadixCache has the
    # _node_registry / cache_protected_len leak that fires under
    # sustained fuzzy hits — discovery-only mode lets us measure hit
    # rate, latency, and quality (cold prefill happens normally) without
    # tripping the leak.
    #
    # Set to False once Chenxin's _delete_leaf fix lands upstream and
    # the leak is confirmed gone, OR when running our own patched fork.
    discovery_only: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "SemBlendProviderConfig":
        """Build from a plain dict, ignoring unknown keys.

        Allows a future SGLang-side wrapper to forward fields from its own
        FuzzyMatchConfig without worrying about version skew.
        """
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

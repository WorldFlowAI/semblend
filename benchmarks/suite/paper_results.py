"""Paper reference values for comparison with v0.2.0 results.

These are the published numbers from the SemBlend paper tables.
Used by compare.py to compute deltas between paper and reproduction runs.

All TTFT values in milliseconds. Speedup = cold_ttft / semblend_ttft.
PPL ratio = semblend_ppl / cold_ppl (1.0 = identical quality).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TTFTEntry:
    """A single TTFT measurement from the paper."""

    context_length: int
    cold_ttft_ms: float
    semblend_ttft_ms: float
    speedup: float
    hit_rate: float  # 0.0–1.0
    model: str = ""  # "qwen" or "llama"
    n_samples: int = 8


@dataclass(frozen=True)
class PPLEntry:
    """A single PPL ratio measurement from the paper."""

    dataset: str
    model: str
    context_length: int
    ppl_ratio: float  # SB/cold, 1.0 = no degradation
    n_samples: int = 8


@dataclass(frozen=True)
class EngineCompEntry:
    """Three-way engine comparison entry."""

    engine: str
    cold_ttft_ms: float
    hit_ttft_ms: float
    hit_rate: float
    speedup: float


@dataclass(frozen=True)
class TableReference:
    """Complete reference data for one paper table."""

    table_number: int
    title: str
    entries: list = field(default_factory=list)
    notes: str = ""


# --------------------------------------------------------------------------
# Table 3: SemBlend KV Donor Injection — TTFT 2K–32K
# --------------------------------------------------------------------------
TABLE_3 = TableReference(
    table_number=3,
    title="SemBlend KV Donor Injection — TTFT Speedup",
    entries=[
        # Qwen2.5-7B-AWQ — validated 2026-03-26, A10G g5.4xlarge, CPU offload 20GB
        # Fresh pod per length, 2-request warmup, bootstrap 95% CIs
        TTFTEntry(8192, 3096, 534, 5.53, 0.90, model="qwen"),   # n=30, CI [5.24, 5.75]
        TTFTEntry(12288, 4894, 623, 7.88, 0.89, model="qwen"),  # n=28, CI [7.83, 7.94]
        TTFTEntry(16384, 6646, 763, 8.68, 0.90, model="qwen"),  # n=20, CI [8.56, 8.79]
        TTFTEntry(24576, 10616, 827, 12.99, 1.00, model="qwen"),  # n=14, CI [12.76, 13.23]
        # LLaMA-3.1-8B-AWQ (previous validation, pending re-run)
        TTFTEntry(8192, 3416, 627, 5.45, 1.00, model="llama"),
    ],
    notes=(
        "Qwen validated 2026-03-26 on A10G g5.4xlarge (64GB RAM), CPU offload 20GB. "
        "Fresh pod per length with 2-request warmup. Speedup is hit-conditional (p50). "
        "~10% miss rate at 8K-16K from instruction variants below cosine threshold."
    ),
)


# --------------------------------------------------------------------------
# Table 4: Prefix Cache Coverage Gap
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PrefixCoverageEntry:
    """Prefix cache vs SemBlend comparison."""

    scenario: str
    context_length: int
    vllm_speedup: float
    semblend_speedup: float
    semblend_hit_rate: float


TABLE_4 = TableReference(
    table_number=4,
    title="Prefix Cache Coverage Gap",
    entries=[
        PrefixCoverageEntry("exact", 16384, 6.35, 8.20, 1.00),
        PrefixCoverageEntry("reorder", 8192, 1.00, 2.57, 1.00),
        PrefixCoverageEntry("partial-75", 8192, 1.01, 1.95, 1.00),
        PrefixCoverageEntry("paraphrase", 8192, 1.00, 2.53, 1.00),
    ],
)


# --------------------------------------------------------------------------
# Table 5: Cross-Instruction RAG Benchmark
# --------------------------------------------------------------------------
TABLE_5 = TableReference(
    table_number=5,
    title="Cross-Instruction RAG Benchmark",
    entries=[
        TTFTEntry(8192, 3626, 1093, 3.32, 1.00),  # mean of 3.28-3.34
        TTFTEntry(16384, 7113, 1295, 5.32, 1.00, n_samples=6),
    ],
    notes="XSum 8K, n=8 articles x 4 instruction variants.",
)


# --------------------------------------------------------------------------
# Table 6: Quality Preservation — PPL Ratios
# --------------------------------------------------------------------------
TABLE_6 = TableReference(
    table_number=6,
    title="Quality Preservation — PPL Ratios",
    entries=[
        # Qwen2.5-7B
        PPLEntry("xsum", "qwen", 2048, 0.988),
        PPLEntry("xsum", "qwen", 4096, 1.008),
        PPLEntry("xsum", "qwen", 8192, 1.025),
        PPLEntry("cnn_dailymail", "qwen", 2048, 0.970),
        PPLEntry("cnn_dailymail", "qwen", 8192, 1.023),
        PPLEntry("cnn_dailymail", "qwen", 16384, 1.000),
        PPLEntry("wikihow", "qwen", 2048, 1.021),
        PPLEntry("wikihow", "qwen", 8192, 1.041),
        PPLEntry("wikihow", "qwen", 16384, 1.003),
        # LLaMA-3.1-8B
        PPLEntry("xsum", "llama", 2048, 0.984),
        PPLEntry("xsum", "llama", 4096, 0.998),
        PPLEntry("xsum", "llama", 8192, 0.990),
    ],
)


# --------------------------------------------------------------------------
# Table 7: Quality Across Datasets and Models — Extended PPL
# --------------------------------------------------------------------------
TABLE_7 = TableReference(
    table_number=7,
    title="Quality Across Datasets and Models — Extended PPL",
    entries=[
        # Qwen2.5-7B across 5 datasets
        PPLEntry("xsum", "qwen", 2048, 0.988),
        PPLEntry("xsum", "qwen", 4096, 1.008),
        PPLEntry("xsum", "qwen", 8192, 1.025),
        PPLEntry("cnn_dailymail", "qwen", 2048, 0.960),
        PPLEntry("cnn_dailymail", "qwen", 4096, 1.060),
        PPLEntry("cnn_dailymail", "qwen", 8192, 1.006),
        PPLEntry("multinews", "qwen", 2048, 0.944),
        PPLEntry("multinews", "qwen", 4096, 1.005),
        PPLEntry("multinews", "qwen", 8192, 1.119),
        PPLEntry("wikihow", "qwen", 2048, 1.005),
        PPLEntry("wikihow", "qwen", 4096, 1.009),
        PPLEntry("wikihow", "qwen", 8192, 1.012),
        PPLEntry("samsum", "qwen", 2048, 1.029),
        PPLEntry("samsum", "qwen", 4096, 1.071),
        PPLEntry("samsum", "qwen", 8192, 1.136),
        # LLaMA-3.1-8B across 5 datasets
        PPLEntry("xsum", "llama", 2048, 1.009),
        PPLEntry("xsum", "llama", 4096, 1.011),
        PPLEntry("xsum", "llama", 8192, 1.065),
        PPLEntry("multinews", "llama", 2048, 1.020),
        PPLEntry("multinews", "llama", 4096, 1.006),
        PPLEntry("multinews", "llama", 8192, 1.063),
        PPLEntry("wikihow", "llama", 2048, 0.992),
        PPLEntry("wikihow", "llama", 4096, 1.042),
        PPLEntry("wikihow", "llama", 8192, 1.001),
        PPLEntry("cnn_dailymail", "llama", 2048, 1.039),
        PPLEntry("cnn_dailymail", "llama", 4096, 1.059),
        PPLEntry("cnn_dailymail", "llama", 8192, 1.077),
        PPLEntry("samsum", "llama", 2048, 1.110),
        PPLEntry("samsum", "llama", 4096, 1.081),
        PPLEntry("samsum", "llama", 8192, 1.171),
    ],
)


# --------------------------------------------------------------------------
# Table 8: Cross-Dataset TTFT Speedup (Qwen)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CrossDatasetEntry:
    """Cross-dataset TTFT entry."""

    dataset: str
    context_length: int
    speedup: float
    hit_rate: float


TABLE_8 = TableReference(
    table_number=8,
    title="Cross-Dataset TTFT Speedup",
    entries=[
        # Synthetic
        CrossDatasetEntry("synthetic", 2048, 2.35, 1.00),
        CrossDatasetEntry("synthetic", 4096, 2.21, 1.00),
        CrossDatasetEntry("synthetic", 8192, 5.66, 1.00),
        CrossDatasetEntry("synthetic", 12288, 1.04, 0.12),
        CrossDatasetEntry("synthetic", 16384, 1.00, 0.25),
        # XSum
        CrossDatasetEntry("xsum", 2048, 1.51, 0.75),
        CrossDatasetEntry("xsum", 8192, 1.97, 1.00),
        CrossDatasetEntry("xsum", 12288, 1.63, 1.00),
        CrossDatasetEntry("xsum", 16384, 1.52, 1.00),
        # MultiNews
        CrossDatasetEntry("multinews", 2048, 1.57, 0.62),
        CrossDatasetEntry("multinews", 4096, 1.50, 0.75),
        CrossDatasetEntry("multinews", 8192, 2.23, 0.75),
        # CNN/DailyMail
        CrossDatasetEntry("cnn_dailymail", 2048, 1.59, 0.75),
        CrossDatasetEntry("cnn_dailymail", 4096, 1.60, 0.38),
        CrossDatasetEntry("cnn_dailymail", 8192, 2.29, 0.50),
        # WikiHow
        CrossDatasetEntry("wikihow", 2048, 1.62, 0.88),
        CrossDatasetEntry("wikihow", 4096, 1.50, 0.88),
        CrossDatasetEntry("wikihow", 8192, 2.24, 1.00),
        # SAMSum
        CrossDatasetEntry("samsum", 2048, 1.61, 0.88),
        CrossDatasetEntry("samsum", 4096, 1.46, 0.75),
        CrossDatasetEntry("samsum", 8192, 2.37, 0.88),
    ],
    notes="Qwen2.5-7B-AWQ, A10G, n=8 per length.",
)


# --------------------------------------------------------------------------
# Table 9: Multi-Model TTFT — LLaMA
# --------------------------------------------------------------------------
TABLE_9 = TableReference(
    table_number=9,
    title="Multi-Model TTFT — LLaMA-3.1-8B",
    entries=[
        CrossDatasetEntry("xsum", 2048, 3.06, 1.00),
        CrossDatasetEntry("xsum", 4096, 2.77, 1.00),
        CrossDatasetEntry("xsum", 8192, 3.34, 1.00),
        CrossDatasetEntry("cnn_dailymail", 2048, 3.00, 0.62),
        CrossDatasetEntry("cnn_dailymail", 4096, 2.66, 0.62),
        CrossDatasetEntry("cnn_dailymail", 8192, 3.26, 0.25),
        CrossDatasetEntry("wikihow", 2048, 2.96, 1.00),
        CrossDatasetEntry("wikihow", 4096, 2.72, 1.00),
        CrossDatasetEntry("wikihow", 8192, 3.33, 1.00),
        CrossDatasetEntry("samsum", 2048, 3.08, 0.88),
        CrossDatasetEntry("samsum", 4096, 2.69, 0.62),
        CrossDatasetEntry("samsum", 8192, 3.78, 0.75),
    ],
    notes="LLaMA-3.1-8B-AWQ, A10G, n=8 per length.",
)


# --------------------------------------------------------------------------
# Table 10: Three-Way Engine Comparison
# --------------------------------------------------------------------------
TABLE_10 = TableReference(
    table_number=10,
    title="Three-Way Engine Comparison",
    entries=[
        EngineCompEntry("sglang_vanilla", 2265, 2277, 0.00, 0.98),
        EngineCompEntry("vllm+semblend+lmcache", 3311, 1026, 1.00, 3.26),
        EngineCompEntry("sglang+semblend", 2306, 624, 1.00, 3.71),
    ],
    notes="8K tokens, cross-instruction RAG, XSum.",
)


# --------------------------------------------------------------------------
# Table 11: WildChat E2E Benchmark
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class WildChatBucket:
    """WildChat hit rate by cosine similarity bucket."""

    cosine_min: float
    cosine_max: float
    hit_rate: float
    speedup: float


TABLE_11 = TableReference(
    table_number=11,
    title="WildChat E2E Benchmark",
    entries=[
        WildChatBucket(0.50, 0.60, 0.83, 1.67),
        WildChatBucket(0.60, 0.70, 0.90, 1.71),
        WildChatBucket(0.70, 0.80, 0.80, 1.69),
        WildChatBucket(0.80, 0.90, 0.77, 1.69),
        WildChatBucket(0.90, 1.00, 0.83, 1.70),
    ],
    notes="150 real conversation pairs (>=4K chars). Overall: 82.7% hit, 1.69x.",
)


# --------------------------------------------------------------------------
# Table 12: Extended Benchmark Suite (Stage 1 summary)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ExtendedSuiteEntry:
    """Extended suite result per dataset."""

    dataset: str
    n_samples: int
    hit_rate: float
    mean_speedup: float
    hit_only_speedup: float


TABLE_12 = TableReference(
    table_number=12,
    title="Extended Benchmark Suite",
    entries=[
        ExtendedSuiteEntry("longeval", 1000, 0.826, 1.43, 1.48),
        ExtendedSuiteEntry("wikitext103", 1000, 0.757, 1.43, 1.55),
        ExtendedSuiteEntry("narrativeqa", 1000, 0.296, 1.24, 2.09),
        ExtendedSuiteEntry("triviaqa", 1000, 0.248, 1.70, 3.73),
        ExtendedSuiteEntry("scbench", 1000, 0.176, 1.86, 5.73),
    ],
    notes="vLLM+SemBlend+LMCache, Qwen2.5-7B-AWQ, A10G. Overall: 46.4% hit.",
)


# --------------------------------------------------------------------------
# Table 13: Enterprise Workload Evaluation
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class EnterpriseEntry:
    """Enterprise workload evaluation."""

    workload: str
    context_length: int
    n_samples: int
    hit_rate: float
    hit_only_speedup: float
    blended_speedup: float


TABLE_13 = TableReference(
    table_number=13,
    title="Enterprise Workload Evaluation",
    entries=[
        EnterpriseEntry("template_reports", 8192, 19, 1.00, 5.38, 5.38),
        EnterpriseEntry("customer_support", 4096, 8, 0.50, 3.47, 2.24),
        EnterpriseEntry("enterprise_rag", 8192, 120, 0.067, 4.17, 1.20),
        EnterpriseEntry("multi_tenant_api", 8192, 50, 0.08, 4.07, 1.23),
    ],
)


# --------------------------------------------------------------------------
# Table 14: Component Latency Decomposition
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ComponentLatency:
    """Component latency measurement."""

    component: str
    latency_ms_min: float
    latency_ms_max: float
    budget_ms: float


TABLE_14 = TableReference(
    table_number=14,
    title="Component Latency Decomposition",
    entries=[
        ComponentLatency("minilm_embed", 2, 3, 8),
        ComponentLatency("cosine_lookup", 1, 2, 3),
        ComponentLatency("chunk_align", 0.1, 1, 1),
        ComponentLatency("total_pipeline", 4, 7, 12),
        ComponentLatency("kv_transfer", 20, 35, 50),
    ],
    notes="Co-located pipeline, 8K tokens, 28 layers, FP16.",
)


# --------------------------------------------------------------------------
# Table 15: Similarity Threshold Ablation
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ThresholdEntry:
    """Threshold ablation entry."""

    tau: float
    speedup_2k: float
    speedup_4k: float
    speedup_8k: float
    speedup_16k: float
    coverage: float
    ppl_ratio: float


TABLE_15 = TableReference(
    table_number=15,
    title="Similarity Threshold Ablation",
    entries=[
        ThresholdEntry(0.40, 1.69, 1.63, 2.46, 3.16, 0.88, 1.117),
        ThresholdEntry(0.50, 1.69, 1.63, 2.46, 3.16, 0.88, 1.117),
        ThresholdEntry(0.60, 1.69, 1.63, 2.46, 3.16, 0.75, 1.059),
        ThresholdEntry(0.70, 1.63, 1.56, 2.24, 2.60, 0.62, 1.015),
        ThresholdEntry(0.80, 1.46, 1.43, 1.94, 2.16, 0.38, 1.019),
    ],
    notes="CNN/DailyMail, Qwen2.5-7B-AWQ, n=160. tau=0.60 is Pareto optimal.",
)


# --------------------------------------------------------------------------
# Table 16: Chunk Size Ablation
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ChunkSizeEntry:
    """Chunk size ablation entry."""

    chunk_size: int
    hit_rate: float
    speedup: float


TABLE_16 = TableReference(
    table_number=16,
    title="Chunk Size Ablation",
    entries=[
        ChunkSizeEntry(64, 0.596, 0.88),
        ChunkSizeEntry(128, 0.596, 0.88),
        ChunkSizeEntry(256, 0.596, 0.88),
        ChunkSizeEntry(512, 0.596, 0.88),
    ],
    notes="1000 cross-instruction pairs, XSum 8K. Binary alignment.",
)


# --------------------------------------------------------------------------
# Table 17: Warm-Path Latency
# --------------------------------------------------------------------------
TABLE_17 = TableReference(
    table_number=17,
    title="Inference Engine Warm-Path Latency",
    entries=[
        ComponentLatency("semblend_pipeline", 5, 7, 12),
        ComponentLatency("kv_transfer_8k", 30, 40, 50),
    ],
    notes=(
        "Cold TTFT baselines: Qwen@8K=3473ms, Qwen@16K=6604ms, LLaMA@8K=3416ms, LLaMA@16K=7396ms."
    ),
)


# --------------------------------------------------------------------------
# Aggregate lookup
# --------------------------------------------------------------------------
ALL_TABLES: dict[int, TableReference] = {
    3: TABLE_3,
    4: TABLE_4,
    5: TABLE_5,
    6: TABLE_6,
    7: TABLE_7,
    8: TABLE_8,
    9: TABLE_9,
    10: TABLE_10,
    11: TABLE_11,
    12: TABLE_12,
    13: TABLE_13,
    14: TABLE_14,
    15: TABLE_15,
    16: TABLE_16,
    17: TABLE_17,
}


def get_reference(table_number: int) -> TableReference | None:
    """Get paper reference data for a table number."""
    return ALL_TABLES.get(table_number)

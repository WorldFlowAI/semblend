"""Paper table-to-benchmark-config mapping.

Maps each of the 19 tables in the SemBlend paper to the benchmark script,
engine configuration, model, and CLI arguments needed to reproduce it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Priority(Enum):
    """Reproduction priority level."""
    P0 = "p0"  # Core results — must reproduce
    P1 = "p1"  # Comparative results
    P2 = "p2"  # Supplementary results


class Engine(Enum):
    """Inference engine backend."""
    VLLM_LMCACHE = "vllm+lmcache"
    SGLANG = "sglang"
    SGLANG_LMCACHE = "sglang+lmcache"
    VLLM_VANILLA = "vllm_vanilla"
    SGLANG_VANILLA = "sglang_vanilla"


@dataclass(frozen=True)
class TableConfig:
    """Configuration to reproduce a single paper table."""

    table_number: int
    title: str
    priority: Priority
    engine: Engine
    models: tuple[str, ...]
    script: str  # Relative to autoresearch-semblend/benchmarks/
    datasets: tuple[str, ...] = ()
    n_samples: int = 8
    context_lengths: tuple[int, ...] = ()
    extra_args: dict[str, str] = field(default_factory=dict)
    description: str = ""
    notes: str = ""


# All models used in the paper
QWEN_AWQ = "Qwen/Qwen2.5-7B-Instruct-AWQ"
LLAMA_AWQ = "meta-llama/Llama-3.1-8B-Instruct-AWQ"

# Engine images are configured via environment variables at deploy time.
# See infra/ in autoresearch-semblend for K8s manifests and Helm values.


PAPER_TABLES: dict[int, TableConfig] = {
    # Table 2: Bootstrap CIs (reference only, not runnable as a single script)

    3: TableConfig(
        table_number=3,
        title="SemBlend KV Donor Injection — TTFT Speedup 2K-32K",
        priority=Priority.P0,
        engine=Engine.VLLM_LMCACHE,
        models=(QWEN_AWQ, LLAMA_AWQ),
        script="e2e/semblend_ttft_bench.py",
        context_lengths=(2048, 4096, 8192, 16384, 24576, 32768),
        n_samples=8,
        description="Core TTFT speedup table across context lengths and models.",
    ),
    4: TableConfig(
        table_number=4,
        title="Prefix Cache Coverage Gap",
        priority=Priority.P0,
        engine=Engine.VLLM_LMCACHE,
        models=(QWEN_AWQ,),
        script="e2e/semblend_vs_lmcache_baseline.py",
        context_lengths=(2048, 8192, 16384),
        n_samples=8,
        description="vLLM prefix cache vs SemBlend across reuse scenarios.",
    ),
    5: TableConfig(
        table_number=5,
        title="Cross-Instruction RAG Benchmark",
        priority=Priority.P0,
        engine=Engine.VLLM_LMCACHE,
        models=(QWEN_AWQ,),
        script="e2e/cross_instruction_bench.py",
        context_lengths=(8192, 16384),
        n_samples=8,
        extra_args={"--variants": "4"},
        description="Same document, different instruction phrasings.",
    ),
    6: TableConfig(
        table_number=6,
        title="Quality Preservation — PPL Ratios",
        priority=Priority.P0,
        engine=Engine.VLLM_LMCACHE,
        models=(QWEN_AWQ, LLAMA_AWQ),
        script="e2e/semblend_quality_bench.py",
        datasets=("xsum", "cnn_dailymail", "wikihow"),
        context_lengths=(2048, 4096, 8192),
        n_samples=8,
        extra_args={"--max-tokens": "256"},
        description="PPL ratio (SemBlend/cold) across datasets and models.",
    ),
    7: TableConfig(
        table_number=7,
        title="Quality Across Datasets and Models — Extended PPL",
        priority=Priority.P0,
        engine=Engine.VLLM_LMCACHE,
        models=(QWEN_AWQ, LLAMA_AWQ),
        script="e2e/semblend_quality_bench.py",
        datasets=("xsum", "cnn_dailymail", "multinews", "wikihow", "samsum"),
        context_lengths=(2048, 4096, 8192),
        n_samples=8,
        extra_args={"--max-tokens": "256"},
        description="Extended quality evaluation across 5 datasets.",
    ),
    8: TableConfig(
        table_number=8,
        title="Cross-Dataset TTFT Speedup",
        priority=Priority.P0,
        engine=Engine.VLLM_LMCACHE,
        models=(QWEN_AWQ,),
        script="e2e/semblend_ttft_bench.py",
        datasets=(
            "synthetic", "xsum", "multinews",
            "cnn_dailymail", "wikihow", "samsum",
        ),
        context_lengths=(2048, 4096, 8192, 12288, 16384),
        n_samples=8,
        description="TTFT speedup across real NLP datasets.",
    ),
    9: TableConfig(
        table_number=9,
        title="Multi-Model TTFT — LLaMA-3.1-8B",
        priority=Priority.P1,
        engine=Engine.VLLM_LMCACHE,
        models=(LLAMA_AWQ,),
        script="e2e/semblend_ttft_bench.py",
        datasets=("xsum", "cnn_dailymail", "wikihow", "samsum"),
        context_lengths=(2048, 4096, 8192),
        n_samples=8,
        description="LLaMA-specific TTFT comparison.",
    ),
    10: TableConfig(
        table_number=10,
        title="Three-Way Engine Comparison",
        priority=Priority.P1,
        engine=Engine.VLLM_LMCACHE,  # Runs against all 3 engines
        models=(QWEN_AWQ,),
        script="e2e/cross_instruction_bench.py",
        context_lengths=(8192,),
        n_samples=8,
        notes=(
            "Requires 3 endpoints: SGLang vanilla, vLLM+SemBlend+LMCache, "
            "SGLang+SemBlend. Run script 3 times with different --endpoint."
        ),
        description="SGLang vanilla vs vLLM+SB vs SGLang+SB.",
    ),
    11: TableConfig(
        table_number=11,
        title="WildChat E2E Benchmark",
        priority=Priority.P1,
        engine=Engine.VLLM_LMCACHE,
        models=(QWEN_AWQ,),
        script="e2e/wildchat_semblend_bench.py",
        n_samples=150,
        extra_args={"--min-chars": "4000"},
        description="Real conversation pairs from WildChat.",
    ),
    12: TableConfig(
        table_number=12,
        title="Extended Benchmark Suite (3K+ samples)",
        priority=Priority.P0,
        engine=Engine.VLLM_LMCACHE,
        models=(QWEN_AWQ,),
        script="suite/cli.py",
        datasets=(
            "triviaqa", "narrativeqa", "longeval",
            "wikitext103", "scbench",
        ),
        n_samples=1000,
        extra_args={"--concurrency": "4", "--cooldown": "0.5"},
        description="Extended 5-dataset evaluation (Table 12).",
    ),
    13: TableConfig(
        table_number=13,
        title="Enterprise Workload Evaluation",
        priority=Priority.P2,
        engine=Engine.VLLM_LMCACHE,
        models=(QWEN_AWQ,),
        script="e2e/enterprise_rag_bench.py",
        n_samples=120,
        description="Template reports, customer support, RAG, multi-tenant.",
    ),
    14: TableConfig(
        table_number=14,
        title="Component Latency Decomposition",
        priority=Priority.P1,
        engine=Engine.VLLM_LMCACHE,
        models=(QWEN_AWQ,),
        script="e2e/component_bench.py",
        description="Pipeline stage latencies: embed, search, align, total.",
    ),
    15: TableConfig(
        table_number=15,
        title="Similarity Threshold Ablation",
        priority=Priority.P1,
        engine=Engine.VLLM_LMCACHE,
        models=(QWEN_AWQ,),
        script="e2e/threshold_ablation_bench.py",
        datasets=("cnn_dailymail",),
        context_lengths=(2048, 4096, 8192, 16384),
        n_samples=160,
        description="tau threshold sweep: 0.40, 0.50, 0.60, 0.70, 0.80.",
    ),
    16: TableConfig(
        table_number=16,
        title="Chunk Size Ablation",
        priority=Priority.P2,
        engine=Engine.VLLM_LMCACHE,
        models=(QWEN_AWQ,),
        script="e2e/chunk_size_e2e_ablation.py",
        datasets=("xsum",),
        context_lengths=(8192,),
        n_samples=1000,
        description="Chunk sizes 64/128/256/512 on cross-instruction pairs.",
    ),
    17: TableConfig(
        table_number=17,
        title="Inference Engine Warm-Path Latency",
        priority=Priority.P2,
        engine=Engine.VLLM_LMCACHE,
        models=(QWEN_AWQ, LLAMA_AWQ),
        script="e2e/component_bench.py",
        context_lengths=(8192, 16384),
        description="Pipeline overhead and KV transfer latencies.",
    ),
    18: TableConfig(
        table_number=18,
        title="RoPE Correction Validation",
        priority=Priority.P1,
        engine=Engine.VLLM_LMCACHE,
        models=(QWEN_AWQ,),
        script="e2e/rope_ablation_bench.py",
        description="RoPE position correction quality validation.",
    ),
}


def get_tables_by_priority(priority: Priority) -> list[TableConfig]:
    """Return all table configs at the given priority level."""
    return [t for t in PAPER_TABLES.values() if t.priority == priority]


def get_all_tables() -> list[TableConfig]:
    """Return all table configs sorted by table number."""
    return sorted(PAPER_TABLES.values(), key=lambda t: t.table_number)


def get_tables(
    table_numbers: list[int] | None = None,
    priority: Priority | None = None,
) -> list[TableConfig]:
    """Get table configs by number or priority filter.

    Args:
        table_numbers: Specific table numbers to include.
        priority: Filter by priority level.

    Returns:
        Matching TableConfig list, sorted by table number.
    """
    tables = get_all_tables()

    if table_numbers is not None:
        tables = [t for t in tables if t.table_number in table_numbers]

    if priority is not None:
        tables = [t for t in tables if t.priority == priority]

    return tables

"""Fuzzy chunk matching benchmark table configurations (Tables 19-27).

Maps each fuzzy matching paper table to its benchmark config.
These extend the existing PAPER_TABLES dict in paper_tables.py.

NOTE: Do NOT execute these benchmarks without explicit go-ahead.
Benchmark runs are managed separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Priority(Enum):
    P0 = "p0"
    P1 = "p1"
    P2 = "p2"


class Engine(Enum):
    VLLM_LMCACHE = "vllm+lmcache"


@dataclass(frozen=True)
class TableConfig:
    table_number: int
    title: str
    priority: Priority
    engine: Engine
    models: tuple[str, ...]
    script: str
    datasets: tuple[str, ...] = ()
    n_samples: int = 100
    context_lengths: tuple[int, ...] = ()
    extra_args: dict[str, str] = field(default_factory=dict)
    description: str = ""
    notes: str = ""


QWEN_AWQ = "Qwen/Qwen2.5-7B-Instruct-AWQ"
LLAMA_AWQ = "meta-llama/Llama-3.1-8B-Instruct-AWQ"


FUZZY_TABLES: dict[int, TableConfig] = {
    19: TableConfig(
        table_number=19,
        title="Fuzzy Matching Recovery: Exact vs Fuzzy Alignment Reuse",
        priority=Priority.P0,
        engine=Engine.VLLM_LMCACHE,
        models=(QWEN_AWQ, LLAMA_AWQ),
        script="e2e/fuzzy_recovery_bench.py",
        datasets=("shifted_prefix_xsum", "shifted_prefix_cnn"),
        n_samples=200,
        context_lengths=(2048, 4096, 8192, 16384),
        extra_args={
            "SEMBLEND_FUZZY_CHUNKS": "1",
            "SEMBLEND_FUZZY_CHUNK_OVERLAP": "0.90",
        },
        description=(
            "Shows fuzzy matching recovers 90%+ reuse in shifted-prefix "
            "scenarios where exact matching gets 0%."
        ),
    ),
    20: TableConfig(
        table_number=20,
        title="Fuzzy Matching TTFT Speedup vs Exact-Only",
        priority=Priority.P0,
        engine=Engine.VLLM_LMCACHE,
        models=(QWEN_AWQ,),
        script="e2e/fuzzy_ttft_bench.py",
        datasets=("shifted_prefix_xsum", "minor_edit_cnn"),
        n_samples=100,
        context_lengths=(2048, 4096, 8192, 16384),
        extra_args={"compare_exact": "1"},
        description=(
            "Measures additional TTFT speedup from fuzzy matching "
            "over exact-only across shifted prefix and minor edit scenarios."
        ),
    ),
    21: TableConfig(
        table_number=21,
        title="PPL by Confidence Threshold (Sweep)",
        priority=Priority.P0,
        engine=Engine.VLLM_LMCACHE,
        models=(QWEN_AWQ, LLAMA_AWQ),
        script="e2e/fuzzy_confidence_ppl_bench.py",
        datasets=("shifted_prefix_xsum", "shifted_prefix_cnn", "cross_instruction_rag"),
        n_samples=160,
        context_lengths=(4096, 8192),
        extra_args={
            "confidence_sweep": "0.70,0.80,0.85,0.90,0.95",
        },
        description=(
            "Quality-coverage tradeoff: PPL ratio and hit rate at various confidence thresholds."
        ),
    ),
    22: TableConfig(
        table_number=22,
        title="Confidence Scoring Component Ablation",
        priority=Priority.P0,
        engine=Engine.VLLM_LMCACHE,
        models=(QWEN_AWQ,),
        script="e2e/confidence_component_bench.py",
        datasets=("shifted_prefix_xsum",),
        n_samples=200,
        extra_args={"component_ablation": "1"},
        description=(
            "Incremental contribution of each confidence component: "
            "overlap-only, +position-delta, +bag-cosine, +segment-similarity."
        ),
    ),
    23: TableConfig(
        table_number=23,
        title="CacheBlend Verification for Fuzzy Matches",
        priority=Priority.P1,
        engine=Engine.VLLM_LMCACHE,
        models=(QWEN_AWQ,),
        script="e2e/fuzzy_cacheblend_bench.py",
        datasets=("shifted_prefix_xsum", "minor_edit_cnn"),
        n_samples=100,
        extra_args={"cacheblend_sweep": "1"},
        description=(
            "CacheBlend layer verification impact on PPL/TTFT across match confidence tiers."
        ),
    ),
    24: TableConfig(
        table_number=24,
        title="Fuzzy Hit Rate by Scenario Type",
        priority=Priority.P1,
        engine=Engine.VLLM_LMCACHE,
        models=(QWEN_AWQ,),
        script="e2e/fuzzy_scenario_hitrate_bench.py",
        datasets=(
            "shifted_prefix_xsum",
            "minor_edit_cnn",
            "same_topic_multinews",
            "multiturn_wildchat",
            "cross_instruction_rag",
        ),
        n_samples=100,
        extra_args={"scenario_breakdown": "1"},
        description=(
            "Hit rate comparison across scenario types: shifted prefix, "
            "minor edit, same-topic different input, multi-turn, cross-instruction."
        ),
    ),
    25: TableConfig(
        table_number=25,
        title="PQ Segment Embedding Store Scalability",
        priority=Priority.P1,
        engine=Engine.VLLM_LMCACHE,
        models=(QWEN_AWQ,),
        script="e2e/segment_scalability_bench.py",
        datasets=("shifted_prefix_xsum",),
        n_samples=50,
        extra_args={"donor_scales": "100,1000,10000,100000"},
        description=(
            "PQ segment store overhead: lookup latency, memory footprint, "
            "and pipeline latency at 100 to 100K donors."
        ),
    ),
    26: TableConfig(
        table_number=26,
        title="Position Delta Decay Function Ablation",
        priority=Priority.P2,
        engine=Engine.VLLM_LMCACHE,
        models=(QWEN_AWQ,),
        script="e2e/position_decay_ablation_bench.py",
        datasets=("minor_edit_cnn",),
        n_samples=200,
        extra_args={"decay_sweep": "exponential,linear,step,none"},
        description=(
            "Comparison of decay functions: exponential, linear, step, "
            "and no decay for position delta confidence."
        ),
    ),
    27: TableConfig(
        table_number=27,
        title="Full Ablation Matrix (8 Configurations)",
        priority=Priority.P0,
        engine=Engine.VLLM_LMCACHE,
        models=(QWEN_AWQ, LLAMA_AWQ),
        script="e2e/fuzzy_ablation_matrix_bench.py",
        datasets=("shifted_prefix_xsum", "minor_edit_cnn"),
        n_samples=100,
        extra_args={"full_ablation": "1"},
        description=(
            "8-config ablation: baseline, +fuzzy, +confidence, +segment, "
            "+cacheblend, +conf+segment, +conf+cacheblend, full stack."
        ),
        notes=(
            "Configs: (1) exact-only, (2) +fuzzy no gating, "
            "(3) +confidence gating, (4) +segment verify, "
            "(5) +CacheBlend verify, (6) +conf+segment, "
            "(7) +conf+CacheBlend, (8) full (all features)."
        ),
    ),
}

"""SemShareKV benchmark scenario definitions for autoresearch-semblend.

Defines the 4-configuration × 3-scenario matrix for comprehensive comparison:

Configurations:
  A. Vanilla vLLM + LMCache (baseline) — no SemBlend
  B. SemBlend chunk-only (existing v0.3.0) — bathtub binary per-layer
  C. SemBlend + selective recompute — graduated per-token per-layer
  D. Full recompute (gold standard) — no cache reuse, cold every time

Scenarios:
  1. Cross-instruction: same document, different instruction prefix
  2. Fuzzy chunk: same document, minor edits / shifted boundaries
  3. Cross-document: different documents, shared vocabulary/domain

Quality metrics per run:
  - PPL ratio (primary, threshold ≤ 1.065)
  - ROUGE-L (secondary)
  - L2 deviation (structural fidelity)
  - Attention Recovery (AR)
  - QA accuracy (where applicable)
  - TTFT speedup vs baseline

Each scenario uses the existing autoresearch-semblend benchmark scripts.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkConfig:
    """A single benchmark configuration."""
    name: str
    description: str
    env_vars: dict[str, str]
    is_baseline: bool = False


@dataclass(frozen=True)
class BenchmarkScenario:
    """A benchmark scenario with dataset and measurement params."""
    name: str
    description: str
    script: str  # relative to autoresearch-semblend/benchmarks/e2e/
    datasets: tuple[str, ...]
    token_lengths: tuple[int, ...]
    n_samples: int
    quality_metrics: tuple[str, ...]


# ── Configurations ──────────────────────────────────────────────────────

CONFIG_A_VANILLA = BenchmarkConfig(
    name="vanilla_lmcache",
    description="vLLM + LMCache only (no SemBlend) — baseline",
    env_vars={
        "SEMBLEND_ENABLED": "0",
        "SEMBLEND_SELECTIVE_RECOMPUTE": "0",
    },
    is_baseline=True,
)

CONFIG_B_CHUNK = BenchmarkConfig(
    name="semblend_chunk",
    description="SemBlend v0.3.0 chunk-level matching + bathtub binary recompute",
    env_vars={
        "SEMBLEND_ENABLED": "1",
        "SEMBLEND_SELECTIVE_RECOMPUTE": "0",
        "SEMBLEND_CHUNK_FAST_PATH": "1",
    },
)

CONFIG_C_SELECTIVE = BenchmarkConfig(
    name="semblend_selective",
    description="SemBlend + SemShareKV selective per-token per-layer recompute",
    env_vars={
        "SEMBLEND_ENABLED": "1",
        "SEMBLEND_SELECTIVE_RECOMPUTE": "1",
        "SEMBLEND_CHUNK_FAST_PATH": "1",
    },
)

CONFIG_D_FULL_RECOMPUTE = BenchmarkConfig(
    name="full_recompute",
    description="Full recompute (gold standard) — no cache reuse",
    env_vars={
        "SEMBLEND_ENABLED": "0",
        "LMCACHE_ENABLED": "0",
        "VLLM_DISABLE_PREFIX_CACHING": "1",
    },
)

ALL_CONFIGS = (CONFIG_A_VANILLA, CONFIG_B_CHUNK, CONFIG_C_SELECTIVE, CONFIG_D_FULL_RECOMPUTE)

# ── Scenarios ───────────────────────────────────────────────────────────

SCENARIO_1_CROSS_INSTRUCTION = BenchmarkScenario(
    name="cross_instruction",
    description=(
        "Same document, different instruction prefix. "
        "LMCache gets 0% hits due to chunk boundary shift. "
        "SemBlend finds donor via embedding similarity + token swap."
    ),
    script="cross_instruction_bench.py",
    datasets=("xsum", "cnn"),
    token_lengths=(4096, 8192),
    n_samples=200,
    quality_metrics=("ppl_ratio", "rouge_l", "ttft_speedup"),
)

SCENARIO_2_FUZZY_CHUNK = BenchmarkScenario(
    name="fuzzy_chunk",
    description=(
        "Same document with minor edits or shifted chunk boundaries. "
        "Tests SemBlend's per-token greedy matching for boundary recovery. "
        "Sub-scenarios: shifted_prefix, minor_edit, negative_control."
    ),
    script="fuzzy_recovery_bench.py",
    datasets=("xsum", "cnn"),
    token_lengths=(4096, 8192),
    n_samples=200,
    quality_metrics=("ppl_ratio", "hit_rate", "ttft_speedup", "false_positive_rate"),
)

SCENARIO_3_CROSS_DOCUMENT = BenchmarkScenario(
    name="cross_document",
    description=(
        "Different documents with shared vocabulary/domain. "
        "Tests whether token-level matching finds meaningful KV reuse "
        "across different texts. This is where SemShareKV's per-token "
        "approach should show value over chunk-level matching."
    ),
    script="cross_document_bench.py",  # to be created
    datasets=("xsum", "multinews", "cnn"),
    token_lengths=(4096, 8192),
    n_samples=200,
    quality_metrics=(
        "ppl_ratio", "token_match_ratio", "ttft_speedup",
        "l2_deviation", "attention_recovery",
    ),
)

ALL_SCENARIOS = (
    SCENARIO_1_CROSS_INSTRUCTION,
    SCENARIO_2_FUZZY_CHUNK,
    SCENARIO_3_CROSS_DOCUMENT,
)


# ── Benchmark Matrix ───────────────────────────────────────────────────

def get_benchmark_matrix() -> list[dict]:
    """Generate the full 4-config × 3-scenario benchmark matrix.

    Returns list of dicts, each representing one benchmark run with:
    - config: BenchmarkConfig
    - scenario: BenchmarkScenario
    - token_length: int
    - dataset: str
    """
    matrix = []
    for scenario in ALL_SCENARIOS:
        for config in ALL_CONFIGS:
            for token_length in scenario.token_lengths:
                for dataset in scenario.datasets:
                    matrix.append({
                        "config": config,
                        "scenario": scenario,
                        "token_length": token_length,
                        "dataset": dataset,
                        "run_id": (
                            f"{config.name}__{scenario.name}__{dataset}"
                            f"__{token_length}"
                        ),
                    })
    return matrix


def print_matrix_summary() -> None:
    """Print a human-readable summary of the benchmark matrix."""
    matrix = get_benchmark_matrix()
    print(f"\nSemShareKV Benchmark Matrix: {len(matrix)} total runs")
    print("=" * 70)

    for scenario in ALL_SCENARIOS:
        runs = [r for r in matrix if r["scenario"] == scenario]
        print(f"\n{scenario.name}: {scenario.description[:60]}...")
        print(f"  Script: {scenario.script}")
        print(f"  Datasets: {', '.join(scenario.datasets)}")
        print(f"  Token lengths: {', '.join(str(t) for t in scenario.token_lengths)}")
        print(f"  Quality metrics: {', '.join(scenario.quality_metrics)}")
        print(f"  Runs: {len(runs)} ({len(ALL_CONFIGS)} configs × "
              f"{len(scenario.token_lengths)} lengths × "
              f"{len(scenario.datasets)} datasets)")

    print(f"\nConfigurations:")
    for cfg in ALL_CONFIGS:
        baseline_tag = " [BASELINE]" if cfg.is_baseline else ""
        print(f"  {cfg.name}{baseline_tag}: {cfg.description}")


if __name__ == "__main__":
    print_matrix_summary()

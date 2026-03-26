"""Result comparison and delta reporting.

Loads paper reference values and v0.2.0 result JSONs, computes deltas
for each metric, flags regressions >10%, and generates markdown tables.

Usage:
    python -m benchmarks.suite.compare \\
        --results-dir benchmarks/results/v0.2.0 \\
        --tables 3 5 8 12
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from benchmarks.suite.paper_results import (
    ALL_TABLES,
    CrossDatasetEntry,
    ExtendedSuiteEntry,
    PPLEntry,
    TTFTEntry,
    get_reference,
)

logger = logging.getLogger(__name__)

REGRESSION_THRESHOLD = 0.10  # 10% regression threshold


@dataclass(frozen=True)
class MetricDelta:
    """Delta between paper and reproduction for a single metric."""

    metric_name: str
    label: str  # Human-readable context (e.g., "8K Qwen")
    paper_value: float
    repro_value: float
    absolute_delta: float
    relative_delta: float  # As fraction, not percentage
    is_regression: bool

    @property
    def delta_pct(self) -> float:
        return self.relative_delta * 100


@dataclass
class TableComparison:
    """Comparison results for one paper table."""

    table_number: int
    title: str
    deltas: list[MetricDelta]
    missing_data: bool = False
    notes: str = ""

    @property
    def has_regressions(self) -> bool:
        return any(d.is_regression for d in self.deltas)

    @property
    def regression_count(self) -> int:
        return sum(1 for d in self.deltas if d.is_regression)


def _compute_delta(
    metric_name: str,
    label: str,
    paper: float,
    repro: float,
    higher_is_better: bool = True,
) -> MetricDelta:
    """Compute delta between paper and reproduction values."""
    absolute = repro - paper
    relative = (repro - paper) / paper if paper != 0 else 0.0

    if higher_is_better:
        is_regression = relative < -REGRESSION_THRESHOLD
    else:
        # For metrics where lower is better (e.g., PPL ratio)
        is_regression = relative > REGRESSION_THRESHOLD

    return MetricDelta(
        metric_name=metric_name,
        label=label,
        paper_value=paper,
        repro_value=repro,
        absolute_delta=absolute,
        relative_delta=relative,
        is_regression=is_regression,
    )


def load_result_json(path: Path) -> dict:
    """Load a benchmark result JSON file."""
    return json.loads(path.read_text())


def find_result_files(results_dir: Path, table_number: int) -> list[Path]:
    """Find result files for a given table number."""
    patterns = [
        f"table_{table_number}_*.json",
        f"table{table_number}_*.json",
        f"t{table_number}_*.json",
    ]
    found = []
    for pattern in patterns:
        found.extend(results_dir.glob(pattern))
    return sorted(found)


def compare_table_3(result: dict) -> list[MetricDelta]:
    """Compare Table 3: TTFT Speedup 2K–32K."""
    ref = get_reference(3)
    if ref is None:
        return []

    deltas = []
    results_by_key = {}

    # Index result data by (model_short, context_length)
    for entry in result.get("entries", []):
        key = (entry.get("model", "qwen"), entry.get("context_length", 0))
        results_by_key[key] = entry

    for paper_entry in ref.entries:
        if not isinstance(paper_entry, TTFTEntry):
            continue

        model_label = paper_entry.model or "unknown"
        label = f"{model_label} {paper_entry.context_length // 1024}K"

        key = (model_label, paper_entry.context_length)
        repro = results_by_key.get(key, {})

        if repro:
            deltas.append(
                _compute_delta(
                    "speedup",
                    label,
                    paper_entry.speedup,
                    repro.get("speedup", 0),
                    higher_is_better=True,
                )
            )
            deltas.append(
                _compute_delta(
                    "hit_rate",
                    label,
                    paper_entry.hit_rate,
                    repro.get("hit_rate", 0),
                    higher_is_better=True,
                )
            )

    return deltas


def compare_table_12(result: dict) -> list[MetricDelta]:
    """Compare Table 12: Extended Benchmark Suite."""
    ref = get_reference(12)
    if ref is None:
        return []

    deltas = []
    results_by_ds = {}

    for entry in result.get("results", []):
        ds = entry.get("dataset", "")
        results_by_ds[ds] = entry

    for paper_entry in ref.entries:
        if not isinstance(paper_entry, ExtendedSuiteEntry):
            continue

        repro = results_by_ds.get(paper_entry.dataset, {})
        if not repro:
            continue

        deltas.append(
            _compute_delta(
                "hit_rate",
                paper_entry.dataset,
                paper_entry.hit_rate,
                repro.get("hit_rate", 0),
                higher_is_better=True,
            )
        )
        deltas.append(
            _compute_delta(
                "mean_speedup",
                paper_entry.dataset,
                paper_entry.mean_speedup,
                repro.get("mean_speedup", 0),
                higher_is_better=True,
            )
        )

    return deltas


def compare_table_generic_ttft(
    table_number: int,
    result: dict,
) -> list[MetricDelta]:
    """Generic comparator for tables with CrossDatasetEntry or TTFTEntry."""
    ref = get_reference(table_number)
    if ref is None:
        return []

    deltas = []
    results_by_key = {}

    for entry in result.get("entries", []):
        key = (
            entry.get("dataset", ""),
            entry.get("context_length", 0),
        )
        results_by_key[key] = entry

    for paper_entry in ref.entries:
        if isinstance(paper_entry, CrossDatasetEntry):
            label = f"{paper_entry.dataset} {paper_entry.context_length // 1024}K"
            key = (paper_entry.dataset, paper_entry.context_length)
            repro = results_by_key.get(key, {})
            if repro:
                deltas.append(
                    _compute_delta(
                        "speedup",
                        label,
                        paper_entry.speedup,
                        repro.get("speedup", 0),
                        higher_is_better=True,
                    )
                )
                deltas.append(
                    _compute_delta(
                        "hit_rate",
                        label,
                        paper_entry.hit_rate,
                        repro.get("hit_rate", 0),
                        higher_is_better=True,
                    )
                )
        elif isinstance(paper_entry, TTFTEntry):
            label = f"{paper_entry.context_length // 1024}K"
            key = ("", paper_entry.context_length)
            repro = results_by_key.get(key, {})
            if repro:
                deltas.append(
                    _compute_delta(
                        "speedup",
                        label,
                        paper_entry.speedup,
                        repro.get("speedup", 0),
                        higher_is_better=True,
                    )
                )

    return deltas


def compare_table_ppl(
    table_number: int,
    result: dict,
) -> list[MetricDelta]:
    """Compare PPL ratio tables (6, 7)."""
    ref = get_reference(table_number)
    if ref is None:
        return []

    deltas = []
    results_by_key = {}

    for entry in result.get("entries", []):
        key = (
            entry.get("dataset", ""),
            entry.get("model", ""),
            entry.get("context_length", 0),
        )
        results_by_key[key] = entry

    for paper_entry in ref.entries:
        if not isinstance(paper_entry, PPLEntry):
            continue

        label = f"{paper_entry.dataset} {paper_entry.model} {paper_entry.context_length // 1024}K"
        key = (paper_entry.dataset, paper_entry.model, paper_entry.context_length)
        repro = results_by_key.get(key, {})

        if repro:
            deltas.append(
                _compute_delta(
                    "ppl_ratio",
                    label,
                    paper_entry.ppl_ratio,
                    repro.get("ppl_ratio", 0),
                    higher_is_better=False,  # Lower PPL ratio = better
                )
            )

    return deltas


# Map table numbers to comparator functions
_COMPARATORS = {
    3: compare_table_3,
    5: lambda r: compare_table_generic_ttft(5, r),
    6: lambda r: compare_table_ppl(6, r),
    7: lambda r: compare_table_ppl(7, r),
    8: lambda r: compare_table_generic_ttft(8, r),
    9: lambda r: compare_table_generic_ttft(9, r),
    12: compare_table_12,
}


def compare_table(
    table_number: int,
    result: dict,
) -> TableComparison:
    """Run comparison for a single table."""
    ref = get_reference(table_number)
    title = ref.title if ref else f"Table {table_number}"

    comparator = _COMPARATORS.get(table_number)
    if comparator is None:
        return TableComparison(
            table_number=table_number,
            title=title,
            deltas=[],
            missing_data=True,
            notes=f"No comparator implemented for Table {table_number}.",
        )

    deltas = comparator(result)

    return TableComparison(
        table_number=table_number,
        title=title,
        deltas=deltas,
    )


def compare_all(
    results_dir: Path,
    table_numbers: list[int] | None = None,
) -> list[TableComparison]:
    """Compare all available tables against paper values.

    Args:
        results_dir: Directory containing v0.2.0 result JSON files.
        table_numbers: Specific tables to compare (all if None).

    Returns:
        List of TableComparison results.
    """
    tables_to_check = table_numbers or sorted(ALL_TABLES.keys())
    comparisons = []

    for tn in tables_to_check:
        files = find_result_files(results_dir, tn)
        if not files:
            comparisons.append(
                TableComparison(
                    table_number=tn,
                    title=get_reference(tn).title if get_reference(tn) else f"Table {tn}",
                    deltas=[],
                    missing_data=True,
                    notes="No result files found.",
                )
            )
            continue

        # Use the most recent file
        result = load_result_json(files[-1])
        comparisons.append(compare_table(tn, result))

    return comparisons


def format_comparison_markdown(comparisons: list[TableComparison]) -> str:
    """Generate a markdown report from comparison results."""
    lines = [
        "# SemBlend v0.2.0 vs Paper — Comparison Report",
        "",
    ]

    # Summary
    total_metrics = sum(len(c.deltas) for c in comparisons)
    total_regressions = sum(c.regression_count for c in comparisons)
    missing = sum(1 for c in comparisons if c.missing_data)

    lines.extend(
        [
            "## Summary",
            "",
            f"- **Tables compared:** {len(comparisons) - missing}/{len(comparisons)}",
            f"- **Metrics evaluated:** {total_metrics}",
            f"- **Regressions (>{REGRESSION_THRESHOLD * 100:.0f}%):** {total_regressions}",
            f"- **Missing data:** {missing} tables",
            "",
        ]
    )

    # Per-table details
    for comp in comparisons:
        status = (
            "MISSING" if comp.missing_data else ("REGRESSION" if comp.has_regressions else "OK")
        )

        lines.extend(
            [
                f"## Table {comp.table_number}: {comp.title} [{status}]",
                "",
            ]
        )

        if comp.missing_data:
            lines.append(f"_{comp.notes}_\n")
            continue

        if not comp.deltas:
            lines.append("_No metrics to compare._\n")
            continue

        lines.extend(
            [
                "| Metric | Context | Paper | v0.2.0 | Delta | % | Flag |",
                "|--------|---------|-------|--------|-------|---|------|",
            ]
        )

        for d in comp.deltas:
            flag = "REGR" if d.is_regression else ""
            lines.append(
                f"| {d.metric_name} | {d.label} | "
                f"{d.paper_value:.3f} | {d.repro_value:.3f} | "
                f"{d.absolute_delta:+.3f} | {d.delta_pct:+.1f}% | {flag} |"
            )

        lines.append("")

    # Regression details
    if total_regressions > 0:
        lines.extend(
            [
                "## Regressions Requiring Investigation",
                "",
            ]
        )
        for comp in comparisons:
            for d in comp.deltas:
                if d.is_regression:
                    lines.append(
                        f"- **Table {comp.table_number} / {d.label}**: "
                        f"{d.metric_name} dropped from {d.paper_value:.3f} "
                        f"to {d.repro_value:.3f} ({d.delta_pct:+.1f}%)"
                    )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    """CLI entry point for comparison."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare v0.2.0 results against paper values",
    )
    parser.add_argument(
        "--results-dir",
        default="benchmarks/results/v0.2.0",
        help="Directory containing v0.2.0 result JSON files",
    )
    parser.add_argument(
        "--tables",
        type=int,
        nargs="*",
        default=None,
        help="Table numbers to compare (all if omitted)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output markdown file (prints to stdout if omitted)",
    )

    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        logger.warning(f"Results directory not found: {results_dir}")
        logger.info("Run benchmarks first with: python -m benchmarks.suite.reproduce")

    comparisons = compare_all(results_dir, args.tables)
    report = format_comparison_markdown(comparisons)

    if args.output:
        Path(args.output).write_text(report)
        print(f"Report written to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()

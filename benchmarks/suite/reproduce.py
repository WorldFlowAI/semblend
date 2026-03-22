"""Master reproduction script for SemBlend paper tables.

Generates and optionally runs benchmark commands for reproducing each paper
table, with full reproducibility metadata in every result JSON.

Usage:
    # List commands for all P0 tables
    python -m benchmarks.suite.reproduce --priority p0 --dry-run

    # Run Table 3 and 5
    python -m benchmarks.suite.reproduce --tables 3 5 \\
        --endpoint http://localhost:8000 \\
        --autoresearch-dir ~/dev/worldflowai/autoresearch-semblend

    # Run all tables, save results with metadata
    python -m benchmarks.suite.reproduce --all \\
        --endpoint http://localhost:8000 \\
        --autoresearch-dir ~/dev/worldflowai/autoresearch-semblend

    # Compare results after running
    python -m benchmarks.suite.compare --results-dir benchmarks/results/v0.2.0
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

from benchmarks.suite.metadata import collect_metadata
from benchmarks.suite.paper_tables import (
    Priority,
    TableConfig,
    get_tables,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = "benchmarks/results/v0.2.0"


def build_command(
    table: TableConfig,
    endpoint: str,
    model_override: str | None = None,
    autoresearch_dir: str = ".",
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> list[list[str]]:
    """Build CLI commands to reproduce a paper table.

    Returns a list of commands (one per model if the table uses multiple).
    Each command is a list of string arguments.
    """
    commands = []
    script_path = str(Path(autoresearch_dir) / "benchmarks" / table.script)
    models = table.models if model_override is None else (model_override,)

    for model in models:
        if table.script == "suite/cli.py":
            # Use the benchmark suite runner
            cmd = [
                sys.executable, "-m", "benchmarks.suite", "run",
                "--endpoint", endpoint,
                "--model", model,
                "--output-dir", output_dir,
                "--run-name", f"table_{table.table_number}",
            ]
            if table.datasets:
                cmd.extend(["--datasets", *table.datasets])
            if table.n_samples:
                cmd.extend(["--n", str(table.n_samples)])
            for key, val in table.extra_args.items():
                cmd.extend([key, val])
        else:
            # Use legacy e2e scripts
            cmd = [
                sys.executable, script_path,
                "--endpoint", endpoint,
                "--model", model,
            ]
            if table.context_lengths:
                for length in table.context_lengths:
                    cmd.extend(["--target-tokens", str(length)])
            for key, val in table.extra_args.items():
                cmd.extend([key, val])

        commands.append(cmd)

    return commands


def run_table(
    table: TableConfig,
    endpoint: str,
    autoresearch_dir: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    model_override: str | None = None,
    timeout_seconds: int = 3600,
) -> list[Path]:
    """Run benchmarks for a single paper table and save results with metadata.

    Returns list of result file paths created.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    commands = build_command(
        table, endpoint, model_override, autoresearch_dir, output_dir,
    )
    result_files = []

    metadata = collect_metadata()

    for cmd in commands:
        model = _extract_model_from_cmd(cmd)
        logger.info(
            f"Table {table.table_number}: {table.title} "
            f"(model={model})"
        )
        logger.info(f"  Command: {' '.join(cmd)}")

        try:
            # Snapshot existing files before running subprocess
            existing_files = set(out_path.glob("*.json"))

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=autoresearch_dir,
            )

            if result.returncode != 0:
                logger.error(
                    f"  FAILED (exit {result.returncode}): {result.stderr[:500]}"
                )
                error_path = _save_error_record(
                    table, model, result.stderr, metadata, out_path,
                )
                result_files.append(error_path)
                continue

            logger.info("  Completed successfully")

            # Detect new files by diffing against pre-run snapshot
            new_files = sorted(set(out_path.glob("*.json")) - existing_files)
            if new_files:
                result_path = new_files[-1]
                _inject_metadata(result_path, table, metadata)
                result_files.append(result_path)
                logger.info(f"  Results: {result_path}")

        except subprocess.TimeoutExpired:
            logger.error(f"  TIMEOUT after {timeout_seconds}s")
        except FileNotFoundError as e:
            logger.error(f"  Script not found: {e}")

    return result_files


def _extract_model_from_cmd(cmd: list[str]) -> str:
    """Extract model name from a command list."""
    for i, arg in enumerate(cmd):
        if arg == "--model" and i + 1 < len(cmd):
            return cmd[i + 1]
    return "unknown"


def _inject_metadata(
    result_path: Path,
    table: TableConfig,
    metadata,
) -> None:
    """Inject reproducibility metadata into an existing result JSON."""
    data = json.loads(result_path.read_text())

    data["_reproducibility"] = {
        "paper_table": table.table_number,
        "paper_table_title": table.title,
        "priority": table.priority.value,
        **metadata.to_dict(),
    }

    result_path.write_text(json.dumps(data, indent=2, default=str))


def _save_error_record(
    table: TableConfig,
    model: str,
    stderr: str,
    metadata,
    output_dir: Path,
) -> Path:
    """Save an error record for a failed benchmark run."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_model = model.replace("/", "_")
    path = output_dir / f"table_{table.table_number}_ERROR_{safe_model}_{timestamp}.json"

    record = {
        "status": "error",
        "paper_table": table.table_number,
        "paper_table_title": table.title,
        "model": model,
        "error": stderr[:2000],
        "_reproducibility": metadata.to_dict(),
    }

    path.write_text(json.dumps(record, indent=2, default=str))
    return path


def print_plan(
    tables: list[TableConfig],
    endpoint: str,
    autoresearch_dir: str,
    output_dir: str,
) -> None:
    """Print the reproduction plan without running anything."""
    print("=" * 80)
    print("SEMBLEND PAPER REPRODUCTION PLAN")
    print("=" * 80)
    print(f"Endpoint:     {endpoint}")
    print(f"Autoresearch: {autoresearch_dir}")
    print(f"Output:       {output_dir}")
    print(f"Tables:       {len(tables)}")
    print()

    for priority in Priority:
        group = [t for t in tables if t.priority == priority]
        if not group:
            continue

        print(f"--- {priority.value.upper()} ({len(group)} tables) ---")
        for table in group:
            models_str = ", ".join(
                m.split("/")[-1] for m in table.models
            )
            engine_str = table.engine.value
            print(
                f"  Table {table.table_number:2d}: {table.title}"
            )
            print(
                f"           Engine={engine_str}  Models={models_str}  "
                f"N={table.n_samples}"
            )
            if table.notes:
                print(f"           NOTE: {table.notes}")

            commands = build_command(
                table, endpoint, autoresearch_dir=autoresearch_dir,
                output_dir=output_dir,
            )
            for cmd in commands:
                print(f"           $ {' '.join(cmd)}")
            print()


def main() -> None:
    """CLI entry point for reproduction."""
    parser = argparse.ArgumentParser(
        prog="reproduce",
        description="Reproduce SemBlend paper benchmark tables",
    )

    # Table selection (mutually exclusive group)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--tables",
        type=int,
        nargs="+",
        help="Specific table numbers to reproduce (e.g., 3 5 8 12)",
    )
    selection.add_argument(
        "--priority",
        choices=["p0", "p1", "p2"],
        help="Run all tables at this priority level",
    )
    selection.add_argument(
        "--all",
        action="store_true",
        help="Run all tables",
    )

    # Endpoints
    parser.add_argument(
        "--endpoint",
        default="http://localhost:8000",
        help="Inference endpoint URL",
    )
    parser.add_argument(
        "--autoresearch-dir",
        default=str(
            Path.home() / "dev" / "worldflowai" / "autoresearch-semblend"
        ),
        help="Path to autoresearch-semblend repo",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for results",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override model for all tables",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout per table in seconds (default: 3600)",
    )

    # Modes
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan without running benchmarks",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip pre-flight code verification (not recommended)",
    )
    parser.add_argument(
        "--verify-pods",
        nargs="*",
        default=None,
        help="K8s pod names to verify before running (auto-detected if omitted)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Resolve table selection
    if args.all:
        tables = get_tables()
    elif args.priority:
        priority = Priority(args.priority)
        tables = get_tables(priority=priority)
    else:
        tables = get_tables(table_numbers=args.tables)

    if not tables:
        logger.error("No tables matched the selection")
        sys.exit(1)

    if args.dry_run:
        print_plan(tables, args.endpoint, args.autoresearch_dir, args.output_dir)
        return

    # Verify autoresearch dir exists
    ar_dir = Path(args.autoresearch_dir)
    if not ar_dir.exists():
        logger.error(f"Autoresearch directory not found: {ar_dir}")
        sys.exit(1)

    # Pre-flight verification: ensure pods run v0.2.0 code
    if not args.skip_verify:
        try:
            from benchmarks.suite.verify import verify_all_pods

            pods = args.verify_pods
            if pods is None:
                # Auto-detect: check common pod names
                pods = ["vllm-semblend-v020", "sglang-semblend"]

            pod_tuples = [(p, p) for p in pods]
            results = verify_all_pods(pod_tuples)

            if not all(r.passed for r in results):
                logger.error(
                    "Pre-flight verification FAILED. Pods are running stale code. "
                    "Fix errors above or use --skip-verify to bypass (not recommended)."
                )
                sys.exit(1)
        except ImportError:
            logger.warning("verify module not available, skipping pre-flight check")
        except Exception as e:
            logger.warning(f"Pre-flight verification skipped: {e}")

    # Run benchmarks
    all_results = []
    start_time = time.time()

    print(f"Reproducing {len(tables)} paper tables...")
    print()

    for table in tables:
        print(f"[Table {table.table_number}] {table.title}")
        result_files = run_table(
            table,
            endpoint=args.endpoint,
            autoresearch_dir=str(ar_dir),
            output_dir=args.output_dir,
            model_override=args.model,
            timeout_seconds=args.timeout,
        )
        all_results.extend(result_files)
        print()

    elapsed = time.time() - start_time
    print("=" * 80)
    print(f"Reproduction complete: {len(all_results)} result files")
    print(f"Elapsed: {elapsed / 60:.1f} minutes")
    print(f"Results: {args.output_dir}")
    print()
    print("Next step: compare against paper values:")
    print(f"  python -m benchmarks.suite.compare --results-dir {args.output_dir}")


if __name__ == "__main__":
    main()

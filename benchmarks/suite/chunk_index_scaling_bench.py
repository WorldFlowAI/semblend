"""ChunkIndex scaling benchmark: memory and latency at 10K/50K/100K donors.

Measures ChunkIndex memory usage, add_donor_chunks latency, and lookup
latency as the index scales from 10K to 100K donors. Uses synthetic
token sequences with realistic chunk sizes (256 tokens/chunk, ~30 chunks
per donor ≈ 7680 tokens, typical for 4K-8K context).

No GPU or inference engine required — runs purely on CPU.

Usage:
    # Dry-run
    python -m benchmarks.suite.chunk_index_scaling_bench --dry-run

    # Run full scaling benchmark
    python -m benchmarks.suite.chunk_index_scaling_bench \
        --scales 10000 50000 100000 \
        --output-dir benchmarks/results/v0.3.0/chunk_index_scaling

    # Quick test at smaller scales
    python -m benchmarks.suite.chunk_index_scaling_bench \
        --scales 1000 5000 10000 \
        --output-dir benchmarks/results/v0.3.0/chunk_index_scaling
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Realistic donor parameters
CHUNK_SIZE = 256  # Production chunk size
CHUNKS_PER_DONOR = 30  # ~7680 tokens ≈ 4K-8K context
TOKENS_PER_DONOR = CHUNK_SIZE * CHUNKS_PER_DONOR

# Lookup benchmark parameters
LOOKUP_ITERATIONS = 1000
FIND_MATCHING_ITERATIONS = 200


@dataclass(frozen=True)
class ScaleResult:
    """Results for a single scale point."""
    n_donors: int
    n_entries: int  # Total chunk entries in index
    n_unique_hashes: int
    memory_mb: float
    memory_estimated_mb: float
    # Latency (milliseconds)
    add_donor_mean_us: float  # Microseconds per add_donor_chunks
    add_donor_p95_us: float
    lookup_chunk_mean_us: float  # Single chunk lookup
    lookup_chunk_p95_us: float
    find_matching_mean_us: float  # Full sequence find_matching_chunks
    find_matching_p95_us: float


def _measure_memory_mb() -> float:
    """Get current process RSS in MB."""
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / 1024.0  # macOS returns KB
    except Exception:
        return 0.0


def _make_donor_tokens(donor_idx: int) -> list[int]:
    """Generate unique token sequence for a donor.

    Each donor has unique chunks to avoid hash collisions masking
    the true entry count. Uses donor_idx to seed the offset.
    """
    base = donor_idx * TOKENS_PER_DONOR
    return list(range(base, base + TOKENS_PER_DONOR))


def _run_scale_point(n_donors: int) -> ScaleResult:
    """Run benchmark at a specific scale."""
    from semblend_core.chunk_index import ChunkIndex

    logger.info("=== Scale: %d donors ===", n_donors)

    gc.collect()
    mem_before = _measure_memory_mb()

    idx = ChunkIndex(max_donors=n_donors + 1000, chunk_size=CHUNK_SIZE)

    # --- Measure add_donor_chunks latency ---
    add_latencies: list[float] = []

    for i in range(n_donors):
        tokens = _make_donor_tokens(i)
        t0 = time.monotonic()
        idx.add_donor_chunks(f"d{i}", tokens)
        elapsed_us = (time.monotonic() - t0) * 1_000_000
        add_latencies.append(elapsed_us)

        if (i + 1) % 10000 == 0:
            logger.info("  Added %d/%d donors", i + 1, n_donors)

    gc.collect()
    mem_after = _measure_memory_mb()

    n_entries = idx.num_entries
    n_unique = idx.num_unique_hashes
    memory_actual = mem_after - mem_before
    memory_estimated = idx.estimated_memory_bytes() / (1024 * 1024)

    logger.info(
        "  Entries: %d, Unique hashes: %d, Memory: %.1f MB (est: %.1f MB)",
        n_entries, n_unique, memory_actual, memory_estimated,
    )

    # --- Measure single chunk lookup latency ---
    # Look up chunks from donors at various positions in the index
    lookup_latencies: list[float] = []
    sample_donors = [
        n_donors // 4, n_donors // 2, 3 * n_donors // 4, n_donors - 1,
    ]

    for _ in range(LOOKUP_ITERATIONS):
        for donor_idx in sample_donors:
            tokens = _make_donor_tokens(donor_idx)
            chunk = tokens[:CHUNK_SIZE]  # First chunk
            t0 = time.monotonic()
            idx.lookup_chunk(chunk)
            elapsed_us = (time.monotonic() - t0) * 1_000_000
            lookup_latencies.append(elapsed_us)

    # --- Measure find_matching_chunks latency ---
    find_latencies: list[float] = []

    for _ in range(FIND_MATCHING_ITERATIONS):
        # Target shares ~50% chunks with a random donor + 50% novel
        donor_idx = n_donors // 2
        donor_tokens = _make_donor_tokens(donor_idx)
        novel_tokens = list(range(
            n_donors * TOKENS_PER_DONOR,
            n_donors * TOKENS_PER_DONOR + TOKENS_PER_DONOR // 2,
        ))
        target = donor_tokens[:TOKENS_PER_DONOR // 2] + novel_tokens

        t0 = time.monotonic()
        idx.find_matching_chunks(target)
        elapsed_us = (time.monotonic() - t0) * 1_000_000
        find_latencies.append(elapsed_us)

    # Compute statistics
    sorted_add = sorted(add_latencies)
    sorted_lookup = sorted(lookup_latencies)
    sorted_find = sorted(find_latencies)

    def p95(data: list[float]) -> float:
        idx = int(len(data) * 0.95)
        return data[min(idx, len(data) - 1)]

    result = ScaleResult(
        n_donors=n_donors,
        n_entries=n_entries,
        n_unique_hashes=n_unique,
        memory_mb=max(memory_actual, 0),
        memory_estimated_mb=memory_estimated,
        add_donor_mean_us=statistics.mean(add_latencies),
        add_donor_p95_us=p95(sorted_add),
        lookup_chunk_mean_us=statistics.mean(lookup_latencies),
        lookup_chunk_p95_us=p95(sorted_lookup),
        find_matching_mean_us=statistics.mean(find_latencies),
        find_matching_p95_us=p95(sorted_find),
    )

    logger.info(
        "  add_donor: %.1f us mean, %.1f us p95",
        result.add_donor_mean_us, result.add_donor_p95_us,
    )
    logger.info(
        "  lookup_chunk: %.1f us mean, %.1f us p95",
        result.lookup_chunk_mean_us, result.lookup_chunk_p95_us,
    )
    logger.info(
        "  find_matching: %.1f us mean, %.1f us p95",
        result.find_matching_mean_us, result.find_matching_p95_us,
    )

    return result


def run_benchmark(
    scales: list[int] = None,
    output_dir: str = "benchmarks/results/v0.3.0/chunk_index_scaling",
    dry_run: bool = False,
) -> None:
    """Run the ChunkIndex scaling benchmark.

    Args:
        scales: List of donor counts to benchmark (default: [10K, 50K, 100K]).
        output_dir: Where to save result JSONs.
        dry_run: Print plan without running.
    """
    if scales is None:
        scales = [10_000, 50_000, 100_000]

    if dry_run:
        print("Would run ChunkIndex scaling benchmark:")
        print(f"  Scales: {scales}")
        print(f"  Chunk size: {CHUNK_SIZE}")
        print(f"  Chunks/donor: {CHUNKS_PER_DONOR}")
        print(f"  Tokens/donor: {TOKENS_PER_DONOR}")
        print(f"  Lookup iterations: {LOOKUP_ITERATIONS}")
        print(f"  Output: {output_dir}")
        return

    from benchmarks.suite.metadata import collect_metadata

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results: list[ScaleResult] = []
    for n_donors in scales:
        result = _run_scale_point(n_donors)
        results.append(result)

    # Save results
    result_data = {
        "_reproducibility": collect_metadata().to_dict(),
        "benchmark": "chunk_index_scaling",
        "chunk_size": CHUNK_SIZE,
        "chunks_per_donor": CHUNKS_PER_DONOR,
        "scales": [
            {
                "n_donors": r.n_donors,
                "n_entries": r.n_entries,
                "n_unique_hashes": r.n_unique_hashes,
                "memory_mb": round(r.memory_mb, 1),
                "memory_estimated_mb": round(r.memory_estimated_mb, 1),
                "add_donor_mean_us": round(r.add_donor_mean_us, 1),
                "add_donor_p95_us": round(r.add_donor_p95_us, 1),
                "lookup_chunk_mean_us": round(r.lookup_chunk_mean_us, 1),
                "lookup_chunk_p95_us": round(r.lookup_chunk_p95_us, 1),
                "find_matching_mean_us": round(r.find_matching_mean_us, 1),
                "find_matching_p95_us": round(r.find_matching_p95_us, 1),
            }
            for r in results
        ],
    }
    result_file = out_path / "chunk_index_scaling.json"
    result_file.write_text(json.dumps(result_data, indent=2))
    logger.info("Saved results to %s", result_file)

    # Print scaling table
    print("\n=== ChunkIndex Scaling Benchmark ===\n")
    print(f"{'Donors':>10} {'Entries':>10} {'Memory MB':>10} "
          f"{'Add (us)':>10} {'Lookup (us)':>12} {'Find (us)':>12}")
    print("-" * 76)
    for r in results:
        print(
            f"{r.n_donors:>10,} {r.n_entries:>10,} "
            f"{r.memory_estimated_mb:>10.1f} "
            f"{r.add_donor_mean_us:>10.1f} "
            f"{r.lookup_chunk_mean_us:>12.1f} "
            f"{r.find_matching_mean_us:>12.1f}"
        )

    # Verify targets
    print("\n--- Target Verification ---")
    for r in results:
        mem_ok = r.memory_estimated_mb < 300  # < 300MB for 100K
        lookup_ok = r.lookup_chunk_mean_us < 100  # < 0.1ms
        print(
            f"  {r.n_donors:>6,} donors: "
            f"memory={'PASS' if mem_ok else 'FAIL'} ({r.memory_estimated_mb:.0f}MB), "
            f"lookup={'PASS' if lookup_ok else 'FAIL'} ({r.lookup_chunk_mean_us:.0f}us)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ChunkIndex scaling benchmark: memory + latency curves",
    )
    parser.add_argument(
        "--scales", type=int, nargs="+",
        default=[10_000, 50_000, 100_000],
        help="Donor counts to benchmark (default: 10000 50000 100000)",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="benchmarks/results/v0.3.0/chunk_index_scaling",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    run_benchmark(
        scales=args.scales,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

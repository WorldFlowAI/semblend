"""Multi-turn conversation benchmark: ChunkIndex fast path vs full pipeline.

Measures per-turn latency for 5-10 turn WildChat conversations with the
ChunkIndex fast path ON vs OFF, quantifying the ~3ms embedding skip.

Uses real multi-turn conversations from the WildChat dataset (HuggingFace).

Usage:
    # Dry-run: show what would be measured
    python -m benchmarks.suite.multi_turn_bench --dry-run

    # Run with local semblend pipeline (no engine needed)
    python -m benchmarks.suite.multi_turn_bench \
        --n-conversations 200 \
        --output-dir benchmarks/results/v0.3.0/multi_turn

    # Compare fast-path ON vs OFF
    python -m benchmarks.suite.multi_turn_bench \
        --n-conversations 200 \
        --compare-fast-path \
        --output-dir benchmarks/results/v0.3.0/multi_turn
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# WildChat multi-turn dataset (real user conversations)
WILDCHAT_DATASET = "allenai/WildChat-1M"
MIN_TURNS = 5
MAX_TURNS = 10
MIN_TOKENS_PER_TURN = 128


@dataclass(frozen=True)
class TurnResult:
    """Timing result for a single conversation turn."""
    turn_idx: int
    total_ms: float
    embed_ms: float
    lookup_ms: float
    bathtub_ms: float
    found: bool
    reuse_ratio: float
    chunk_fast_path_used: bool
    num_donors: int


@dataclass
class ConversationResult:
    """Results for an entire multi-turn conversation."""
    conversation_id: str
    num_turns: int
    turn_results: list[TurnResult] = field(default_factory=list)

    @property
    def mean_total_ms(self) -> float:
        if not self.turn_results:
            return 0.0
        return statistics.mean(r.total_ms for r in self.turn_results)

    @property
    def mean_embed_ms(self) -> float:
        if not self.turn_results:
            return 0.0
        return statistics.mean(r.embed_ms for r in self.turn_results)

    @property
    def fast_path_rate(self) -> float:
        if not self.turn_results:
            return 0.0
        fp = sum(1 for r in self.turn_results if r.chunk_fast_path_used)
        return fp / len(self.turn_results)


@dataclass
class BenchmarkResult:
    """Aggregate benchmark results."""
    mode: str  # "fast_path_on" or "fast_path_off"
    n_conversations: int
    n_turns_total: int
    mean_total_ms: float
    median_total_ms: float
    p95_total_ms: float
    mean_embed_ms: float
    fast_path_hit_rate: float
    mean_reuse_ratio: float
    hit_rate: float  # Fraction of turns that found a donor


def _load_wildchat_conversations(
    n_conversations: int,
    min_turns: int = MIN_TURNS,
    max_turns: int = MAX_TURNS,
) -> list[list[str]]:
    """Load multi-turn conversations from WildChat.

    Returns list of conversations, each a list of accumulated prompts
    (turn 1, turn 1+2, turn 1+2+3, ...) simulating the multi-turn pattern.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError(
            "datasets package required: pip install datasets"
        )

    logger.info("Loading WildChat dataset (streaming)...")
    ds = load_dataset(WILDCHAT_DATASET, split="train", streaming=True)

    conversations: list[list[str]] = []
    seen = 0

    for example in ds:
        if len(conversations) >= n_conversations:
            break
        seen += 1

        # WildChat format: list of messages with role/content
        messages = example.get("conversation", [])
        if not messages:
            continue

        # Build accumulated turns (user messages only, concatenated)
        user_turns = [
            m["content"] for m in messages
            if m.get("role") == "user" and len(m.get("content", "")) > 50
        ]

        if len(user_turns) < min_turns:
            continue

        # Truncate to max_turns
        user_turns = user_turns[:max_turns]

        # Build accumulated prompts: turn N = all messages up to turn N
        accumulated = []
        running = ""
        for turn_text in user_turns:
            running = running + "\n\n" + turn_text if running else turn_text
            accumulated.append(running)

        conversations.append(accumulated)

    logger.info(
        "Loaded %d multi-turn conversations (scanned %d examples)",
        len(conversations), seen,
    )

    if len(conversations) < n_conversations:
        logger.warning(
            "Only found %d/%d conversations with >= %d turns",
            len(conversations), n_conversations, min_turns,
        )

    return conversations


def _run_conversation(
    pipeline: object,
    turns: list[str],
    conversation_id: str,
) -> ConversationResult:
    """Run a single multi-turn conversation through the pipeline.

    Each turn is first registered as a donor (simulating completion),
    then the next turn is looked up (simulating the new request).
    """
    result = ConversationResult(
        conversation_id=conversation_id,
        num_turns=len(turns),
    )

    for turn_idx, turn_text in enumerate(turns):
        # Tokenize (approximate: 4 chars per token)
        token_ids = list(range(len(turn_text) // 4))

        if turn_idx > 0:
            # Find donor for this turn (should match prior turn's prefix)
            pipeline_result = pipeline.find_donor(
                token_ids=token_ids,
                prompt_text=turn_text,
            )

            timings = pipeline_result.timings
            result.turn_results.append(TurnResult(
                turn_idx=turn_idx,
                total_ms=timings.total_ms,
                embed_ms=timings.embed_ms,
                lookup_ms=timings.lookup_ms,
                bathtub_ms=timings.bathtub_ms,
                found=pipeline_result.found,
                reuse_ratio=pipeline_result.reuse_ratio,
                chunk_fast_path_used=getattr(
                    pipeline_result, "chunk_fast_path_used", False,
                ),
                num_donors=len(getattr(pipeline_result, "donor_ids", [])),
            ))

        # Register this turn as a donor for future turns
        pipeline.register_donor(
            request_id=f"{conversation_id}_turn{turn_idx}",
            token_ids=token_ids,
            prompt_text=turn_text,
        )

    return result


def _compute_aggregate(
    results: list[ConversationResult],
    mode: str,
) -> BenchmarkResult:
    """Compute aggregate statistics from conversation results."""
    all_totals = []
    all_embeds = []
    all_reuse = []
    n_found = 0
    n_fast_path = 0
    n_total = 0

    for conv in results:
        for tr in conv.turn_results:
            all_totals.append(tr.total_ms)
            all_embeds.append(tr.embed_ms)
            if tr.found:
                all_reuse.append(tr.reuse_ratio)
                n_found += 1
            if tr.chunk_fast_path_used:
                n_fast_path += 1
            n_total += 1

    if not all_totals:
        return BenchmarkResult(
            mode=mode, n_conversations=len(results), n_turns_total=0,
            mean_total_ms=0, median_total_ms=0, p95_total_ms=0,
            mean_embed_ms=0, fast_path_hit_rate=0,
            mean_reuse_ratio=0, hit_rate=0,
        )

    sorted_totals = sorted(all_totals)
    p95_idx = int(len(sorted_totals) * 0.95)

    return BenchmarkResult(
        mode=mode,
        n_conversations=len(results),
        n_turns_total=n_total,
        mean_total_ms=statistics.mean(all_totals),
        median_total_ms=statistics.median(all_totals),
        p95_total_ms=sorted_totals[min(p95_idx, len(sorted_totals) - 1)],
        mean_embed_ms=statistics.mean(all_embeds),
        fast_path_hit_rate=n_fast_path / max(n_total, 1),
        mean_reuse_ratio=statistics.mean(all_reuse) if all_reuse else 0.0,
        hit_rate=n_found / max(n_total, 1),
    )


def run_benchmark(
    n_conversations: int = 200,
    compare_fast_path: bool = False,
    output_dir: str = "benchmarks/results/v0.3.0/multi_turn",
    dry_run: bool = False,
) -> None:
    """Run the multi-turn benchmark.

    Args:
        n_conversations: Number of WildChat conversations to use (n>=200).
        compare_fast_path: Run both ON/OFF and compare.
        output_dir: Where to save result JSONs.
        dry_run: Print what would be done without running.
    """
    if dry_run:
        print(f"Would run multi-turn benchmark with {n_conversations} conversations")
        print(f"  Dataset: {WILDCHAT_DATASET}")
        print(f"  Turns per conversation: {MIN_TURNS}-{MAX_TURNS}")
        print(f"  Compare fast path: {compare_fast_path}")
        print(f"  Output: {output_dir}")
        return

    from benchmarks.suite.metadata import collect_metadata
    from semblend_core.pipeline import SemBlendPipeline

    conversations = _load_wildchat_conversations(n_conversations)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    modes = ["fast_path_on"]
    if compare_fast_path:
        modes.append("fast_path_off")

    all_results: dict[str, BenchmarkResult] = {}

    for mode in modes:
        logger.info("Running multi-turn benchmark: mode=%s", mode)

        # Configure fast path
        if mode == "fast_path_on":
            os.environ["SEMBLEND_CHUNK_FAST_PATH"] = "1"
        else:
            os.environ["SEMBLEND_CHUNK_FAST_PATH"] = "0"

        pipeline = SemBlendPipeline(
            max_donors=10_000,
            chunk_size=256,
        )

        conv_results = []
        for i, turns in enumerate(conversations):
            conv_id = f"conv_{i:04d}"
            conv_result = _run_conversation(pipeline, turns, conv_id)
            conv_results.append(conv_result)

            if (i + 1) % 50 == 0:
                logger.info(
                    "  Progress: %d/%d conversations", i + 1, len(conversations),
                )

        aggregate = _compute_aggregate(conv_results, mode)
        all_results[mode] = aggregate

        # Save per-mode results
        result_data = {
            "_reproducibility": collect_metadata().to_dict(),
            "benchmark": "multi_turn",
            "mode": mode,
            "n_conversations": aggregate.n_conversations,
            "n_turns_total": aggregate.n_turns_total,
            "mean_total_ms": round(aggregate.mean_total_ms, 3),
            "median_total_ms": round(aggregate.median_total_ms, 3),
            "p95_total_ms": round(aggregate.p95_total_ms, 3),
            "mean_embed_ms": round(aggregate.mean_embed_ms, 3),
            "fast_path_hit_rate": round(aggregate.fast_path_hit_rate, 4),
            "mean_reuse_ratio": round(aggregate.mean_reuse_ratio, 4),
            "hit_rate": round(aggregate.hit_rate, 4),
        }
        result_file = out_path / f"multi_turn_{mode}.json"
        result_file.write_text(json.dumps(result_data, indent=2))
        logger.info("Saved results to %s", result_file)

    # Print comparison
    print("\n=== Multi-Turn Benchmark Results ===\n")
    for mode, agg in all_results.items():
        print(f"Mode: {mode}")
        print(f"  Conversations: {agg.n_conversations}")
        print(f"  Total turns:   {agg.n_turns_total}")
        print(f"  Mean total:    {agg.mean_total_ms:.2f} ms")
        print(f"  Median total:  {agg.median_total_ms:.2f} ms")
        print(f"  P95 total:     {agg.p95_total_ms:.2f} ms")
        print(f"  Mean embed:    {agg.mean_embed_ms:.2f} ms")
        print(f"  Fast path:     {agg.fast_path_hit_rate:.1%}")
        print(f"  Mean reuse:    {agg.mean_reuse_ratio:.1%}")
        print(f"  Hit rate:      {agg.hit_rate:.1%}")
        print()

    if compare_fast_path and "fast_path_on" in all_results and "fast_path_off" in all_results:
        on = all_results["fast_path_on"]
        off = all_results["fast_path_off"]
        if off.mean_total_ms > 0:
            speedup = off.mean_total_ms / on.mean_total_ms
            savings = off.mean_embed_ms - on.mean_embed_ms
            print(f"  Speedup: {speedup:.2f}x (fast path ON vs OFF)")
            print(f"  Embed savings: {savings:.2f} ms/turn")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-turn conversation benchmark for ChunkIndex fast path",
    )
    parser.add_argument(
        "--n-conversations", type=int, default=200,
        help="Number of WildChat conversations (default: 200, min: 200)",
    )
    parser.add_argument(
        "--compare-fast-path", action="store_true",
        help="Run both fast-path ON and OFF for comparison",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="benchmarks/results/v0.3.0/multi_turn",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    if args.n_conversations < 200:
        parser.error("--n-conversations must be >= 200 (benchmark standard)")

    run_benchmark(
        n_conversations=args.n_conversations,
        compare_fast_path=args.compare_fast_path,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

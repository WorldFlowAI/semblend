"""Multi-donor RAG benchmark: composite vs single-donor reuse.

Simulates a 3-document RAG scenario where each document is cached as a
separate donor. Compares composite multi-donor reuse ratio against the
best single-donor match.

Uses real documents from XSum and CNN/DailyMail datasets (HuggingFace).

Usage:
    # Dry-run
    python -m benchmarks.suite.multi_donor_rag_bench --dry-run

    # Run benchmark
    python -m benchmarks.suite.multi_donor_rag_bench \
        --n-queries 200 \
        --docs-per-query 3 \
        --output-dir benchmarks/results/v0.3.0/multi_donor_rag

    # Custom dataset
    python -m benchmarks.suite.multi_donor_rag_bench \
        --n-queries 200 \
        --dataset EdinburghNLP/xsum \
        --output-dir benchmarks/results/v0.3.0/multi_donor_rag
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Default datasets for document retrieval simulation
DEFAULT_DATASETS = ("EdinburghNLP/xsum", "abisee/cnn_dailymail")
MIN_DOC_CHARS = 1000  # Minimum document length to ensure 4K+ tokens


@dataclass(frozen=True)
class QueryResult:
    """Result for a single RAG query."""

    query_id: str
    n_docs: int
    # Single-donor (best of N)
    single_reuse_ratio: float
    single_donor_id: str
    single_total_ms: float
    # Multi-donor composite
    multi_reuse_ratio: float
    multi_donor_count: int
    multi_total_ms: float
    # Improvement
    reuse_improvement: float  # multi / single ratio


@dataclass
class RAGBenchmarkResult:
    """Aggregate RAG benchmark results."""

    n_queries: int
    docs_per_query: int
    dataset: str
    mean_single_reuse: float
    mean_multi_reuse: float
    mean_improvement: float
    median_improvement: float
    mean_single_ms: float
    mean_multi_ms: float
    mean_donor_count: float
    multi_better_rate: float  # Fraction where multi > single


def _load_documents(
    dataset_name: str,
    n_documents: int,
    min_chars: int = MIN_DOC_CHARS,
) -> list[str]:
    """Load documents from a HuggingFace dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("datasets package required: pip install datasets")

    logger.info("Loading documents from %s...", dataset_name)

    # Handle dataset-specific format
    if "cnn" in dataset_name.lower():
        config = "3.0.0"
        text_field = "article"
    else:
        config = None
        text_field = "document"

    kwargs = {"split": "train", "streaming": True}
    if config:
        kwargs["name"] = config

    ds = load_dataset(dataset_name, **kwargs)

    documents = []
    for example in ds:
        if len(documents) >= n_documents:
            break

        text = example.get(text_field, "")
        if len(text) >= min_chars:
            documents.append(text)

    logger.info("Loaded %d documents from %s", len(documents), dataset_name)
    return documents


def _build_rag_query(docs: list[str], query_suffix: str) -> str:
    """Build a RAG query by concatenating retrieved documents + query."""
    parts = []
    for i, doc in enumerate(docs):
        parts.append(f"[Document {i + 1}]\n{doc}")
    parts.append(f"\n[Query]\n{query_suffix}")
    return "\n\n".join(parts)


def run_benchmark(
    n_queries: int = 200,
    docs_per_query: int = 3,
    dataset_name: str = "EdinburghNLP/xsum",
    output_dir: str = "benchmarks/results/v0.3.0/multi_donor_rag",
    dry_run: bool = False,
) -> None:
    """Run the multi-donor RAG benchmark.

    Args:
        n_queries: Number of RAG queries to simulate (n>=200).
        docs_per_query: Documents per query (default: 3).
        dataset_name: HuggingFace dataset for documents.
        output_dir: Where to save result JSONs.
        dry_run: Print plan without running.
    """
    if dry_run:
        print("Would run multi-donor RAG benchmark:")
        print(f"  Queries: {n_queries}")
        print(f"  Docs per query: {docs_per_query}")
        print(f"  Dataset: {dataset_name}")
        print(f"  Output: {output_dir}")
        return

    from benchmarks.suite.metadata import collect_metadata
    from semblend_core.pipeline import SemBlendPipeline

    # Need enough documents for all queries × docs_per_query with some overlap
    n_docs_needed = n_queries * 2  # Pool of docs (queries draw overlapping subsets)
    documents = _load_documents(dataset_name, n_docs_needed)

    if len(documents) < docs_per_query:
        raise RuntimeError(f"Need at least {docs_per_query} documents, got {len(documents)}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    query_results: list[QueryResult] = []

    # --- Single-donor baseline ---
    logger.info("Phase 1: Single-donor baseline (SEMBLEND_MULTI_DONOR=0)")
    os.environ["SEMBLEND_MULTI_DONOR"] = "0"
    os.environ["SEMBLEND_CHUNK_FAST_PATH"] = "1"

    pipeline_single = SemBlendPipeline(
        max_donors=10_000,
        chunk_size=256,
    )

    # Register all documents as individual donors
    for i, doc in enumerate(documents):
        token_ids = list(range(len(doc) // 4))
        pipeline_single.register_donor(
            request_id=f"doc_{i:04d}",
            token_ids=token_ids,
            prompt_text=doc,
        )

    # Run queries (single-donor)
    single_results: list[tuple[float, str, float]] = []
    for q_idx in range(n_queries):
        # Pick docs_per_query documents (overlapping window)
        doc_start = q_idx % max(len(documents) - docs_per_query, 1)
        query_docs = documents[doc_start : doc_start + docs_per_query]
        query_text = _build_rag_query(query_docs, f"Summarize the above documents. Query {q_idx}")
        token_ids = list(range(len(query_text) // 4))

        result = pipeline_single.find_donor(
            token_ids=token_ids,
            prompt_text=query_text,
        )
        single_results.append(
            (
                result.reuse_ratio if result.found else 0.0,
                result.donor_id or "",
                result.timings.total_ms,
            )
        )

        if (q_idx + 1) % 50 == 0:
            logger.info("  Single-donor progress: %d/%d", q_idx + 1, n_queries)

    # --- Multi-donor composite ---
    logger.info("Phase 2: Multi-donor composite (SEMBLEND_MULTI_DONOR=1)")
    os.environ["SEMBLEND_MULTI_DONOR"] = "1"

    pipeline_multi = SemBlendPipeline(
        max_donors=10_000,
        chunk_size=256,
    )

    # Register same documents
    for i, doc in enumerate(documents):
        token_ids = list(range(len(doc) // 4))
        pipeline_multi.register_donor(
            request_id=f"doc_{i:04d}",
            token_ids=token_ids,
            prompt_text=doc,
        )

    # Run queries (multi-donor)
    for q_idx in range(n_queries):
        doc_start = q_idx % max(len(documents) - docs_per_query, 1)
        query_docs = documents[doc_start : doc_start + docs_per_query]
        query_text = _build_rag_query(query_docs, f"Summarize the above documents. Query {q_idx}")
        token_ids = list(range(len(query_text) // 4))

        result = pipeline_multi.find_donor(
            token_ids=token_ids,
            prompt_text=query_text,
        )

        s_reuse, s_donor, s_ms = single_results[q_idx]
        m_reuse = result.reuse_ratio if result.found else 0.0
        m_donors = len(getattr(result, "donor_ids", [])) if result.found else 0
        m_ms = result.timings.total_ms

        improvement = m_reuse / max(s_reuse, 0.001) if s_reuse > 0 else 1.0

        query_results.append(
            QueryResult(
                query_id=f"q_{q_idx:04d}",
                n_docs=docs_per_query,
                single_reuse_ratio=s_reuse,
                single_donor_id=s_donor,
                single_total_ms=s_ms,
                multi_reuse_ratio=m_reuse,
                multi_donor_count=m_donors,
                multi_total_ms=m_ms,
                reuse_improvement=improvement,
            )
        )

        if (q_idx + 1) % 50 == 0:
            logger.info("  Multi-donor progress: %d/%d", q_idx + 1, n_queries)

    # Compute aggregates
    single_reuses = [r.single_reuse_ratio for r in query_results]
    multi_reuses = [r.multi_reuse_ratio for r in query_results]
    improvements = [r.reuse_improvement for r in query_results]

    aggregate = RAGBenchmarkResult(
        n_queries=n_queries,
        docs_per_query=docs_per_query,
        dataset=dataset_name,
        mean_single_reuse=statistics.mean(single_reuses),
        mean_multi_reuse=statistics.mean(multi_reuses),
        mean_improvement=statistics.mean(improvements),
        median_improvement=statistics.median(improvements),
        mean_single_ms=statistics.mean(r.single_total_ms for r in query_results),
        mean_multi_ms=statistics.mean(r.multi_total_ms for r in query_results),
        mean_donor_count=statistics.mean(r.multi_donor_count for r in query_results),
        multi_better_rate=sum(
            1 for r in query_results if r.multi_reuse_ratio > r.single_reuse_ratio
        )
        / max(n_queries, 1),
    )

    # Save results
    result_data = {
        "_reproducibility": collect_metadata().to_dict(),
        "benchmark": "multi_donor_rag",
        "dataset": dataset_name,
        "n_queries": aggregate.n_queries,
        "docs_per_query": aggregate.docs_per_query,
        "mean_single_reuse": round(aggregate.mean_single_reuse, 4),
        "mean_multi_reuse": round(aggregate.mean_multi_reuse, 4),
        "mean_improvement": round(aggregate.mean_improvement, 4),
        "median_improvement": round(aggregate.median_improvement, 4),
        "mean_single_ms": round(aggregate.mean_single_ms, 3),
        "mean_multi_ms": round(aggregate.mean_multi_ms, 3),
        "mean_donor_count": round(aggregate.mean_donor_count, 2),
        "multi_better_rate": round(aggregate.multi_better_rate, 4),
    }
    result_file = out_path / "multi_donor_rag.json"
    result_file.write_text(json.dumps(result_data, indent=2))
    logger.info("Saved results to %s", result_file)

    # Print results
    print("\n=== Multi-Donor RAG Benchmark ===\n")
    print(f"Dataset: {dataset_name}")
    print(f"Queries: {aggregate.n_queries}, Docs/query: {aggregate.docs_per_query}")
    print(f"\nSingle-donor reuse:  {aggregate.mean_single_reuse:.1%}")
    print(f"Multi-donor reuse:   {aggregate.mean_multi_reuse:.1%}")
    print(f"Mean improvement:    {aggregate.mean_improvement:.2f}x")
    print(f"Median improvement:  {aggregate.median_improvement:.2f}x")
    print(f"Multi better rate:   {aggregate.multi_better_rate:.1%}")
    print(f"Mean donor count:    {aggregate.mean_donor_count:.1f}")
    print(f"\nSingle latency:      {aggregate.mean_single_ms:.2f} ms")
    print(f"Multi latency:       {aggregate.mean_multi_ms:.2f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-donor RAG benchmark: composite vs single-donor reuse",
    )
    parser.add_argument(
        "--n-queries",
        type=int,
        default=200,
        help="Number of RAG queries (default: 200, min: 200)",
    )
    parser.add_argument(
        "--docs-per-query",
        type=int,
        default=3,
        help="Documents per query (default: 3)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="EdinburghNLP/xsum",
        help="HuggingFace dataset for documents",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/results/v0.3.0/multi_donor_rag",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    if args.n_queries < 200:
        parser.error("--n-queries must be >= 200 (benchmark standard)")

    run_benchmark(
        n_queries=args.n_queries,
        docs_per_query=args.docs_per_query,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

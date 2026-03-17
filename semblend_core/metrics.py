"""Prometheus metrics for SemBlend production monitoring.

Optional dependency: works silently if prometheus_client is not installed.
Metrics are registered lazily on first use.

Usage:
    from semblend_core.metrics import METRICS
    METRICS.record_pipeline_result(hit=True, similarity=0.85, ttft_ms=120.0)
    METRICS.record_embedding_latency(3.2)
    METRICS.record_donor_store_size(150)
"""
from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger(__name__)

# Disable metrics with SEMBLEND_METRICS=0
_METRICS_ENABLED = os.environ.get("SEMBLEND_METRICS", "1") != "0"


class _NoopMetrics:
    """No-op metrics when prometheus_client is unavailable."""

    def record_pipeline_result(
        self,
        hit: bool,
        similarity: float = 0.0,
        reuse_ratio: float = 0.0,
        ttft_ms: float = 0.0,
        fuzzy_chunks: int = 0,
    ) -> None:
        pass

    def record_embedding_latency(self, ms: float) -> None:
        pass

    def record_alignment_latency(self, ms: float) -> None:
        pass

    def record_cosine_search_latency(self, ms: float) -> None:
        pass

    def record_donor_store_size(self, n: int) -> None:
        pass

    def record_pipeline_error(self, stage: str) -> None:
        pass

    def record_rope_correction(self, n_positions: int, layer: int) -> None:
        pass


class _PrometheusMetrics:
    """Real Prometheus metrics using prometheus_client."""

    def __init__(self) -> None:
        from prometheus_client import Counter, Gauge, Histogram

        prefix = "semblend_"

        self._pipeline_total = Counter(
            f"{prefix}pipeline_total",
            "Total pipeline invocations",
            ["result"],  # hit, miss, error
        )
        self._hit_similarity = Histogram(
            f"{prefix}hit_similarity",
            "Cosine similarity of accepted donors",
            buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
        )
        self._reuse_ratio = Histogram(
            f"{prefix}reuse_ratio",
            "Token reuse ratio on hits",
            buckets=[0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0],
        )
        self._ttft_ms = Histogram(
            f"{prefix}ttft_milliseconds",
            "TTFT in milliseconds",
            ["type"],  # hit, cold
            buckets=[50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000],
        )
        self._embedding_latency = Histogram(
            f"{prefix}embedding_latency_ms",
            "MiniLM embedding latency in ms",
            buckets=[1, 2, 3, 5, 8, 10, 20, 50],
        )
        self._alignment_latency = Histogram(
            f"{prefix}alignment_latency_ms",
            "Token alignment latency in ms",
            buckets=[0.1, 0.5, 1, 2, 5, 10, 20],
        )
        self._cosine_search_latency = Histogram(
            f"{prefix}cosine_search_latency_ms",
            "Donor store cosine search latency in ms",
            buckets=[0.05, 0.1, 0.2, 0.5, 1, 2, 5],
        )
        self._donor_store_size = Gauge(
            f"{prefix}donor_store_size",
            "Number of entries in the donor store",
        )
        self._pipeline_errors = Counter(
            f"{prefix}pipeline_errors_total",
            "Pipeline errors by stage",
            ["stage"],  # embedding, search, alignment, injection
        )
        self._fuzzy_chunks = Counter(
            f"{prefix}fuzzy_chunks_total",
            "Total fuzzy-matched chunks (non-exact)",
        )
        self._rope_corrections = Counter(
            f"{prefix}rope_corrections_total",
            "Total RoPE corrections applied",
        )

    def record_pipeline_result(
        self,
        hit: bool,
        similarity: float = 0.0,
        reuse_ratio: float = 0.0,
        ttft_ms: float = 0.0,
        fuzzy_chunks: int = 0,
    ) -> None:
        self._pipeline_total.labels(result="hit" if hit else "miss").inc()
        if hit:
            self._hit_similarity.observe(similarity)
            self._reuse_ratio.observe(reuse_ratio)
            if ttft_ms > 0:
                self._ttft_ms.labels(type="hit").observe(ttft_ms)
            if fuzzy_chunks > 0:
                self._fuzzy_chunks.inc(fuzzy_chunks)
        else:
            if ttft_ms > 0:
                self._ttft_ms.labels(type="cold").observe(ttft_ms)

    def record_embedding_latency(self, ms: float) -> None:
        self._embedding_latency.observe(ms)

    def record_alignment_latency(self, ms: float) -> None:
        self._alignment_latency.observe(ms)

    def record_cosine_search_latency(self, ms: float) -> None:
        self._cosine_search_latency.observe(ms)

    def record_donor_store_size(self, n: int) -> None:
        self._donor_store_size.set(n)

    def record_pipeline_error(self, stage: str) -> None:
        self._pipeline_errors.labels(stage=stage).inc()

    def record_rope_correction(self, n_positions: int, layer: int) -> None:
        self._rope_corrections.inc()


def _create_metrics() -> _NoopMetrics | _PrometheusMetrics:
    """Create metrics instance (real or no-op)."""
    if not _METRICS_ENABLED:
        return _NoopMetrics()

    try:
        import prometheus_client  # noqa: F401
        metrics = _PrometheusMetrics()
        logger.info("SemBlend Prometheus metrics initialized")
        return metrics
    except ImportError:
        logger.debug(
            "prometheus_client not installed, metrics disabled"
        )
        return _NoopMetrics()


METRICS = _create_metrics()

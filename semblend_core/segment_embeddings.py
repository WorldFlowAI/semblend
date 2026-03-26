"""Per-chunk segment embeddings for SemBlend fuzzy matching.

Stores one embedding per KV-aligned chunk for fine-grained donor-target
comparison. Used by the PQ segment store for memory-efficient storage
and the confidence-gated fuzzy matcher for semantic verification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SegmentEmbeddings:
    """Per-chunk embedding vectors aligned to KV block boundaries.

    Each embedding corresponds to exactly one KV chunk (chunk_size tokens).
    The number of segments varies by input length:
    n_segments = ceil(n_tokens / chunk_size).
    """

    matrix: np.ndarray  # [n_segments, dim], L2-normalized float32
    chunk_token_ranges: tuple[tuple[int, int], ...]  # (start, end) per segment
    chunk_size: int  # KV block size used for segmentation

    @property
    def n_segments(self) -> int:
        return self.matrix.shape[0]

    @property
    def dim(self) -> int:
        return self.matrix.shape[1]

    @property
    def nbytes(self) -> int:
        return self.matrix.nbytes


@dataclass(frozen=True)
class EmbedResult:
    """Embedding result with optional per-segment detail.

    Short inputs (single chunk) have segments=None since the pooled
    embedding is sufficient. Multi-chunk inputs include per-segment
    embeddings for fine-grained fuzzy match verification.
    """

    pooled: np.ndarray  # [dim] L2-normalized
    segments: SegmentEmbeddings | None = None

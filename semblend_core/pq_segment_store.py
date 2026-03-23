"""Product-quantized segment embedding store for SemBlend.

Stores per-chunk segment embeddings using product quantization (PQ) for
32x memory compression. Supports asymmetric distance computation (ADC)
for quality-preserving segment comparison.

Memory at 100K donors (~30 segments each):
  - Naive float32: 4.4 GB
  - PQ (M=48, K=256): 137 MB (97% reduction)

Training: Codebook trained lazily from first N donors (default 500).
Before training, falls back to buffered float32 with exact cosine.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# PQ defaults for 384-dim embeddings
DEFAULT_M = 48  # sub-quantizers (384 / 48 = 8 dims each)
DEFAULT_K = 256  # centroids per sub-quantizer
DEFAULT_D_SUB = 8  # dimensions per sub-vector


@dataclass
class PQCodebook:
    """Product quantization codebook with M sub-quantizers, K centroids each."""
    centroids: np.ndarray  # [M, K, d_sub] float32
    n_subquantizers: int = DEFAULT_M
    n_centroids: int = DEFAULT_K
    d_sub: int = DEFAULT_D_SUB
    trained: bool = False

    @property
    def nbytes(self) -> int:
        return self.centroids.nbytes if self.trained else 0


def train_pq_codebook(
    embeddings: np.ndarray,
    n_subquantizers: int = DEFAULT_M,
    n_centroids: int = DEFAULT_K,
    n_iter: int = 25,
    seed: int = 42,
) -> PQCodebook:
    """Train a PQ codebook from sample embeddings.

    Args:
        embeddings: [N, dim] float32 sample embeddings.
        n_subquantizers: M — number of sub-vector groups.
        n_centroids: K — centroids per sub-quantizer.
        n_iter: K-means iterations per sub-quantizer.
        seed: Random seed for reproducibility.

    Returns:
        Trained PQCodebook.
    """
    n_samples, dim = embeddings.shape
    d_sub = dim // n_subquantizers
    if dim % n_subquantizers != 0:
        raise ValueError(
            f"dim={dim} not divisible by n_subquantizers={n_subquantizers}"
        )

    rng = np.random.RandomState(seed)
    centroids = np.zeros((n_subquantizers, n_centroids, d_sub), dtype=np.float32)

    for m in range(n_subquantizers):
        sub_vectors = embeddings[:, m * d_sub:(m + 1) * d_sub].copy()

        # K-means for this sub-quantizer
        # Initialize centroids with k-means++
        indices = [rng.randint(n_samples)]
        for _ in range(1, min(n_centroids, n_samples)):
            dists = np.min(
                np.stack([
                    np.sum((sub_vectors - sub_vectors[idx]) ** 2, axis=1)
                    for idx in indices
                ], axis=0),
                axis=0,
            )
            probs = dists / (dists.sum() + 1e-12)
            indices.append(rng.choice(n_samples, p=probs))

        centers = sub_vectors[indices[:n_centroids]].copy()
        if len(centers) < n_centroids:
            # Pad with random samples if fewer unique points
            extra = sub_vectors[rng.choice(n_samples, n_centroids - len(centers))]
            centers = np.vstack([centers, extra])

        for _ in range(n_iter):
            # Assign
            diffs = sub_vectors[:, None, :] - centers[None, :, :]  # [N, K, d]
            sq_dists = np.sum(diffs ** 2, axis=2)  # [N, K]
            assignments = np.argmin(sq_dists, axis=1)  # [N]

            # Update
            new_centers = np.zeros_like(centers)
            counts = np.zeros(n_centroids)
            for k in range(n_centroids):
                mask = assignments == k
                if mask.any():
                    new_centers[k] = sub_vectors[mask].mean(axis=0)
                    counts[k] = mask.sum()
                else:
                    new_centers[k] = centers[k]

            if np.allclose(centers, new_centers, atol=1e-6):
                break
            centers = new_centers

        centroids[m] = centers

    logger.info(
        "PQ codebook trained: M=%d, K=%d, d_sub=%d from %d samples",
        n_subquantizers, n_centroids, d_sub, n_samples,
    )
    return PQCodebook(
        centroids=centroids,
        n_subquantizers=n_subquantizers,
        n_centroids=n_centroids,
        d_sub=d_sub,
        trained=True,
    )


def pq_encode(embedding: np.ndarray, codebook: PQCodebook) -> np.ndarray:
    """Encode a single embedding into PQ codes.

    Args:
        embedding: [dim] float32 vector.
        codebook: Trained PQCodebook.

    Returns:
        [M] uint8 array of centroid indices.
    """
    codes = np.zeros(codebook.n_subquantizers, dtype=np.uint8)
    for m in range(codebook.n_subquantizers):
        sub = embedding[m * codebook.d_sub:(m + 1) * codebook.d_sub]
        dists = np.sum((codebook.centroids[m] - sub) ** 2, axis=1)
        codes[m] = np.argmin(dists)
    return codes


def pq_encode_batch(embeddings: np.ndarray, codebook: PQCodebook) -> np.ndarray:
    """Encode a batch of embeddings into PQ codes.

    Args:
        embeddings: [N, dim] float32.
        codebook: Trained PQCodebook.

    Returns:
        [N, M] uint8 array.
    """
    n = embeddings.shape[0]
    codes = np.zeros((n, codebook.n_subquantizers), dtype=np.uint8)
    for m in range(codebook.n_subquantizers):
        subs = embeddings[:, m * codebook.d_sub:(m + 1) * codebook.d_sub]
        # [N, 1, d] - [1, K, d] -> [N, K, d] -> [N, K]
        dists = np.sum(
            (subs[:, None, :] - codebook.centroids[m][None, :, :]) ** 2,
            axis=2,
        )
        codes[:, m] = np.argmin(dists, axis=1).astype(np.uint8)
    return codes


def adc_distance_table(query: np.ndarray, codebook: PQCodebook) -> np.ndarray:
    """Compute asymmetric distance table for a query vector.

    Pre-computes squared distances from each query sub-vector to all
    centroids. Used for fast distance computation against PQ codes.

    Args:
        query: [dim] float32 query vector.
        codebook: Trained PQCodebook.

    Returns:
        [M, K] float32 distance table.
    """
    table = np.zeros((codebook.n_subquantizers, codebook.n_centroids), dtype=np.float32)
    for m in range(codebook.n_subquantizers):
        sub = query[m * codebook.d_sub:(m + 1) * codebook.d_sub]
        table[m] = np.sum((codebook.centroids[m] - sub) ** 2, axis=1)
    return table


def adc_distances(table: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """Compute distances from pre-computed ADC table and PQ codes.

    Args:
        table: [M, K] distance table from adc_distance_table().
        codes: [N, M] uint8 PQ codes.

    Returns:
        [N] float32 squared distances.
    """
    n, m = codes.shape
    distances = np.zeros(n, dtype=np.float32)
    for sub_idx in range(m):
        distances += table[sub_idx, codes[:, sub_idx]]
    return distances


def adc_cosine_similarities(
    query: np.ndarray,
    codes: np.ndarray,
    codebook: PQCodebook,
) -> np.ndarray:
    """Approximate cosine similarities using ADC.

    Converts PQ squared L2 distances to approximate cosine similarities.
    Assumes both query and stored embeddings were L2-normalized before encoding.

    Args:
        query: [dim] float32 L2-normalized query.
        codes: [N, M] uint8 PQ codes.
        codebook: Trained PQCodebook.

    Returns:
        [N] float32 approximate cosine similarities.
    """
    table = adc_distance_table(query, codebook)
    sq_dists = adc_distances(table, codes)
    # For L2-normalized vectors: cosine_sim = 1 - sq_dist / 2
    return 1.0 - sq_dists / 2.0


class PQSegmentStore:
    """Product-quantized segment embedding store.

    Stores per-chunk segment embeddings as PQ codes for memory-efficient
    comparison. Before the codebook is trained (first N donors), buffers
    embeddings in float32 for exact comparison and codebook training.

    Thread-safe: reads are lock-free, writes use a lock.

    Args:
        max_entries: Maximum number of donors to store segments for.
        max_segments_per_entry: Maximum segments per donor.
        train_threshold: Number of donors before training PQ codebook.
        n_subquantizers: M for PQ.
        n_centroids: K for PQ.
    """

    def __init__(
        self,
        max_entries: int = 10_000,
        max_segments_per_entry: int = 64,
        train_threshold: int = 500,
        n_subquantizers: int = DEFAULT_M,
        n_centroids: int = DEFAULT_K,
    ) -> None:
        self._max_entries = max_entries
        self._max_segments = max_segments_per_entry
        self._train_threshold = train_threshold
        self._n_subquantizers = n_subquantizers
        self._n_centroids = n_centroids

        self._codebook: PQCodebook | None = None
        self._lock = threading.Lock()

        # PQ-encoded storage: [max_entries * max_segments, M] uint8
        self._codes = np.zeros(
            (max_entries * max_segments_per_entry, n_subquantizers),
            dtype=np.uint8,
        )
        self._segment_counts: dict[str, int] = {}
        self._donor_offsets: dict[str, int] = {}
        self._next_offset = 0

        # Pre-training float32 buffer
        self._buffer: dict[str, np.ndarray] = {}  # donor_id -> [n_seg, dim]
        self._buffer_total_segments = 0

    @property
    def codebook_trained(self) -> bool:
        return self._codebook is not None and self._codebook.trained

    @property
    def size(self) -> int:
        return len(self._segment_counts) + len(self._buffer)

    @property
    def nbytes(self) -> int:
        if self.codebook_trained:
            total_segs = sum(self._segment_counts.values())
            return total_segs * self._n_subquantizers + (self._codebook.nbytes if self._codebook else 0)
        else:
            return sum(arr.nbytes for arr in self._buffer.values())

    def add_segments(self, donor_id: str, segments: np.ndarray) -> None:
        """Store segment embeddings for a donor.

        Args:
            donor_id: Donor request ID.
            segments: [n_segments, dim] L2-normalized float32 embeddings.
        """
        if donor_id in self._segment_counts or donor_id in self._buffer:
            return  # Already stored

        n_seg = min(segments.shape[0], self._max_segments)
        segments = segments[:n_seg]

        with self._lock:
            if self.codebook_trained:
                self._store_pq(donor_id, segments)
            else:
                self._buffer[donor_id] = segments.copy()
                self._buffer_total_segments += n_seg

                # Check if we should train
                if len(self._buffer) >= self._train_threshold:
                    self._train_and_encode()

    def _store_pq(self, donor_id: str, segments: np.ndarray) -> None:
        """Encode and store segments as PQ codes. Caller holds lock."""
        n_seg = segments.shape[0]

        # Evict if at capacity
        while len(self._segment_counts) >= self._max_entries:
            self._evict_oldest()

        offset = self._next_offset
        self._next_offset += n_seg

        codes = pq_encode_batch(segments, self._codebook)
        end = offset + n_seg
        if end <= self._codes.shape[0]:
            self._codes[offset:end] = codes
        else:
            # Wrap around: resize
            new_size = max(self._codes.shape[0] * 2, end)
            new_codes = np.zeros(
                (new_size, self._n_subquantizers), dtype=np.uint8,
            )
            new_codes[:self._codes.shape[0]] = self._codes
            self._codes = new_codes
            self._codes[offset:end] = codes

        self._donor_offsets[donor_id] = offset
        self._segment_counts[donor_id] = n_seg

    def _train_and_encode(self) -> None:
        """Train codebook from buffer and re-encode all buffered segments. Caller holds lock."""
        all_segments = np.vstack(list(self._buffer.values()))
        logger.info(
            "Training PQ codebook from %d segments across %d donors",
            all_segments.shape[0], len(self._buffer),
        )

        self._codebook = train_pq_codebook(
            all_segments,
            n_subquantizers=self._n_subquantizers,
            n_centroids=self._n_centroids,
        )

        # Re-encode all buffered donors
        for did, segs in self._buffer.items():
            self._store_pq(did, segs)

        self._buffer.clear()
        self._buffer_total_segments = 0

    def _evict_oldest(self) -> None:
        """Evict the donor with the smallest offset. Caller holds lock."""
        if not self._donor_offsets:
            return
        oldest_id = min(self._donor_offsets, key=self._donor_offsets.get)
        del self._donor_offsets[oldest_id]
        del self._segment_counts[oldest_id]

    def extend_segments(self, donor_id: str, new_segments: np.ndarray) -> None:
        """Append new segments to an existing donor entry.

        Used when a donor is extended with new chunks (e.g., multi-turn
        conversation where Turn N+1 shares prefix with Turn N but adds
        new content). Appends without re-encoding existing segments.

        Args:
            donor_id: Existing donor ID.
            new_segments: [n_new, dim] L2-normalized float32 embeddings.
        """
        n_new = min(new_segments.shape[0], self._max_segments)
        new_segments = new_segments[:n_new]

        with self._lock:
            if self.codebook_trained:
                existing_count = self._segment_counts.get(donor_id, 0)
                total = existing_count + n_new
                if total > self._max_segments:
                    n_new = self._max_segments - existing_count
                    if n_new <= 0:
                        return
                    new_segments = new_segments[:n_new]

                new_codes = pq_encode_batch(new_segments, self._codebook)

                if donor_id in self._donor_offsets:
                    offset = self._donor_offsets[donor_id]
                    append_start = offset + existing_count
                    append_end = append_start + n_new

                    if append_end <= self._codes.shape[0]:
                        self._codes[append_start:append_end] = new_codes
                    else:
                        new_size = max(self._codes.shape[0] * 2, append_end)
                        resized = np.zeros(
                            (new_size, self._n_subquantizers), dtype=np.uint8,
                        )
                        resized[:self._codes.shape[0]] = self._codes
                        self._codes = resized
                        self._codes[append_start:append_end] = new_codes

                    self._segment_counts[donor_id] = existing_count + n_new
                else:
                    self._store_pq(donor_id, new_segments)
            else:
                if donor_id in self._buffer:
                    existing = self._buffer[donor_id]
                    total = existing.shape[0] + n_new
                    if total > self._max_segments:
                        n_new = self._max_segments - existing.shape[0]
                        if n_new <= 0:
                            return
                        new_segments = new_segments[:n_new]
                    self._buffer[donor_id] = np.vstack([existing, new_segments])
                    self._buffer_total_segments += n_new
                else:
                    self._buffer[donor_id] = new_segments.copy()
                    self._buffer_total_segments += n_new

                if len(self._buffer) >= self._train_threshold:
                    self._train_and_encode()

    def evict(self, donor_id: str) -> None:
        """Remove a donor's segments from the store."""
        with self._lock:
            self._donor_offsets.pop(donor_id, None)
            self._segment_counts.pop(donor_id, None)
            self._buffer.pop(donor_id, None)

    def compare_segments(
        self,
        query_segments: np.ndarray,
        donor_ids: list[str],
    ) -> list[float]:
        """Compare query segment embeddings against stored donors.

        For each donor, computes the mean best-match similarity across
        query segments using ADC (or exact cosine if pre-training).

        Args:
            query_segments: [Q, dim] float32 L2-normalized query segments.
            donor_ids: List of donor IDs to compare against.

        Returns:
            List of similarity scores [0, 1] per donor, same order as donor_ids.
        """
        scores = []
        for did in donor_ids:
            score = self._compare_one(query_segments, did)
            scores.append(score)
        return scores

    def _compare_one(self, query_segments: np.ndarray, donor_id: str) -> float:
        """Compare query segments against one donor."""
        # Check PQ store
        if donor_id in self._segment_counts and self._codebook is not None:
            offset = self._donor_offsets[donor_id]
            n_seg = self._segment_counts[donor_id]
            donor_codes = self._codes[offset:offset + n_seg]
            return self._adc_segment_score(query_segments, donor_codes)

        # Check buffer (pre-training)
        if donor_id in self._buffer:
            donor_segs = self._buffer[donor_id]
            return self._exact_segment_score(query_segments, donor_segs)

        return 0.0

    def _adc_segment_score(
        self, query_segments: np.ndarray, donor_codes: np.ndarray,
    ) -> float:
        """Segment similarity using ADC. Returns mean best-match cosine."""
        n_query = query_segments.shape[0]
        total_sim = 0.0

        for i in range(n_query):
            sims = adc_cosine_similarities(
                query_segments[i], donor_codes, self._codebook,
            )
            total_sim += float(np.max(sims))

        return total_sim / max(n_query, 1)

    def _exact_segment_score(
        self, query_segments: np.ndarray, donor_segments: np.ndarray,
    ) -> float:
        """Segment similarity using exact cosine. Returns mean best-match."""
        # [Q, D] @ [D, S] -> [Q, S]
        sim_matrix = query_segments @ donor_segments.T
        # Mean of per-query-segment best match
        return float(np.mean(np.max(sim_matrix, axis=1)))

    def get_donor_codes(self, donor_id: str) -> np.ndarray | None:
        """Get raw PQ codes for a donor (for serialization/transfer)."""
        if donor_id in self._segment_counts:
            offset = self._donor_offsets[donor_id]
            n_seg = self._segment_counts[donor_id]
            return self._codes[offset:offset + n_seg].copy()
        return None

    def get_segment_similarity(
        self,
        query_segment: np.ndarray,
        donor_id: str,
        chunk_idx: int,
    ) -> float:
        """Get similarity for a specific chunk between query and donor.

        Used by the fuzzy matcher to verify individual chunk matches.

        Args:
            query_segment: [dim] float32 L2-normalized.
            donor_id: Donor ID.
            chunk_idx: Index of the donor chunk to compare.

        Returns:
            Cosine similarity, or 0.0 if not available.
        """
        if donor_id in self._segment_counts and self._codebook is not None:
            offset = self._donor_offsets[donor_id]
            n_seg = self._segment_counts[donor_id]
            if chunk_idx >= n_seg:
                return 0.0
            code = self._codes[offset + chunk_idx:offset + chunk_idx + 1]
            sims = adc_cosine_similarities(query_segment, code, self._codebook)
            return float(sims[0])

        if donor_id in self._buffer:
            segs = self._buffer[donor_id]
            if chunk_idx >= segs.shape[0]:
                return 0.0
            return float(np.dot(query_segment, segs[chunk_idx]))

        return 0.0

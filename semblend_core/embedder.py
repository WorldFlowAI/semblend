"""In-process MiniLM CPU embedding for SemBlend donor matching.

Loads all-MiniLM-L6-v2 (80MB, 384-dim) on CPU at init. Produces embeddings
in ~5ms with zero network hops, replacing the 150ms jina-v4 HTTP call.

Resolution chain:
  1. SimHash pre-filter -> reject obvious non-matches (~0.1ms)
  2. MiniLM CPU embedding -> 384-dim vector (~5ms)
  3. Fallback: Jaccard similarity on token IDs (0ms, less accurate)
"""
from __future__ import annotations

import logging
import os
import time
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class EmbedderType(Enum):
    MINILM = "minilm"
    JINA = "jina"
    JACCARD = "jaccard"


class MiniLMEmbedder:
    """In-process embedder using all-MiniLM-L6-v2.

    Auto-detects GPU — MiniLM is tiny (22M params, ~88MB) so it coexists
    with inference models on the same GPU easily. Falls back to CPU.

    Thread-safe: model is read-only after initialization.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"
    DIMENSION = 384

    def __init__(self) -> None:
        self._model = None
        self._available = False
        self._load_time_ms = 0.0
        self._init()

    def _init(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            t0 = time.monotonic()
            try:
                import torch
                if torch.cuda.is_available():
                    # Use last available GPU to avoid contending with inference model
                    gpu_count = torch.cuda.device_count()
                    device = f"cuda:{gpu_count - 1}" if gpu_count > 1 else "cuda:0"
                else:
                    device = "cpu"
            except ImportError:
                device = "cpu"
            self._model = SentenceTransformer(self.MODEL_NAME, device=device)
            self._load_time_ms = (time.monotonic() - t0) * 1000
            self._available = True
            logger.info(
                "MiniLM embedder loaded: %s (%.0fms)",
                self.MODEL_NAME, self._load_time_ms,
            )
        except ImportError:
            logger.warning("sentence-transformers not installed - MiniLM disabled")
        except Exception:
            logger.warning("MiniLM model load failed", exc_info=True)

    @property
    def available(self) -> bool:
        return self._available

    @property
    def dimension(self) -> int:
        return self.DIMENSION

    # Segmented embedding parameters
    MAX_TOKENS = 512       # MiniLM native context window
    OVERLAP_TOKENS = 64    # overlap between adjacent segments
    MIN_SEGMENT_TOKENS = 32  # skip trailing segments shorter than this

    def embed(self, text: str) -> np.ndarray | None:
        """Embed text with full-document coverage via segmented mean pooling.

        For short texts (≤512 tokens): single-pass embedding (~5ms).
        For long texts: segments into overlapping windows, batch-embeds
        all segments, and mean-pools into a single vector.

        Returns:
            Normalized embedding vector, or None if unavailable.
        """
        if not self._available or not text.strip():
            return None

        t0 = time.monotonic()

        # Estimate token count (~4 chars/token) to decide segmentation
        est_tokens = len(text) // 4
        if est_tokens <= self.MAX_TOKENS:
            # Short text: single pass
            embedding = self._model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        else:
            # Long text: segment and batch embed
            embedding = self._embed_segmented(text)

        elapsed_ms = (time.monotonic() - t0) * 1000
        if elapsed_ms > 10:
            logger.debug("MiniLM embed took %.1fms (text=%d chars)", elapsed_ms, len(text))

        return embedding

    def _embed_segmented(self, text: str) -> np.ndarray:
        """Segment long text and batch-embed with mean pooling."""
        tokenizer = self._model.tokenizer
        full_ids = tokenizer.encode(text, add_special_tokens=False)

        usable = self.MAX_TOKENS - 2  # reserve [CLS] and [SEP]
        stride = usable - self.OVERLAP_TOKENS

        segments = []
        for start in range(0, len(full_ids), stride):
            chunk_ids = full_ids[start:start + usable]
            if len(chunk_ids) < self.MIN_SEGMENT_TOKENS:
                break
            chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
            segments.append(chunk_text)

        if not segments:
            return self._model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

        # Batch encode all segments at once
        embeddings = self._model.encode(
            segments,
            batch_size=len(segments),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )  # [N, 384]

        # Mean pool across segments
        pooled = embeddings.mean(axis=0)

        # L2 normalize the pooled vector
        norm = np.linalg.norm(pooled)
        return pooled / max(norm, 1e-12)

    def embed_with_segments(
        self, text: str, chunk_size: int = 256,
    ) -> "EmbedResult | None":
        """Embed text with per-segment embeddings aligned to KV block boundaries.

        Tokenizes the full text, splits into non-overlapping chunk_size-token
        windows (matching the KV cache block size), and batch-embeds all chunks
        in a single forward pass.

        Args:
            text: Input text to embed.
            chunk_size: Tokens per segment, aligned to KV block boundaries.
                Defaults to 256 (LMCache block size). Use 128 for TRT-LLM.

        Returns:
            EmbedResult with pooled embedding and optional per-segment detail,
            or None if the embedder is unavailable.
        """
        if not self._available or not text.strip():
            return None

        # Lazy import to avoid circular dependency
        from semblend_core.segment_embeddings import EmbedResult, SegmentEmbeddings

        tokenizer = self._model.tokenizer
        full_ids = tokenizer.encode(text, add_special_tokens=False)

        usable = chunk_size  # no overlap — aligned to KV block boundaries
        n_chunks = max(1, (len(full_ids) + usable - 1) // usable)

        # Single-chunk fast path: no per-segment detail needed
        if n_chunks == 1:
            embedding = self._model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            pooled = embedding.flatten()
            return EmbedResult(pooled=pooled, segments=None)

        # Build non-overlapping chunk texts and token ranges
        chunk_texts: list[str] = []
        token_ranges: list[tuple[int, int]] = []

        for i in range(n_chunks):
            start = i * usable
            end = min(start + usable, len(full_ids))
            chunk_ids = full_ids[start:end]
            if not chunk_ids:
                break
            chunk_texts.append(
                tokenizer.decode(chunk_ids, skip_special_tokens=True),
            )
            token_ranges.append((start, end))

        # Batch-embed all chunks in a single forward pass
        segment_matrix = self._model.encode(
            chunk_texts,
            batch_size=len(chunk_texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )  # [n_chunks, 384]

        # Mean-pool across segments for the pooled prompt embedding
        pooled = segment_matrix.mean(axis=0)
        norm = np.linalg.norm(pooled)
        pooled = pooled / max(norm, 1e-12)

        segments = SegmentEmbeddings(
            matrix=segment_matrix,
            chunk_token_ranges=tuple(token_ranges),
            chunk_size=chunk_size,
        )

        return EmbedResult(pooled=pooled, segments=segments)


class JinaEmbedder:
    """HTTP-based jina-v4 embedder (kept for benchmarks and fallback)."""

    DIMENSION = 1024

    def __init__(self, url: str | None = None) -> None:
        self._url = url or os.environ.get(
            "SEMBLEND_JINA_URL", "http://synapse-staging-jina-v4:8080"
        )
        self._available = bool(self._url)

    @property
    def available(self) -> bool:
        return self._available

    @property
    def dimension(self) -> int:
        return self.DIMENSION

    def embed(self, text: str) -> np.ndarray | None:
        """Embed text via jina-v4 HTTP endpoint."""
        if not self._available or not text.strip():
            return None

        try:
            import requests

            resp = requests.post(
                f"{self._url}/embed",
                json={"inputs": [text[:2000]]},
                timeout=10.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and data:
                    vec = data[0] if isinstance(data[0], list) else data
                    return np.array(vec, dtype=np.float32)
        except Exception:
            logger.warning("Jina embedding failed", exc_info=True)

        return None


class OnnxGpuEmbedder:
    """ONNX Runtime GPU embedder for MiniLM — ~2ms vs 58ms on CPU.

    Uses ONNX Runtime with CUDA EP to run MiniLM on the same GPU as vLLM.
    MiniLM is tiny (22M params, ~88MB) so it coexists with vLLM easily.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"
    DIMENSION = 384

    def __init__(self) -> None:
        self._session = None
        self._tokenizer = None
        self._available = False
        self._load_time_ms = 0.0
        self._init()

    def _init(self) -> None:
        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer

            t0 = time.monotonic()

            # Download and convert model to ONNX if needed
            model_path = self._get_onnx_model_path()
            if model_path is None:
                logger.warning("ONNX model not available, skipping GPU embedder")
                return

            # Create ONNX session with CUDA EP + memory limit
            # MiniLM is 22M params (~88MB). Limit GPU arena to 256MB to prevent
            # workspace buffer growth from starving the inference model's KV cache.
            cuda_ep_opts = {
                "device_id": 0,
                "gpu_mem_limit": 256 * 1024 * 1024,  # 256MB max
                "arena_extend_strategy": "kSameAsRequested",  # don't over-allocate
            }
            providers = [
                ("CUDAExecutionProvider", cuda_ep_opts),
                "CPUExecutionProvider",
            ]
            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_opts.enable_mem_pattern = False  # prevent memory pattern caching growth
            self._session = ort.InferenceSession(
                str(model_path), sess_options=sess_opts, providers=providers
            )

            # Check which EP is actually being used
            actual_provider = self._session.get_providers()[0]

            self._tokenizer = AutoTokenizer.from_pretrained(
                f"sentence-transformers/{self.MODEL_NAME}"
            )

            self._load_time_ms = (time.monotonic() - t0) * 1000
            self._available = True
            logger.info(
                "ONNX GPU embedder loaded: %s on %s (%.0fms)",
                self.MODEL_NAME, actual_provider, self._load_time_ms,
            )
        except ImportError:
            logger.warning("onnxruntime not installed - ONNX GPU embedder disabled")
        except Exception:
            logger.warning("ONNX GPU embedder init failed", exc_info=True)

    # Version tag for ONNX cache invalidation — bump when export config changes.
    _ONNX_VERSION = "v2-dynbatch"

    def _get_onnx_model_path(self):
        """Get path to ONNX model, exporting from PyTorch if needed.

        Uses a versioned filename so stale exports (e.g., without
        dynamic batch axes) are automatically re-exported.
        """
        from pathlib import Path

        cache_dir = Path.home() / ".cache" / "semblend_onnx"
        onnx_path = cache_dir / f"{self.MODEL_NAME}-{self._ONNX_VERSION}.onnx"

        if onnx_path.exists():
            return onnx_path

        # Export from sentence-transformers to ONNX
        try:
            import torch
            from sentence_transformers import SentenceTransformer

            cache_dir.mkdir(parents=True, exist_ok=True)
            model = SentenceTransformer(self.MODEL_NAME, device="cpu")

            # Get the underlying transformer model
            transformer = model[0].auto_model
            tokenizer = model.tokenizer

            dummy = tokenizer("test", return_tensors="pt", padding=True, truncation=True)
            input_names = ["input_ids", "attention_mask"]
            if "token_type_ids" in dummy:
                input_names.append("token_type_ids")

            torch.onnx.export(
                transformer,
                tuple(dummy[k] for k in input_names),
                str(onnx_path),
                input_names=input_names,
                output_names=["last_hidden_state"],
                dynamic_axes={
                    name: {0: "batch", 1: "seq"} for name in input_names
                } | {"last_hidden_state": {0: "batch", 1: "seq"}},
                opset_version=14,
            )
            logger.info("Exported ONNX model to %s", onnx_path)
            return onnx_path

        except Exception:
            logger.warning("ONNX export failed", exc_info=True)
            return None

    @property
    def available(self) -> bool:
        return self._available

    @property
    def dimension(self) -> int:
        return self.DIMENSION

    # Segmented embedding parameters
    MAX_TOKENS = 256       # model context window per segment
    OVERLAP_TOKENS = 64    # overlap between adjacent segments
    MIN_SEGMENT_TOKENS = 32  # skip trailing segments shorter than this

    def embed(self, text: str) -> np.ndarray | None:
        """Embed text using ONNX Runtime GPU with full-document coverage.

        For short texts (≤MAX_TOKENS): single-pass embedding (~2ms).
        For long texts: segments the text into overlapping windows,
        embeds all segments in a single batched forward pass, and
        mean-pools the segment embeddings into one vector (~10-15ms
        for 32K-token prompts on A10G).
        """
        if not self._available or not text.strip():
            return None

        t0 = time.monotonic()

        # Tokenize full text to determine if segmentation is needed
        full_ids = self._tokenizer.encode(text, add_special_tokens=False)
        usable = self.MAX_TOKENS - 2  # reserve [CLS] and [SEP]

        if len(full_ids) <= usable:
            # Short text: single pass
            embedding = self._embed_single(text)
        else:
            # Long text: segment with overlap, batch embed, mean pool
            embedding = self._embed_segmented(full_ids)

        elapsed_ms = (time.monotonic() - t0) * 1000
        if elapsed_ms > 5:
            n_seg = max(1, (len(full_ids) - usable) // (usable - self.OVERLAP_TOKENS) + 2)
            logger.debug(
                "ONNX GPU embed took %.1fms (%d tokens, %d segments)",
                elapsed_ms, len(full_ids), n_seg,
            )

        return embedding

    def _embed_single(self, text: str) -> np.ndarray:
        """Single-pass embedding for short texts."""
        inputs = self._tokenizer(
            text,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=self.MAX_TOKENS,
        )
        return self._run_and_pool(inputs)

    def _embed_segmented(self, full_ids: list[int]) -> np.ndarray:
        """Segment long token sequences and embed on GPU with mean pooling.

        Attempts batched ONNX inference (single forward pass for all
        segments). Falls back to sequential per-segment inference if the
        ONNX model was cached without dynamic batch support.
        """
        usable = self.MAX_TOKENS - 2
        stride = usable - self.OVERLAP_TOKENS

        segments = []
        for start in range(0, len(full_ids), stride):
            chunk_ids = full_ids[start:start + usable]
            if len(chunk_ids) < self.MIN_SEGMENT_TOKENS:
                break
            chunk_text = self._tokenizer.decode(chunk_ids, skip_special_tokens=True)
            segments.append(chunk_text)

        if not segments:
            text = self._tokenizer.decode(full_ids[:usable], skip_special_tokens=True)
            return self._embed_single(text)

        # Try batched inference first (fastest — single GPU kernel launch)
        try:
            inputs = self._tokenizer(
                segments,
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=self.MAX_TOKENS,
            )
            per_segment = self._run_and_pool(inputs)  # [N, 384]
        except Exception:
            # Fallback: loop over segments individually
            logger.debug(
                "Batched ONNX failed (%d segments), falling back to sequential",
                len(segments),
            )
            embeddings = []
            for seg_text in segments:
                emb = self._embed_single(seg_text)
                if emb.ndim == 2:
                    emb = emb[0]
                embeddings.append(emb)
            per_segment = np.stack(embeddings)

        pooled = per_segment.mean(axis=0) if per_segment.ndim == 2 else per_segment

        norm = np.linalg.norm(pooled)
        return pooled / max(norm, 1e-12)

    def _run_and_pool(self, inputs: dict) -> np.ndarray:
        """Run ONNX inference and apply mean pooling."""
        feed = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
        if "token_type_ids" in inputs:
            feed["token_type_ids"] = inputs["token_type_ids"]

        outputs = self._session.run(None, feed)
        hidden = outputs[0]  # [batch, seq_len, 384]
        mask = inputs["attention_mask"].astype(np.float32)
        masked = hidden * mask[:, :, np.newaxis]
        pooled = masked.sum(axis=1) / mask.sum(axis=1, keepdims=True)

        # L2 normalize each segment embedding
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        normalized = pooled / np.maximum(norms, 1e-12)

        return normalized  # [batch, 384] or [1, 384]

    def embed_with_segments(
        self, text: str, chunk_size: int = 256,
    ) -> "EmbedResult | None":
        """Embed text with per-segment embeddings aligned to KV block boundaries.

        Same as MiniLMEmbedder.embed_with_segments but uses ONNX GPU inference.
        Produces per-chunk embeddings that can be stored in the PQ segment store
        for semantic cross-donor chunk matching.
        """
        if not self._available or not text.strip():
            return None

        from semblend_core.segment_embeddings import EmbedResult, SegmentEmbeddings

        full_ids = self._tokenizer.encode(text, add_special_tokens=False)
        usable = chunk_size
        n_chunks = max(1, (len(full_ids) + usable - 1) // usable)

        # Single-chunk fast path
        if n_chunks == 1:
            pooled = self._embed_single(text)
            if pooled is None:
                return None
            return EmbedResult(pooled=pooled.flatten(), segments=None)

        # Build non-overlapping chunk texts aligned to KV block boundaries
        chunk_texts: list[str] = []
        token_ranges: list[tuple[int, int]] = []

        for i in range(n_chunks):
            start = i * usable
            end = min(start + usable, len(full_ids))
            chunk_ids = full_ids[start:end]
            if not chunk_ids:
                break
            chunk_texts.append(
                self._tokenizer.decode(chunk_ids, skip_special_tokens=True),
            )
            token_ranges.append((start, end))

        if not chunk_texts:
            return None

        # Batch-embed all chunks via ONNX GPU
        try:
            inputs = self._tokenizer(
                chunk_texts,
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=self.MAX_TOKENS,
            )
            segment_matrix = self._run_and_pool(inputs)
        except Exception:
            # Fallback to sequential if batched fails
            segment_list = []
            for ct in chunk_texts:
                vec = self._embed_single(ct)
                if vec is not None:
                    segment_list.append(vec.flatten())
            if not segment_list:
                return None
            segment_matrix = np.stack(segment_list, axis=0)

        # Mean-pool for the pooled embedding
        pooled = segment_matrix.mean(axis=0)
        norm = np.linalg.norm(pooled)
        pooled = pooled / max(norm, 1e-12)

        segments = SegmentEmbeddings(
            matrix=segment_matrix,
            chunk_token_ranges=tuple(token_ranges),
            chunk_size=chunk_size,
        )

        return EmbedResult(pooled=pooled, segments=segments)


class E5SmallEmbedder:
    """In-process CPU embedder using intfloat/e5-small-v2.

    E5 models use instruction prefixes ("query: " / "passage: ") for
    better semantic clustering on instruction-following tasks.
    384-dim output, same as MiniLM. ~33M params, ~130MB.
    """

    MODEL_NAME = "intfloat/e5-small-v2"
    DIMENSION = 384

    def __init__(self) -> None:
        self._model = None
        self._available = False
        self._load_time_ms = 0.0
        self._init()

    def _init(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            t0 = time.monotonic()
            self._model = SentenceTransformer(self.MODEL_NAME, device="cpu")
            self._load_time_ms = (time.monotonic() - t0) * 1000
            self._available = True
            logger.info(
                "E5-small embedder loaded: %s (%.0fms)",
                self.MODEL_NAME, self._load_time_ms,
            )
        except ImportError:
            logger.warning("sentence-transformers not installed - E5 disabled")
        except Exception:
            logger.warning("E5-small model load failed", exc_info=True)

    @property
    def available(self) -> bool:
        return self._available

    @property
    def dimension(self) -> int:
        return self.DIMENSION

    def embed(self, text: str) -> np.ndarray | None:
        """Embed text with E5 query prefix for semantic matching."""
        if not self._available or not text.strip():
            return None

        t0 = time.monotonic()
        # E5 models require "query: " prefix for retrieval tasks
        prefixed = f"query: {text}"
        embedding = self._model.encode(
            prefixed,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000

        if elapsed_ms > 10:
            logger.debug("E5-small embed took %.1fms (text=%d chars)", elapsed_ms, len(text))

        return embedding


class JaccardEmbedder:
    """Null embedder — uses no embedding model.

    When this embedder is active, the donor store relies entirely on
    SimHash pre-filtering and token-level alignment (no cosine similarity).
    Used for testing and environments without model dependencies.
    """

    DIMENSION = 384  # Matches MiniLM for compatibility

    @property
    def available(self) -> bool:
        return True

    @property
    def dimension(self) -> int:
        return self.DIMENSION

    def embed(self, text: str) -> np.ndarray | None:
        return None


def create_embedder(
    embedder_type: str | None = None,
) -> MiniLMEmbedder | OnnxGpuEmbedder | E5SmallEmbedder | JinaEmbedder | JaccardEmbedder:
    """Create an embedder based on configuration.

    Args:
        embedder_type: "onnx-gpu", "minilm", "e5", "jina", "jaccard", or None (auto from env).

    Returns:
        Embedder instance.
    """
    choice = (embedder_type or os.environ.get("SEMBLEND_EMBEDDER", "minilm")).lower()

    if choice == "jaccard":
        return JaccardEmbedder()

    if choice == "jina":
        return JinaEmbedder()

    if choice in ("e5", "e5-small", "e5_small"):
        embedder = E5SmallEmbedder()
        if embedder.available:
            return embedder
        logger.warning("E5-small unavailable, falling back to MiniLM")
        return MiniLMEmbedder()

    if choice in ("onnx-gpu", "onnx_gpu", "onnxgpu"):
        embedder = OnnxGpuEmbedder()
        if embedder.available:
            return embedder
        logger.warning("ONNX GPU embedder unavailable, falling back to MiniLM CPU")

    # Default: try ONNX GPU first if GPU is available, then MiniLM CPU
    if choice == "minilm":
        # Auto-detect: try ONNX GPU first for lower latency
        try:
            import torch
            if torch.cuda.is_available():
                onnx_embedder = OnnxGpuEmbedder()
                if onnx_embedder.available:
                    logger.info("Auto-selected ONNX GPU embedder (CUDA available)")
                    return onnx_embedder
        except ImportError:
            pass

    embedder = MiniLMEmbedder()
    if not embedder.available:
        logger.warning("MiniLM unavailable, falling back to Jina")
        return JinaEmbedder()
    return embedder

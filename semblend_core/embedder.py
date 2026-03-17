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
    """In-process CPU embedder using all-MiniLM-L6-v2.

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
            self._model = SentenceTransformer(self.MODEL_NAME, device="cpu")
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

    def embed(self, text: str) -> np.ndarray | None:
        """Embed text to a 384-dim vector.

        Args:
            text: Input text (truncated to first 512 tokens by model).

        Returns:
            Normalized embedding vector [384], or None if unavailable.
        """
        if not self._available or not text.strip():
            return None

        t0 = time.monotonic()
        embedding = self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000

        if elapsed_ms > 10:
            logger.debug("MiniLM embed took %.1fms (text=%d chars)", elapsed_ms, len(text))

        return embedding


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

            # Create ONNX session with CUDA EP
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
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

    def _get_onnx_model_path(self):
        """Get path to ONNX model, exporting from PyTorch if needed."""
        from pathlib import Path

        cache_dir = Path.home() / ".cache" / "semblend_onnx"
        onnx_path = cache_dir / f"{self.MODEL_NAME}.onnx"

        if onnx_path.exists():
            return onnx_path

        # Export from sentence-transformers to ONNX
        try:
            from sentence_transformers import SentenceTransformer
            import torch

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

    def embed(self, text: str) -> np.ndarray | None:
        """Embed text using ONNX Runtime GPU — ~2ms."""
        if not self._available or not text.strip():
            return None

        t0 = time.monotonic()
        inputs = self._tokenizer(
            text,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=256,
        )

        feed = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
        if "token_type_ids" in inputs:
            feed["token_type_ids"] = inputs["token_type_ids"]

        outputs = self._session.run(None, feed)
        # Mean pooling over token dimension
        hidden = outputs[0]  # [1, seq_len, 384]
        mask = inputs["attention_mask"].astype(np.float32)
        masked = hidden * mask[:, :, np.newaxis]
        pooled = masked.sum(axis=1) / mask.sum(axis=1, keepdims=True)

        # L2 normalize
        norm = np.linalg.norm(pooled, axis=1, keepdims=True)
        embedding = (pooled / np.maximum(norm, 1e-12))[0]

        elapsed_ms = (time.monotonic() - t0) * 1000
        if elapsed_ms > 5:
            logger.debug("ONNX GPU embed took %.1fms (text=%d chars)", elapsed_ms, len(text))

        return embedding


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

"""Token embedding capture from KV cache layer 0.

Captures per-token embeddings from the first transformer layer's K vectors
for LSH indexing. This avoids modifying vLLM's model forward pass — instead,
we extract embeddings from the KV cache that's already being saved.

The K vectors from layer 0 are linear projections of the token embeddings:
    K_0 = W_K @ embed_tokens(token_ids) + b_K

While not identical to raw token embeddings, they carry the same semantic
information and are suitable for LSH matching. This is equivalent to
SemShareKV's approach but extracted post-projection rather than pre-projection.

Integration: Called from save_kv_layer() when layer_idx == 0 and
SEMBLEND_SELECTIVE_RECOMPUTE=1.
"""
from __future__ import annotations

import logging
import threading
from collections import OrderedDict

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingCaptureBuffer:
    """Thread-safe buffer for capturing layer-0 K vectors as token embeddings.

    Stores per-request K vectors from layer 0 for later LSH indexing.
    The buffer is size-limited and auto-evicts oldest entries.
    """

    def __init__(self, max_entries: int = 1000) -> None:
        self._buffer: OrderedDict[str, np.ndarray] = OrderedDict()
        self._max_entries = max_entries
        self._lock = threading.Lock()

    def capture(self, request_id: str, k_vectors: np.ndarray) -> None:
        """Capture K vectors from layer 0 for a request.

        Args:
            request_id: Request identifier.
            k_vectors: shape [seq_len, num_kv_heads, head_dim] or [seq_len, dim]
                K vectors from the first transformer layer.
        """
        # Reshape to [seq_len, dim] if needed (collapse head dimensions)
        if k_vectors.ndim == 3:
            seq_len, num_heads, head_dim = k_vectors.shape
            flat = k_vectors.reshape(seq_len, num_heads * head_dim)
        elif k_vectors.ndim == 2:
            flat = k_vectors
        else:
            logger.warning(
                "Unexpected K vector shape %s for req=%s", k_vectors.shape, request_id
            )
            return

        # Store as float32 numpy on CPU
        embeddings = flat.astype(np.float32) if flat.dtype != np.float32 else flat

        with self._lock:
            if request_id in self._buffer:
                return  # already captured
            self._buffer[request_id] = embeddings
            # Evict oldest if over capacity
            while len(self._buffer) > self._max_entries:
                self._buffer.popitem(last=False)

    def pop(self, request_id: str) -> np.ndarray | None:
        """Retrieve and remove captured embeddings for a request.

        Args:
            request_id: Request identifier.

        Returns:
            numpy array of shape [seq_len, dim] or None if not found.
        """
        with self._lock:
            return self._buffer.pop(request_id, None)

    def get(self, request_id: str) -> np.ndarray | None:
        """Retrieve captured embeddings without removing.

        Args:
            request_id: Request identifier.

        Returns:
            numpy array of shape [seq_len, dim] or None if not found.
        """
        with self._lock:
            return self._buffer.get(request_id)

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._buffer)

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()


def extract_layer0_embeddings(
    kv_layer: object,
    layer_name: str,
) -> np.ndarray | None:
    """Extract K vectors from a KV layer tensor if it's layer 0.

    Args:
        kv_layer: vLLM KV layer tensor. Expected shape varies by backend:
            - [2, seq_len, num_kv_heads, head_dim] (K=index 0, V=index 1)
            - Or torch.Tensor on CUDA
        layer_name: Name like "layers.0.self_attn.kv_proj" — we only capture layer 0.

    Returns:
        K vectors as numpy array [seq_len, num_kv_heads * head_dim] or None.
    """
    # Only capture from layer 0
    if not _is_layer_zero(layer_name):
        return None

    try:
        import torch

        if not isinstance(kv_layer, torch.Tensor):
            return None

        # Extract K vectors (index 0 along the K/V dimension)
        if kv_layer.ndim == 4:
            # [2, seq_len, num_heads, head_dim] — standard layout
            k = kv_layer[0]  # [seq_len, num_heads, head_dim]
        elif kv_layer.ndim == 3:
            # [seq_len, 2 * num_heads, head_dim] — interleaved
            num_heads = kv_layer.shape[1] // 2
            k = kv_layer[:, :num_heads, :]  # [seq_len, num_heads, head_dim]
        else:
            return None

        # Move to CPU and convert to numpy
        k_np = k.detach().cpu().float().numpy()
        # Reshape to [seq_len, dim]
        seq_len = k_np.shape[0]
        return k_np.reshape(seq_len, -1)

    except Exception:
        logger.debug("Failed to extract layer 0 embeddings", exc_info=True)
        return None


def _is_layer_zero(layer_name: str) -> bool:
    """Check if this is layer 0 based on the layer name string."""
    # vLLM layer names: "model.layers.0.self_attn.kv_proj", "layers.0.attn", etc.
    name = layer_name.lower()
    for pattern in ("layers.0.", "layer.0.", "layer_0.", "layer0."):
        if pattern in name:
            return True
    # Also check for explicit layer index
    if ".0." in name and "layer" in name:
        return True
    return False

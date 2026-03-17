"""Re-exports from semblend_core.embedder.

All embedder logic lives in the shared semblend_core package.
This module re-exports everything for existing import paths.
"""
from semblend_core.embedder import (  # noqa: F401
    EmbedderType,
    JaccardEmbedder,
    JinaEmbedder,
    MiniLMEmbedder,
    OnnxGpuEmbedder,
    create_embedder,
)

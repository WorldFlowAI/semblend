"""Launch SGLang with SemBlend semantic KV cache reuse.

Patches SGLang's RadixCache before the server starts, then delegates
to SGLang's normal server launch flow.

Usage:
    python -m semblend.integration.sglang.launch_semblend_sglang \\
        --model-path Qwen/Qwen2.5-7B-Instruct \\
        --host 0.0.0.0 --port 8000

Environment variables (SemBlend-specific):
    SEMBLEND_ENABLED=1              Enable semantic fallback (default: 1)
    SEMBLEND_MODEL_NAME=<model>     Model name for tokenizer (auto-detected
                                    from --model-path if not set)
    SEMBLEND_MIN_SIMILARITY=0.60    Cosine similarity threshold
    SEMBLEND_EMBEDDER=minilm        Embedder type (minilm, onnx_gpu, e5, jaccard)
    SEMBLEND_MAX_DONORS=1000        Maximum donors in the semantic store
"""

from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger("semblend.sglang.launch")


def _auto_detect_model_name() -> None:
    """Set SEMBLEND_MODEL_NAME from --model-path if not already set."""
    if os.environ.get("SEMBLEND_MODEL_NAME"):
        return

    for i, arg in enumerate(sys.argv):
        if arg == "--model-path" and i + 1 < len(sys.argv):
            model_name = sys.argv[i + 1]
            os.environ["SEMBLEND_MODEL_NAME"] = model_name
            logger.info(f"Auto-detected SEMBLEND_MODEL_NAME={model_name} from --model-path")
            return


def main() -> None:
    """Patch RadixCache, then launch SGLang server."""
    import multiprocessing

    multiprocessing.freeze_support()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    _auto_detect_model_name()

    try:
        from .radix_patcher import patch_radix_cache
    except ImportError:
        # Handle direct execution (python -m ...)
        from semblend.integration.sglang.radix_patcher import patch_radix_cache

    try:
        patch_radix_cache()
    except ImportError as exc:
        logger.error(f"Cannot patch RadixCache: {exc}")
        logger.error("Ensure SGLang is installed: pip install sglang")
        sys.exit(1)

    logger.info("Launching SGLang with SemBlend semantic KV cache reuse")

    try:
        from sglang.srt.entrypoints.http_server import launch_server
        from sglang.srt.server_args import prepare_server_args
        from sglang.srt.utils import kill_process_tree
    except ImportError:
        logger.error("sglang.srt not found. Ensure SGLang is installed: pip install sglang")
        sys.exit(1)

    server_args = prepare_server_args(sys.argv[1:])
    try:
        launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()

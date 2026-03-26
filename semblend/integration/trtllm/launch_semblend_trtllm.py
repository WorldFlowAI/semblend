"""Launch TRT-LLM with SemBlend semantic KV cache reuse.

Configures the TRT-LLM PyTorch backend with SemBlend hooks, then starts
serving with an OpenAI-compatible API endpoint.

Usage:
    # Via entry point:
    semblend-trtllm --engine-dir ./engines/qwen2.5-7b \\
        --model Qwen/Qwen2.5-7B-Instruct-AWQ \\
        --host 0.0.0.0 --port 8000

    # Via module:
    python -m semblend.integration.trtllm.launch_semblend_trtllm \\
        --engine-dir ./engines/qwen2.5-7b \\
        --model Qwen/Qwen2.5-7B-Instruct-AWQ

    # Via environment variables:
    SEMBLEND_ENABLED=1 SEMBLEND_MIN_SIMILARITY=0.60 \\
    python -m semblend.integration.trtllm.launch_semblend_trtllm \\
        --engine-dir ./engines/qwen2.5-7b

Environment variables (SemBlend-specific):
    SEMBLEND_ENABLED=1              Enable semantic donor search (default: 1)
    SEMBLEND_MODEL_NAME=<model>     Model name for tokenizer (auto-detected
                                    from --model if not set)
    SEMBLEND_MIN_SIMILARITY=0.60    Cosine similarity threshold
    SEMBLEND_EMBEDDER=minilm        Embedder type (minilm, jaccard, onnx_gpu)
    SEMBLEND_MAX_DONORS=1000        Maximum donors in the semantic store
    SEMBLEND_MIN_REUSE_RATIO=0.50   Minimum alignment reuse ratio
    SEMBLEND_TRTLLM_APPROACH=auto   Hook approach (auto, token_sub, radix_patch)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any

logger = logging.getLogger("semblend.trtllm.launch")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch TRT-LLM with SemBlend semantic KV cache reuse",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--engine-dir",
        required=True,
        help="Path to TRT-LLM engine directory",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Model name/path for tokenizer loading",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Server host address",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=64,
        help="Maximum batch size for inference",
    )
    parser.add_argument(
        "--tokens-per-block",
        type=int,
        default=128,
        help="KV cache block size in tokens",
    )
    parser.add_argument(
        "--semblend-enabled",
        action="store_true",
        default=True,
        help="Enable SemBlend semantic KV reuse",
    )
    parser.add_argument(
        "--semblend-min-similarity",
        type=float,
        default=0.60,
        help="Minimum cosine similarity for semantic matching",
    )
    parser.add_argument(
        "--semblend-approach",
        default="auto",
        choices=["auto", "token_sub", "radix_patch", "block_inject"],
        help="SemBlend hook approach for TRT-LLM interception",
    )
    parser.add_argument(
        "--no-semblend",
        action="store_true",
        help="Disable SemBlend (baseline mode)",
    )
    return parser.parse_args()


def _setup_env(args: argparse.Namespace) -> None:
    """Configure environment variables from CLI args."""
    if args.no_semblend:
        os.environ["SEMBLEND_ENABLED"] = "0"
    elif args.semblend_enabled:
        os.environ.setdefault("SEMBLEND_ENABLED", "1")

    if args.model:
        os.environ.setdefault("SEMBLEND_MODEL_NAME", args.model)

    os.environ.setdefault("SEMBLEND_MIN_SIMILARITY", str(args.semblend_min_similarity))
    os.environ.setdefault("SEMBLEND_TRTLLM_APPROACH", args.semblend_approach)


def main() -> None:
    """Configure SemBlend hooks and launch TRT-LLM server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    args = _parse_args()
    _setup_env(args)

    # Validate engine directory
    engine_dir = os.path.abspath(args.engine_dir)
    if not os.path.isdir(engine_dir):
        logger.error("Engine directory not found: %s", engine_dir)
        sys.exit(1)

    logger.info("SemBlend TRT-LLM launcher starting")
    logger.info("  Engine directory: %s", engine_dir)
    logger.info("  Model: %s", args.model or "(auto-detect)")
    logger.info("  SemBlend enabled: %s", not args.no_semblend)
    logger.info("  Approach: %s", args.semblend_approach)

    try:
        _launch_server(args, engine_dir)
    except ImportError as exc:
        logger.error(
            "TensorRT-LLM is required but not installed: %s\n"
            "Install with: pip install tensorrt_llm",
            exc,
        )
        sys.exit(1)
    except Exception as exc:
        logger.error("Server launch failed: %s", exc, exc_info=True)
        sys.exit(1)


def _launch_server(args: argparse.Namespace, engine_dir: str) -> None:
    """Import TRT-LLM and launch the server with SemBlend hooks.

    This function is separated to keep the import error handling clean.
    """
    # Import TRT-LLM components
    try:
        from tensorrt_llm import LLM
        from tensorrt_llm.serve import OpenAIServer
    except ImportError:
        # Fall back to older API names
        from tensorrt_llm import LLM  # type: ignore[assignment]
        from tensorrt_llm.serve import OpenAIServer  # type: ignore[assignment]

    # Build LLM instance
    logger.info("Loading TRT-LLM engine from %s", engine_dir)
    llm = LLM(
        model=engine_dir,
        enable_prefix_caching=True,
        tokens_per_block=args.tokens_per_block,
    )

    # Attach SemBlend hooks if enabled
    if os.environ.get("SEMBLEND_ENABLED", "1") == "1":
        _attach_semblend(llm, args)

    # Start OpenAI-compatible server
    logger.info(
        "Starting OpenAI-compatible server at http://%s:%d",
        args.host,
        args.port,
    )
    server = OpenAIServer(llm)
    server.start(host=args.host, port=args.port)


def _attach_semblend(llm: Any, args: argparse.Namespace) -> None:
    """Attach SemBlend hooks to the TRT-LLM instance.

    Discovers the KVCacheManager and ModelEngine from the LLM instance,
    creates the backend and hook, and wires everything together.
    """
    from semblend.integration.trtllm.model_engine_hook import (
        SemBlendModelEngineHook,
    )
    from semblend.integration.trtllm.pytorch_backend import TRTLLMPyTorchBackend

    # Discover KVCacheManager from the LLM instance
    # TRT-LLM's internal structure varies by version; probe common paths
    kv_mgr = None
    engine = None

    for attr in ("_engine", "engine", "_model_engine", "model_engine"):
        candidate = getattr(llm, attr, None)
        if candidate is not None:
            engine = candidate
            for mgr_attr in ("kv_cache_manager", "_kv_cache_manager", "kv_mgr"):
                kv_candidate = getattr(candidate, mgr_attr, None)
                if kv_candidate is not None:
                    kv_mgr = kv_candidate
                    break
            if kv_mgr is not None:
                break

    if kv_mgr is None:
        logger.warning(
            "Could not discover KVCacheManager from LLM instance. "
            "SemBlend hooks disabled. TRT-LLM version may not expose "
            "the PyTorch backend's KV manager."
        )
        return

    # Build model config from LLM metadata
    model_config = {
        "tokens_per_block": args.tokens_per_block,
        "model_name": args.model or os.environ.get("SEMBLEND_MODEL_NAME", ""),
    }
    # Try to extract additional config from engine
    for config_attr in ("config", "model_config", "_config"):
        cfg = getattr(engine, config_attr, None)
        if cfg is not None:
            if hasattr(cfg, "num_hidden_layers"):
                model_config["num_layers"] = cfg.num_hidden_layers
            if hasattr(cfg, "num_key_value_heads"):
                model_config["num_kv_heads"] = cfg.num_key_value_heads
            if hasattr(cfg, "hidden_size") and hasattr(cfg, "num_attention_heads"):
                model_config["head_dim"] = cfg.hidden_size // cfg.num_attention_heads
            if hasattr(cfg, "rope_theta"):
                model_config["rope_base"] = cfg.rope_theta
            break

    backend = TRTLLMPyTorchBackend(
        kv_cache_manager=kv_mgr,
        model_config=model_config,
    )

    hook = SemBlendModelEngineHook(engine=engine, backend=backend)
    approach = hook.wrap()
    logger.info("SemBlend hooks attached: approach=%s", approach)


if __name__ == "__main__":
    main()

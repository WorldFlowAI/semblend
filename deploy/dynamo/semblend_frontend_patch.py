"""SemBlend patch for Dynamo Frontend — semantic KV cache lookup.

This script patches the Dynamo frontend's VllmProcessor to add SemBlend
semantic donor discovery before the KV router's generate() call.

When the KV router's RadixTree would miss (low overlap), SemBlend's
embedding search finds a semantically similar cached prompt and substitutes
the donor's token IDs. The KV router then matches the donor's blocks.

Usage (inside the Dynamo frontend container):
    pip install semblend sentence-transformers
    python semblend_frontend_patch.py  # patches and starts frontend

For the upstream PR, this logic moves into the Rust KvIndexer as
SemanticKvIndexer (synapse-sem-indexer crate).
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("semblend.dynamo.patch")


def patch_frontend():
    """Monkey-patch the Dynamo frontend to add SemBlend semantic lookup."""
    try:
        from semblend_core.embedder import create_embedder
        from semblend_core.pipeline import SemBlendPipeline
    except ImportError:
        logger.error("semblend not installed — pip install semblend sentence-transformers")
        return False

    min_similarity = float(os.environ.get("SEMBLEND_MIN_SIMILARITY", "0.60"))
    model_name = os.environ.get("SEMBLEND_MODEL_NAME", "")
    enabled = os.environ.get("SEMBLEND_ENABLED", "1") == "1"

    if not enabled:
        logger.info("SemBlend disabled (SEMBLEND_ENABLED=0)")
        return False

    # Initialize pipeline
    pipeline = SemBlendPipeline(
        max_donors=int(os.environ.get("SEMBLEND_MAX_DONORS", "10000")),
        min_similarity=min_similarity,
        min_reuse_ratio=float(os.environ.get("SEMBLEND_MIN_REUSE_RATIO", "0.30")),
        embedder_type=os.environ.get("SEMBLEND_EMBEDDER", "minilm"),
        model_name=model_name,
        chunk_size=int(os.environ.get("SEMBLEND_CHUNK_SIZE", "32")),
    )

    # Tokenizer for donor registration
    tokenizer = None
    if model_name:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            logger.info("SemBlend tokenizer loaded: %s", model_name)
        except Exception as e:
            logger.warning("Tokenizer load failed: %s", e)

    stats = {
        "total_requests": 0,
        "semantic_hits": 0,
        "semantic_misses": 0,
        "donors_registered": 0,
    }

    # Patch the KvRouter.generate method
    try:
        from dynamo._core import KvRouter
    except ImportError:
        logger.error("dynamo._core.KvRouter not found — not a Dynamo runtime")
        return False

    original_generate = KvRouter.generate

    async def patched_generate(self, token_ids, model, *args, **kwargs):
        """Wrapper around KvRouter.generate that adds semantic donor lookup."""
        stats["total_requests"] += 1

        # Try semantic donor discovery
        if len(token_ids) > 100 and pipeline.donor_count > 0:
            prompt_text = None
            if tokenizer:
                try:
                    sampled = token_ids[:2000] if len(token_ids) > 2000 else token_ids
                    prompt_text = tokenizer.decode(sampled, skip_special_tokens=True)
                except Exception:
                    pass

            if prompt_text:
                result = pipeline.find_donor(
                    token_ids=token_ids,
                    prompt_text=prompt_text,
                )

                if result.found:
                    stats["semantic_hits"] += 1
                    logger.info(
                        "SemBlend hit: donor=%s sim=%.3f reuse=%.2f (embed=%.1fms lookup=%.1fms)",
                        result.donor_id,
                        result.similarity,
                        result.reuse_ratio,
                        result.timings.embed_ms,
                        result.timings.lookup_ms,
                    )
                    # Substitute donor tokens — KV router will match donor's blocks
                    token_ids = result.donor_tokens
                else:
                    stats["semantic_misses"] += 1

        # Call original generate
        response_iter = original_generate(self, token_ids, model, *args, **kwargs)

        # Register as donor after generation completes
        # (capture token_ids for donor registration)
        original_token_ids = token_ids

        async def wrap_and_register():
            async for chunk in response_iter:
                yield chunk
            # After stream completes, register as donor
            if tokenizer and len(original_token_ids) > 100:
                try:
                    text = tokenizer.decode(
                        original_token_ids[:2000],
                        skip_special_tokens=True,
                    )
                    pipeline.register_donor(
                        request_id=f"req-{stats['total_requests']}",
                        token_ids=original_token_ids,
                        prompt_text=text,
                    )
                    stats["donors_registered"] += 1
                except Exception:
                    pass

        return wrap_and_register()

    KvRouter.generate = patched_generate
    logger.info(
        "SemBlend patched KvRouter.generate (min_sim=%.2f, model=%s)",
        min_similarity,
        model_name,
    )
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    success = patch_frontend()
    if success:
        logger.info("SemBlend patch applied — starting Dynamo frontend")
    else:
        logger.warning("SemBlend patch NOT applied — starting vanilla frontend")

    # Start the normal Dynamo frontend
    from dynamo.frontend import main

    main()

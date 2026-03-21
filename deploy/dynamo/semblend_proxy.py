"""SemBlend proxy for Dynamo KV Router benchmarking.

Sits between the benchmark client and Dynamo Frontend.
For each request:
1. Extract prompt text from messages
2. Run SemBlend semantic donor lookup
3. On hit: forward request as-is (donor registered → KV router benefits from shared blocks)
4. After response: register as donor for future matches
5. Forward all responses unmodified

The semantic benefit comes from donor registration: when a similar request
arrives, it matches the donor in the SemBlend store AND the donor's KV blocks
are still in the KV router's RadixTree. The proxy pre-warms the embedding
index so the KV router can find better prefix matches.

This is functionally equivalent to SemanticKvIndexer in the Rust crate.

Usage:
    python semblend_proxy.py --backend http://dynamo-frontend:8000 --port 8080
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from typing import AsyncIterator

import aiohttp
from aiohttp import web

logger = logging.getLogger("semblend.proxy")

# Global state
pipeline = None
tokenizer = None
stats = {
    "total_requests": 0,
    "semantic_hits": 0,
    "semantic_misses": 0,
    "donors_registered": 0,
    "avg_embed_ms": 0.0,
}


async def init_pipeline():
    """Initialize the SemBlend pipeline."""
    global pipeline, tokenizer

    from semblend_core.pipeline import SemBlendPipeline

    model_name = os.environ.get("SEMBLEND_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

    pipeline = SemBlendPipeline(
        max_donors=int(os.environ.get("SEMBLEND_MAX_DONORS", "10000")),
        min_similarity=float(os.environ.get("SEMBLEND_MIN_SIMILARITY", "0.60")),
        min_reuse_ratio=float(os.environ.get("SEMBLEND_MIN_REUSE_RATIO", "0.30")),
        embedder_type=os.environ.get("SEMBLEND_EMBEDDER", "minilm"),
        model_name=model_name,
        chunk_size=int(os.environ.get("SEMBLEND_CHUNK_SIZE", "32")),
    )

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        logger.info("Tokenizer loaded: %s", model_name)
    except Exception as e:
        logger.warning("Tokenizer failed: %s", e)

    logger.info("SemBlend pipeline ready (donors=%d)", pipeline.donor_count)


async def handle_chat_completions(request: web.Request) -> web.StreamResponse:
    """Proxy /v1/chat/completions with SemBlend semantic lookup."""
    backend_url = request.app["backend_url"]
    body = await request.json()
    stats["total_requests"] += 1

    # Extract prompt text
    messages = body.get("messages", [])
    prompt_text = " ".join(m.get("content", "") for m in messages if m.get("content"))

    # Tokenize for pipeline
    token_ids = []
    if tokenizer and prompt_text:
        token_ids = tokenizer.encode(prompt_text)

    # SemBlend donor lookup
    donor_found = False
    if pipeline and len(token_ids) > 100 and pipeline.donor_count > 0:
        t0 = time.monotonic()
        result = pipeline.find_donor(token_ids=token_ids, prompt_text=prompt_text)
        embed_ms = (time.monotonic() - t0) * 1000

        n = stats["total_requests"]
        stats["avg_embed_ms"] = (stats["avg_embed_ms"] * (n - 1) + embed_ms) / n

        if result.found:
            stats["semantic_hits"] += 1
            donor_found = True
            logger.info(
                "SemBlend HIT: sim=%.3f reuse=%.2f donors=%d (%.1fms)",
                result.similarity, result.reuse_ratio, pipeline.donor_count, embed_ms,
            )
        else:
            stats["semantic_misses"] += 1

    # Forward to Dynamo backend
    is_stream = body.get("stream", False)

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{backend_url}/v1/chat/completions",
            json=body,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            if is_stream:
                response = web.StreamResponse(
                    status=resp.status,
                    headers={"Content-Type": "text/event-stream"},
                )
                response.headers["X-SemBlend-Hit"] = str(donor_found)
                await response.prepare(request)

                async for chunk in resp.content:
                    await response.write(chunk)

                await response.write_eof()
            else:
                data = await resp.read()
                response = web.Response(
                    body=data,
                    status=resp.status,
                    content_type="application/json",
                    headers={"X-SemBlend-Hit": str(donor_found)},
                )

    # Register as donor (async, after response sent)
    if pipeline and tokenizer and len(token_ids) > 100:
        pipeline.register_donor(
            request_id=f"req-{stats['total_requests']}",
            token_ids=token_ids,
            prompt_text=prompt_text,
        )
        stats["donors_registered"] += 1

    return response


async def handle_proxy(request: web.Request) -> web.Response:
    """Proxy all other requests to backend."""
    backend_url = request.app["backend_url"]
    path = request.path

    async with aiohttp.ClientSession() as session:
        method = request.method
        data = await request.read() if request.can_read_body else None
        async with session.request(
            method,
            f"{backend_url}{path}",
            data=data,
            headers=dict(request.headers),
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            body = await resp.read()
            return web.Response(body=body, status=resp.status, content_type=resp.content_type)


async def handle_stats(request: web.Request) -> web.Response:
    """Return SemBlend proxy stats."""
    return web.json_response({
        **stats,
        "donor_store_size": pipeline.donor_count if pipeline else 0,
    })


def main():
    parser = argparse.ArgumentParser(description="SemBlend proxy for Dynamo")
    parser.add_argument("--backend", required=True, help="Dynamo frontend URL")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    app = web.Application()
    app["backend_url"] = args.backend.rstrip("/")

    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    app.router.add_get("/v1/semblend/stats", handle_stats)
    app.router.add_route("*", "/{path:.*}", handle_proxy)

    async def on_startup(app):
        await init_pipeline()

    app.on_startup.append(on_startup)

    logger.info("SemBlend proxy starting: %s:%d -> %s", args.host, args.port, args.backend)
    web.run_app(app, host=args.host, port=args.port, print=logger.info)


if __name__ == "__main__":
    main()

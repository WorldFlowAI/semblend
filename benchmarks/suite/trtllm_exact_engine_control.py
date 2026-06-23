"""Exact-prefix control for the TensorRT-LLM SemBlend engine hook."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any

PREFIX = (
    "System note: Capacity events in Atlas are routed through the east control plane. "
)


def main() -> None:
    args = _parse_args()
    os.environ["SEMBLEND_ENABLED"] = "1"
    os.environ["SEMBLEND_TRTLLM_AUDIT_PATH"] = args.audit_path
    os.environ["SEMBLEND_TRTLLM_ENABLE_MATERIALIZATION"] = "1"
    os.environ["SEMBLEND_TRTLLM_MIN_MATCH_LENGTH"] = str(args.min_match_length)
    Path(args.audit_path).unlink(missing_ok=True)

    prefix = PREFIX * args.prefix_repeats
    donor_prompt = prefix + "Summarize the note."
    query_prompt = prefix + "Which control plane handles capacity events?"

    baseline = _build_llm(args, use_semblend=False)
    baseline_latency, baseline_text = _timed_generate(
        baseline, query_prompt, args.max_tokens
    )
    del baseline
    gc.collect()
    _empty_cuda_cache()

    semblend = _build_llm(args, use_semblend=True)
    _generate(semblend, donor_prompt, args.max_tokens)
    semblend_latency, semblend_text = _timed_generate(
        semblend, query_prompt, args.max_tokens
    )

    payload = {
        "model": args.model,
        "baseline_latency_s": baseline_latency,
        "semblend_latency_s": semblend_latency,
        "speedup": baseline_latency / max(semblend_latency, 1e-9),
        "baseline_text": baseline_text,
        "semblend_text": semblend_text,
        "audit": _audit_summary(args.audit_path),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_llm(args: argparse.Namespace, *, use_semblend: bool) -> Any:
    from tensorrt_llm import LLM

    kwargs: dict[str, Any] = {
        "model": args.model,
        "backend": "pytorch",
        "cuda_graph_config": None,
    }
    if use_semblend:
        from tensorrt_llm.llmapi.llm_args import KvCacheConfig, KvCacheConnectorConfig

        kwargs["kv_connector_config"] = KvCacheConnectorConfig(
            connector_module="semblend.integration.trtllm.connector",
            connector_scheduler_class="SemBlendKvConnectorScheduler",
            connector_worker_class="SemBlendKvConnectorWorker",
        )
        kwargs["kv_cache_config"] = KvCacheConfig(
            enable_partial_reuse=False,
            tokens_per_block=args.tokens_per_block,
        )
    return LLM(**kwargs)


def _timed_generate(llm: Any, prompt: str, max_tokens: int) -> tuple[float, str]:
    start = time.perf_counter()
    text = _generate(llm, prompt, max_tokens)
    return time.perf_counter() - start, text


def _generate(llm: Any, prompt: str, max_tokens: int) -> str:
    from tensorrt_llm import SamplingParams

    outputs = llm.generate(
        [prompt],
        SamplingParams(max_tokens=max_tokens, temperature=0.0, top_k=1, seed=1),
    )
    return outputs[0].outputs[0].text


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def _audit_summary(path: str) -> dict[str, Any]:
    events = []
    audit_path = Path(path)
    if audit_path.exists():
        for line in audit_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(json.loads(line))
    return {
        "events": len(events),
        "semantic_hits": sum(e.get("event") == "lookup" and e.get("found") for e in events),
        "materializations": sum(e.get("event") == "materialized" for e in events),
        "engine_boundaries": sum(e.get("event") == "engine_blend_boundary" for e in events),
        "last_events": events[-8:],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="results/trtllm-exact-engine-control.json")
    parser.add_argument(
        "--audit-path", default="results/trtllm-exact-engine-control-audit.jsonl"
    )
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--min-match-length", type=int, default=32)
    parser.add_argument("--prefix-repeats", type=int, default=1)
    parser.add_argument("--tokens-per-block", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    main()

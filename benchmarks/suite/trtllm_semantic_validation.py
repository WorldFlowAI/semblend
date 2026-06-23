"""TensorRT-LLM validation harness for SemBlend semantic KV reuse.

Run on a GPU host with TensorRT-LLM installed:

    python -m benchmarks.suite.trtllm_semantic_validation \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --output results/trtllm-semblend.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class Sample:
    name: str
    donor_prompt: str
    query_prompt: str
    reference_terms: tuple[str, ...]


@dataclass
class SampleResult:
    name: str
    baseline_latency_s: float
    semblend_latency_s: float
    speedup: float
    baseline_text: str
    semblend_text: str
    rouge1_f1: float
    token_f1: float
    ppl_delta: float | None = None
    judge_score: float | None = None


ATLAS_NOTE = (
    "The Atlas storage fleet uses RDMA replication, async compaction, and a two-stage "
    "admission controller. Capacity events are routed through the east control plane before "
    "shard movement begins. Hot shards first enter a pending migration queue, then a placement "
    "solver checks rack locality, donor pressure, and recovery budget. The west control plane "
    "only mirrors the final placement decision for audit trails. Operators should not move "
    "shards directly when the admission controller reports saturation. Instead, they should "
    "raise the compaction priority, wait for donor pressure to fall below the yellow threshold, "
    "and let the east control plane issue the movement lease. The fleet records each lease with "
    "a monotonic epoch so workers can reject stale migration commands during failover."
)

NOVAPAY_NOTE = (
    "Article: NovaPay settles cross-border invoices by batching merchant transfers, screening "
    "sanctions lists, and writing a Merkle proof for each batch. Reconciliation runs at 03:00 "
    "UTC after the second sanctions pass completes. The risk service assigns every merchant a "
    "rolling exposure score, while the ledger service stores a signed checkpoint before funds "
    "leave the settlement account. If a checkpoint fails validation, the batch remains parked "
    "and the operations team receives a review task. Merchants see the transfer as pending "
    "until the Merkle proof and exposure score both pass. The payment network publishes a final "
    "settlement receipt only after reconciliation finishes."
)


SAMPLES = (
    Sample(
        name="shifted_ops_note",
        donor_prompt=(
            "Read this operations note carefully before answering. "
            f"{ATLAS_NOTE} Summarize the routing rule."
        ),
        query_prompt=(
            f" {ATLAS_NOTE} Which control plane handles capacity events? "
            "Answer only with the control plane name."
        ),
        reference_terms=("east", "control", "plane"),
    ),
    Sample(
        name="shifted_article_question",
        donor_prompt=(
            "Read carefully before answering. "
            f"{NOVAPAY_NOTE} Summarize the article."
        ),
        query_prompt=(
            f" {NOVAPAY_NOTE} What time does reconciliation run? "
            "Answer only with the UTC time."
        ),
        reference_terms=("03:00", "utc"),
    ),
)


def main() -> None:
    args = _parse_args()
    results = run_validation(args)
    payload = {
        "model": args.model,
        "num_samples": len(results),
        "mean_speedup": sum(r.speedup for r in results) / max(len(results), 1),
        "mean_rouge1_f1": sum(r.rouge1_f1 for r in results) / max(len(results), 1),
        "mean_token_f1": sum(r.token_f1 for r in results) / max(len(results), 1),
        "audit": _summarize_audit(args.audit_path),
        "samples": [asdict(r) for r in results],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if payload["mean_speedup"] < args.min_speedup:
        raise SystemExit(
            f"mean speedup {payload['mean_speedup']:.3f} < required {args.min_speedup:.3f}"
        )
    if payload["mean_rouge1_f1"] < args.min_rouge1:
        raise SystemExit(
            f"mean rouge1_f1 {payload['mean_rouge1_f1']:.3f} < required {args.min_rouge1:.3f}"
        )
    if payload["mean_token_f1"] < args.min_token_f1:
        raise SystemExit(
            f"mean token_f1 {payload['mean_token_f1']:.3f} < required {args.min_token_f1:.3f}"
        )
    if payload["audit"]["donor_registrations"] < args.min_donor_registrations:
        raise SystemExit(
            "donor registrations "
            f"{payload['audit']['donor_registrations']} < required "
            f"{args.min_donor_registrations}"
        )
    if payload["audit"]["semantic_hits"] < args.min_semantic_hits:
        raise SystemExit(
            f"semantic hits {payload['audit']['semantic_hits']} < required "
            f"{args.min_semantic_hits}"
        )
    if payload["audit"]["materializations"] < args.min_materializations:
        raise SystemExit(
            f"materializations {payload['audit']['materializations']} < required "
            f"{args.min_materializations}"
        )
    if payload["audit"]["materialized_units"] < args.min_materialized_units:
        raise SystemExit(
            "materialized units "
            f"{payload['audit']['materialized_units']} < required "
            f"{args.min_materialized_units}"
        )
    if payload["audit"]["incremental_matched_tokens"] < args.min_incremental_matched_tokens:
        raise SystemExit(
            "incremental matched tokens "
            f"{payload['audit']['incremental_matched_tokens']} < required "
            f"{args.min_incremental_matched_tokens}"
        )
    if payload["audit"]["rope_corrections"] < args.min_rope_corrections:
        raise SystemExit(
            f"rope corrections {payload['audit']['rope_corrections']} < required "
            f"{args.min_rope_corrections}"
        )


def run_validation(args: argparse.Namespace) -> list[SampleResult]:
    os.environ["SEMBLEND_ENABLED"] = "1"
    os.environ["SEMBLEND_TRTLLM_AUDIT_PATH"] = args.audit_path
    os.environ["SEMBLEND_TRTLLM_ENABLE_MATERIALIZATION"] = "1"
    os.environ["SEMBLEND_TRTLLM_MIN_MATCH_LENGTH"] = str(args.min_match_length)
    Path(args.audit_path).unlink(missing_ok=True)

    baseline = _build_llm(args, use_semblend=False)
    baseline_outputs = {}
    for sample in SAMPLES:
        baseline_outputs[sample.name] = _timed_generate(
            baseline,
            sample.query_prompt,
            args.max_tokens,
        )
    del baseline
    gc.collect()
    _empty_cuda_cache()

    semblend = _build_llm(args, use_semblend=True)
    results = []
    for sample in SAMPLES:
        _generate(semblend, sample.donor_prompt, args.max_tokens)
        baseline_latency, baseline_text = baseline_outputs[sample.name]
        semblend_latency, semblend_text = _timed_generate(
            semblend,
            sample.query_prompt,
            args.max_tokens,
        )

        speedup = baseline_latency / max(semblend_latency, 1e-9)
        results.append(
            SampleResult(
                name=sample.name,
                baseline_latency_s=baseline_latency,
                semblend_latency_s=semblend_latency,
                speedup=speedup,
                baseline_text=baseline_text,
                semblend_text=semblend_text,
                rouge1_f1=rouge1_f1(baseline_text, semblend_text),
                token_f1=reference_token_f1(semblend_text, sample.reference_terms),
                ppl_delta=_optional_ppl_delta(args, baseline_text, semblend_text),
                judge_score=_optional_judge_score(args, baseline_text, semblend_text),
            )
        )

    return results


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def _build_llm(args: argparse.Namespace, *, use_semblend: bool):
    from tensorrt_llm import LLM

    kwargs: dict[str, Any] = {
        "model": args.model,
        "backend": "pytorch",
        "cuda_graph_config": None,
    }
    if use_semblend:
        from tensorrt_llm.llmapi.llm_args import KvCacheConnectorConfig

        kwargs["kv_connector_config"] = KvCacheConnectorConfig(
            connector_module="semblend.integration.trtllm.connector",
            connector_scheduler_class="SemBlendKvConnectorScheduler",
            connector_worker_class="SemBlendKvConnectorWorker",
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


def rouge1_f1(a: str, b: str) -> float:
    a_tokens = _tokens(a)
    b_tokens = _tokens(b)
    if not a_tokens or not b_tokens:
        return 0.0
    overlap = _overlap(a_tokens, b_tokens)
    precision = overlap / len(b_tokens)
    recall = overlap / len(a_tokens)
    return 2 * precision * recall / max(precision + recall, 1e-9)


def reference_token_f1(text: str, reference_terms: tuple[str, ...]) -> float:
    text_tokens = set(_tokens(text))
    refs = set(_tokens(" ".join(reference_terms)))
    if not refs:
        return 1.0
    overlap = len(text_tokens & refs)
    precision = overlap / max(len(text_tokens), 1)
    recall = overlap / len(refs)
    return 2 * precision * recall / max(precision + recall, 1e-9)


def _tokens(text: str) -> list[str]:
    return [token.strip(".,:;!?()[]{}\"'").lower() for token in text.split() if token.strip()]


def _overlap(a_tokens: list[str], b_tokens: list[str]) -> int:
    remaining: dict[str, int] = {}
    for token in a_tokens:
        remaining[token] = remaining.get(token, 0) + 1
    overlap = 0
    for token in b_tokens:
        count = remaining.get(token, 0)
        if count:
            overlap += 1
            remaining[token] = count - 1
    return overlap


def _optional_ppl_delta(args: argparse.Namespace, baseline: str, semblend: str) -> float | None:
    if not args.ppl_model:
        return None
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.ppl_model)
        model = AutoModelForCausalLM.from_pretrained(args.ppl_model).cuda().eval()

        def score(text: str) -> float:
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                loss = model(**inputs, labels=inputs["input_ids"]).loss
            return float(torch.exp(loss).detach().cpu())

        return score(semblend) - score(baseline)
    except Exception:
        return None


def _optional_judge_score(args: argparse.Namespace, baseline: str, semblend: str) -> float | None:
    if not args.judge_command:
        return None
    payload = json.dumps({"baseline": baseline, "semblend": semblend})
    try:
        proc = subprocess.run(
            args.judge_command,
            input=payload,
            text=True,
            shell=True,
            check=True,
            capture_output=True,
        )
        return float(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return None


def _summarize_audit(path: str) -> dict[str, Any]:
    events = []
    audit_path = Path(path)
    if audit_path.exists():
        for line in audit_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(json.loads(line))
    return {
        "path": path,
        "events": len(events),
        "donor_registrations": sum(e.get("event") == "donor_registered" for e in events),
        "semantic_hits": sum(e.get("event") == "lookup" and e.get("found") for e in events),
        "materializations": sum(e.get("event") == "materialized" for e in events),
        "materialized_units": sum(int(e.get("materialized", 0)) for e in events),
        "engine_boundaries": sum(e.get("event") == "engine_blend_boundary" for e in events),
        "incremental_matched_tokens": sum(
            int(e.get("num_new_matched_tokens", 0))
            for e in events
            if e.get("event") == "lookup"
        ),
        "rope_corrections": sum(
            e.get("event") == "materialized" and e.get("requires_rope_correction")
            for e in events
        ),
        "rejection_reasons": sorted(
            {
                e.get("rejection_reason")
                for e in events
                if e.get("event") == "lookup" and e.get("rejection_reason")
            }
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="results/trtllm-semblend.json")
    parser.add_argument("--audit-path", default="results/trtllm-semblend-audit.jsonl")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--min-match-length", type=int, default=64)
    parser.add_argument("--min-speedup", type=float, default=1.05)
    parser.add_argument("--min-rouge1", type=float, default=0.35)
    parser.add_argument("--min-token-f1", type=float, default=0.10)
    parser.add_argument("--min-donor-registrations", type=int, default=1)
    parser.add_argument("--min-semantic-hits", type=int, default=1)
    parser.add_argument("--min-materializations", type=int, default=1)
    parser.add_argument("--min-materialized-units", type=int, default=1)
    parser.add_argument("--min-incremental-matched-tokens", type=int, default=1)
    parser.add_argument("--min-rope-corrections", type=int, default=1)
    parser.add_argument("--ppl-model", default="")
    parser.add_argument(
        "--judge-command",
        default="",
        help="Optional command that reads JSON on stdin and prints a numeric LLM-judge score.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()

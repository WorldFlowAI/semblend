#!/usr/bin/env python3
"""KV Blending Experiment: alpha-blend donor KV with partial target computation.

Instead of correcting donor KV (which failed — affine assumption doesn't hold),
blend donor KV with freshly computed target KV using a per-layer gate:

    K_final[l] = alpha[l] * K_target + (1-alpha[l]) * K_donor
    V_final[l] = alpha[l] * V_target + (1-alpha[l]) * V_donor

Where alpha[l] comes from the bathtub curve (high deviation → more target,
low deviation → more donor).

The key question: does blending produce better PPL than either pure donor
injection or pure cold prefill, while being faster than cold?

Three conditions + two alpha schedules:
  A) Cold prefill (ground truth baseline)
  B) Pure donor injection (no blending — expected to degrade)
  C) Blended: alpha from bathtub curve (high alpha for early/late layers)
  D) Blended: alpha=0.5 uniform (equal mix everywhere)

The "cost" is that blending requires BOTH donor KV AND target KV, meaning
we do a full forward pass. BUT — the target forward pass with donor KV
already injected as past_key_values is much cheaper (only new tokens
need Q computation, cached KV provides K,V). So the speedup comes from
not recomputing K,V for the donor-covered positions.

Usage (run inside GPU pod):
    python kv_blending_experiment.py --n-pairs 10
"""
from __future__ import annotations

import argparse
import math
import statistics
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct-AWQ"
FALLBACK_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

INSTRUCTION_PAIRS = [
    (
        "You are a helpful assistant. Answer the following question based on the context provided below.",
        "Based on the text below, please answer:",
    ),
    (
        "Read the document carefully and answer the question that follows.",
        "Given the following document, respond to the query.",
    ),
    (
        "Using the information in the text below, provide a concise answer to the question.",
        "Refer to the passage and provide your answer.",
    ),
    (
        "You are an expert reader. Study the context and respond to the question accurately.",
        "Answer this question using the context provided.",
    ),
    (
        "Analyze the following passage and answer the question at the end.",
        "Read and answer:",
    ),
]

PROMPT_TEMPLATE = (
    "<|im_start|>system\n{instruction}<|im_end|>\n"
    "<|im_start|>user\n"
    "Context:\n{context}\n\n"
    "Question: {question}\nAnswer:<|im_end|>\n"
    "<|im_start|>assistant\n"
)


@dataclass
class BlendResult:
    sample_id: str
    question: str
    reference: str
    # Texts
    cold_text: str
    donor_text: str
    bathtub_blend_text: str
    uniform_blend_text: str
    # PPL
    cold_ppl: float
    donor_ppl: float
    bathtub_blend_ppl: float
    uniform_blend_ppl: float
    # QA
    cold_match: bool
    donor_match: bool
    bathtub_match: bool
    uniform_match: bool
    # Timing
    cold_ms: float
    blend_ms: float
    # Per-layer cosine sim between blended and target KV
    bathtub_layer_sims: list[float] = field(default_factory=list)


def load_model(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_name}...")
    t0 = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype=torch.float16,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"Failed ({e}), trying {FALLBACK_MODEL}...")
        tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            FALLBACK_MODEL, device_map="auto", torch_dtype=torch.float16,
            trust_remote_code=True,
        )

    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"Loaded in {time.monotonic()-t0:.1f}s: {model.config._name_or_path}, "
          f"{n_layers}L, {model.config.hidden_size}H")
    return model, tokenizer


def get_kv(model, tokenizer, text: str):
    """Forward pass → extract per-layer (K, V) tuples."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs, use_cache=True, return_dict=True)
    kv = out.past_key_values
    # Extract as list of (K, V) tensors
    layers = []
    for i in range(len(kv)):
        layers.append((kv[i][0].clone(), kv[i][1].clone()))
    return layers, out.logits, inputs["input_ids"]


def compute_bathtub_alpha(num_layers: int) -> list[float]:
    """Compute per-layer blending alpha from bathtub curve.

    alpha=1.0 → use target KV (recompute)
    alpha=0.0 → use donor KV (inject)

    Early/late layers get alpha close to 1.0 (high deviation → recompute).
    Middle layers get alpha close to 0.0 (low deviation → inject donor).
    """
    alphas = []
    for li in range(num_layers):
        # Bathtub: sigma = base + early_exp + late_exp
        big_ll = num_layers - 1
        sigma_base = 0.12
        sigma_e = 0.35 * math.exp(-li / 3.0)
        sigma_l = 0.20 * math.exp(-(big_ll - li) / 4.0)
        sigma = min(sigma_base + sigma_e + sigma_l, 1.0)
        # Map sigma to alpha: high deviation → high alpha (use target)
        # Normalize: sigma ranges ~0.12-0.55, map to alpha 0.1-0.9
        alpha = min(0.9, max(0.1, (sigma - 0.10) / 0.50))
        alphas.append(alpha)
    return alphas


def blend_kv(
    donor_kv: list[tuple[torch.Tensor, torch.Tensor]],
    target_kv: list[tuple[torch.Tensor, torch.Tensor]],
    alphas: list[float],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Blend donor and target KV per layer using alpha schedule.

    blended[l] = alpha[l] * target[l] + (1-alpha[l]) * donor[l]
    """
    blended = []
    for li, (dk_dv, tk_tv) in enumerate(zip(donor_kv, target_kv)):
        dk, dv = dk_dv
        tk, tv = tk_tv
        a = alphas[li]
        min_len = min(dk.shape[2], tk.shape[2])

        # Blend only the overlapping region
        bk = a * tk[:, :, :min_len, :].float() + (1 - a) * dk[:, :, :min_len, :].float()
        bv = a * tv[:, :, :min_len, :].float() + (1 - a) * dv[:, :, :min_len, :].float()

        blended.append((bk.to(dk.dtype), bv.to(dv.dtype)))
    return blended


def generate_from_kv(
    model, tokenizer, input_ids, past_kv, max_new_tokens: int = 64,
) -> tuple[str, float, float]:
    """Generate using pre-computed KV cache via manual autoregressive loop.

    Uses direct forward passes instead of model.generate() to avoid
    DynamicCache API compatibility issues with newer transformers.
    """
    t0 = time.monotonic()

    from transformers import DynamicCache

    cache_len = past_kv[0][0].shape[2]
    # Start from the last token in the cached sequence
    current_token = input_ids[:, cache_len - 1:cache_len]

    # Build DynamicCache from our KV tuples
    past = DynamicCache()
    for layer_idx_kv, (k_kv, v_kv) in enumerate(past_kv):
        past.update(k_kv, v_kv, layer_idx_kv)

    generated_ids = []
    total_lp = 0.0

    with torch.no_grad():
        for step in range(max_new_tokens):
            outputs = model(
                input_ids=current_token,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )

            logits = outputs.logits[:, -1, :]  # [1, vocab_size]
            log_probs = F.log_softmax(logits.float(), dim=-1)

            # Greedy selection
            next_token = logits.argmax(dim=-1, keepdim=True)  # [1, 1]
            token_lp = log_probs[0, next_token[0, 0]].item()
            total_lp += token_lp

            generated_ids.append(next_token[0, 0].item())
            past = outputs.past_key_values
            current_token = next_token

            # Stop on EOS
            if next_token[0, 0].item() == tokenizer.eos_token_id:
                break

    time_ms = (time.monotonic() - t0) * 1000

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    n = len(generated_ids)
    ppl = math.exp(-total_lp / max(n, 1)) if n > 0 else float('inf')

    return text, ppl, time_ms


def cold_generate(model, tokenizer, prompt: str, max_new_tokens: int = 64):
    """Standard cold generation via autoregressive loop. Returns (text, ppl, time_ms)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    t0 = time.monotonic()

    # First forward: process entire prompt, get KV cache
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True, return_dict=True)

    past = outputs.past_key_values
    logits = outputs.logits[:, -1, :]
    log_probs = F.log_softmax(logits.float(), dim=-1)
    next_token = logits.argmax(dim=-1, keepdim=True)
    total_lp = log_probs[0, next_token[0, 0]].item()
    generated_ids = [next_token[0, 0].item()]
    current_token = next_token

    # Autoregressive loop
    with torch.no_grad():
        for step in range(max_new_tokens - 1):
            outputs = model(
                input_ids=current_token,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            logits = outputs.logits[:, -1, :]
            log_probs = F.log_softmax(logits.float(), dim=-1)
            next_token = logits.argmax(dim=-1, keepdim=True)
            total_lp += log_probs[0, next_token[0, 0]].item()
            generated_ids.append(next_token[0, 0].item())
            past = outputs.past_key_values
            current_token = next_token

            if next_token[0, 0].item() == tokenizer.eos_token_id:
                break

    time_ms = (time.monotonic() - t0) * 1000
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    n = len(generated_ids)
    ppl = math.exp(-total_lp / max(n, 1)) if n > 0 else float('inf')
    return text, ppl, time_ms


def check_answer(text: str, reference: str) -> bool:
    if not reference or not text:
        return False
    return reference.lower().strip() in text.lower().strip()


def load_triviaqa(n: int):
    from datasets import load_dataset
    print(f"Loading {n} TriviaQA pairs...")
    ds = load_dataset("trivia_qa", "rc", split="validation")
    pairs = []
    for row in ds:
        if len(pairs) >= n:
            break
        wiki = row.get("entity_pages", {}).get("wiki_context", [])
        if not wiki:
            continue
        ctx = max(wiki, key=len)
        if len(ctx) < 2000:
            continue
        ctx = ctx[:8000]
        q = row.get("question", "")
        ans = row.get("answer", {})
        a = ans.get("value", "") if isinstance(ans, dict) else str(ans)
        if q and a:
            pairs.append({"context": ctx, "question": q, "answer": a})
    print(f"Loaded {len(pairs)}")
    return pairs


def run(n_pairs: int = 10, max_tokens: int = 64):
    model, tokenizer = load_model(MODEL_NAME)
    n_layers = model.config.num_hidden_layers

    # Compute alpha schedules
    bathtub_alphas = compute_bathtub_alpha(n_layers)
    uniform_alphas = [0.5] * n_layers

    print("\nBathtub alpha schedule (first/mid/last):")
    print(f"  L0={bathtub_alphas[0]:.2f}, L{n_layers//2}={bathtub_alphas[n_layers//2]:.2f}, "
          f"L{n_layers-1}={bathtub_alphas[-1]:.2f}")

    pairs = load_triviaqa(n_pairs)
    import random
    rng = random.Random(42)

    results: list[BlendResult] = []

    for idx, pair in enumerate(pairs):
        donor_instr, target_instr = INSTRUCTION_PAIRS[rng.randint(0, len(INSTRUCTION_PAIRS) - 1)]

        donor_prompt = PROMPT_TEMPLATE.format(
            instruction=donor_instr, context=pair["context"], question=pair["question"],
        )
        target_prompt = PROMPT_TEMPLATE.format(
            instruction=target_instr, context=pair["context"], question=pair["question"],
        )

        print(f"\n--- Sample {idx+1}/{len(pairs)}: {pair['question'][:60]}... ---")
        print(f"  Answer: {pair['answer']}")

        # Extract KV for both
        donor_kv, _, _ = get_kv(model, tokenizer, donor_prompt)
        target_kv, _, target_ids = get_kv(model, tokenizer, target_prompt)

        # Condition A: Cold prefill
        cold_text, cold_ppl, cold_ms = cold_generate(model, tokenizer, target_prompt, max_tokens)

        # Condition B: Pure donor KV injection
        try:
            donor_text, donor_ppl, _ = generate_from_kv(
                model, tokenizer, target_ids, donor_kv, max_tokens,
            )
        except Exception as e:
            print(f"  Donor inject failed: {e}")
            donor_text, donor_ppl = "(failed)", 999.0

        # Condition C: Bathtub-blended KV
        blended_bt = blend_kv(donor_kv, target_kv, bathtub_alphas)
        try:
            bt_text, bt_ppl, blend_ms = generate_from_kv(
                model, tokenizer, target_ids, blended_bt, max_tokens,
            )
        except Exception as e:
            print(f"  Bathtub blend failed: {e}")
            bt_text, bt_ppl, blend_ms = "(failed)", 999.0, 0.0

        # Condition D: Uniform blend (alpha=0.5)
        blended_uni = blend_kv(donor_kv, target_kv, uniform_alphas)
        try:
            uni_text, uni_ppl, _ = generate_from_kv(
                model, tokenizer, target_ids, blended_uni, max_tokens,
            )
        except Exception as e:
            print(f"  Uniform blend failed: {e}")
            uni_text, uni_ppl = "(failed)", 999.0

        # Per-layer similarity between bathtub-blended and target KV
        layer_sims = []
        for li2 in range(n_layers):
            bk = blended_bt[li2][0]
            tk = target_kv[li2][0]
            ml = min(bk.shape[2], tk.shape[2])
            sim = F.cosine_similarity(
                bk[0, :, :ml, :].reshape(-1, bk.shape[-1]).float(),
                tk[0, :, :ml, :].reshape(-1, tk.shape[-1]).float(),
                dim=-1,
            ).mean().item()
            layer_sims.append(sim)

        cold_match = check_answer(cold_text, pair["answer"])
        donor_match = check_answer(donor_text, pair["answer"])
        bt_match = check_answer(bt_text, pair["answer"])
        uni_match = check_answer(uni_text, pair["answer"])

        print(f"  Cold:     PPL={cold_ppl:.2f}  QA={'Y' if cold_match else 'N'}  {cold_text[:50]}...")
        print(f"  Donor:    PPL={donor_ppl:.2f}  QA={'Y' if donor_match else 'N'}  {donor_text[:50]}...")
        print(f"  Bathtub:  PPL={bt_ppl:.2f}  QA={'Y' if bt_match else 'N'}  {bt_text[:50]}...")
        print(f"  Uniform:  PPL={uni_ppl:.2f}  QA={'Y' if uni_match else 'N'}  {uni_text[:50]}...")

        results.append(BlendResult(
            sample_id=f"triviaqa_{idx}",
            question=pair["question"],
            reference=pair["answer"],
            cold_text=cold_text, donor_text=donor_text,
            bathtub_blend_text=bt_text, uniform_blend_text=uni_text,
            cold_ppl=cold_ppl, donor_ppl=donor_ppl,
            bathtub_blend_ppl=bt_ppl, uniform_blend_ppl=uni_ppl,
            cold_match=cold_match, donor_match=donor_match,
            bathtub_match=bt_match, uniform_match=uni_match,
            cold_ms=cold_ms, blend_ms=blend_ms,
            bathtub_layer_sims=layer_sims,
        ))

        del donor_kv, target_kv, blended_bt, blended_uni
        torch.cuda.empty_cache()

    # AGGREGATE
    print("\n" + "=" * 70)
    print("KV BLENDING RESULTS")
    print("=" * 70)

    n = len(results)
    valid = [r for r in results if r.cold_ppl < 100 and r.bathtub_blend_ppl < 100]

    def safe_mean(vals):
        v = [x for x in vals if x < 100]
        return statistics.mean(v) if v else 0

    print(f"\nSamples: {n}")
    print(f"\n{'Condition':<18} {'Mean PPL':>10} {'QA Match':>10} {'Rate':>8}")
    print("-" * 50)
    print(f"{'Cold':<18} {safe_mean([r.cold_ppl for r in results]):>10.3f} "
          f"{sum(r.cold_match for r in results):>10} {sum(r.cold_match for r in results)/n:>7.0%}")
    print(f"{'Pure donor':<18} {safe_mean([r.donor_ppl for r in results]):>10.3f} "
          f"{sum(r.donor_match for r in results):>10} {sum(r.donor_match for r in results)/n:>7.0%}")
    print(f"{'Bathtub blend':<18} {safe_mean([r.bathtub_blend_ppl for r in results]):>10.3f} "
          f"{sum(r.bathtub_match for r in results):>10} {sum(r.bathtub_match for r in results)/n:>7.0%}")
    print(f"{'Uniform blend':<18} {safe_mean([r.uniform_blend_ppl for r in results]):>10.3f} "
          f"{sum(r.uniform_match for r in results):>10} {sum(r.uniform_match for r in results)/n:>7.0%}")

    # PPL ratios
    if valid:
        donor_ratios = [r.donor_ppl / r.cold_ppl for r in valid if r.donor_ppl < 100]
        bt_ratios = [r.bathtub_blend_ppl / r.cold_ppl for r in valid]
        uni_ratios = [r.uniform_blend_ppl / r.cold_ppl for r in valid if r.uniform_blend_ppl < 100]

        print("\nPPL ratios vs cold:")
        if donor_ratios:
            print(f"  Pure donor:    {statistics.mean(donor_ratios):.4f}")
        if bt_ratios:
            print(f"  Bathtub blend: {statistics.mean(bt_ratios):.4f}")
        if uni_ratios:
            print(f"  Uniform blend: {statistics.mean(uni_ratios):.4f}")

    # Timing
    cold_times = [r.cold_ms for r in results]
    blend_times = [r.blend_ms for r in results if r.blend_ms > 0]
    if cold_times and blend_times:
        print("\nTiming:")
        print(f"  Cold:  {statistics.mean(cold_times):.0f}ms")
        print(f"  Blend: {statistics.mean(blend_times):.0f}ms")

    # Verdict
    print("\n--- VERDICT ---")
    if valid and bt_ratios:
        bt_mean = statistics.mean(bt_ratios)
        cold_qa = sum(r.cold_match for r in results) / n
        bt_qa = sum(r.bathtub_match for r in results) / n
        print(f"Bathtub PPL ratio: {bt_mean:.4f} {'PASS' if bt_mean <= 1.065 else 'FAIL'}")
        print(f"QA degradation: {cold_qa - bt_qa:+.0%} {'PASS' if (cold_qa - bt_qa) <= 0.05 else 'FAIL'}")
        if bt_mean <= 1.065 and (cold_qa - bt_qa) <= 0.05:
            print("KV BLENDING: FEASIBLE")
        else:
            print("KV BLENDING: NEEDS WORK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-pairs", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()
    run(n_pairs=args.n_pairs, max_tokens=args.max_tokens)

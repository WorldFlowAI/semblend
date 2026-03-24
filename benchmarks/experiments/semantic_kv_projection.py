#!/usr/bin/env python3
"""Semantic KV Projection — prototype experiment.

Novel technique: transfer KV cache between semantically similar but lexically
different text by estimating a per-layer affine correction matrix from probe
tokens.

Experiment:
  1. Load Qwen2.5-7B-AWQ, generate KV for paraphrased passage pairs
  2. Measure per-layer KV cosine similarity (validate bathtub assumption)
  3. Estimate affine correction W per layer from 32 probe tokens
  4. Inject corrected donor KV, measure output PPL vs cold baseline
  5. Verify QA accuracy on TriviaQA with ground truth answers

Three conditions compared:
  A) Cold prefill (ground truth)
  B) Verbatim donor KV injection (no correction — expected to fail)
  C) Probe-corrected donor KV injection (the novel approach)

Usage (run inside the GPU pod):
    python semantic_kv_projection.py --n-pairs 20

Requires: torch, transformers, awq (already in vLLM image)
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct-AWQ"

# Paraphrase instruction pairs — same meaning, different wording
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
class LayerDeviation:
    layer_idx: int
    k_cosine_sim: float  # mean cosine similarity of K vectors
    v_cosine_sim: float  # mean cosine similarity of V vectors
    k_l2_dist: float     # mean L2 distance of K vectors
    v_l2_dist: float     # mean L2 distance of V vectors


@dataclass
class CorrectionResult:
    layer_idx: int
    before_k_sim: float   # K similarity before correction
    after_k_sim: float    # K similarity after affine correction
    before_v_sim: float
    after_v_sim: float
    correction_residual: float  # how well W fits the probe data


@dataclass
class SampleResult:
    sample_id: str
    question: str
    reference_answer: str
    # Outputs
    cold_text: str
    verbatim_text: str
    corrected_text: str
    # PPL
    cold_ppl: float
    verbatim_ppl: float
    corrected_ppl: float
    # QA
    cold_answer_match: bool
    verbatim_answer_match: bool
    corrected_answer_match: bool
    # Timing
    cold_time_ms: float
    corrected_time_ms: float
    probe_time_ms: float
    # Layer analysis
    layer_deviations: list[LayerDeviation] = field(default_factory=list)
    corrections: list[CorrectionResult] = field(default_factory=list)


def load_model_and_tokenizer(model_name: str):
    """Load the model with KV cache access.

    Tries AWQ quantized loading first, falls back to float16 with a
    smaller non-AWQ model if AWQ dependencies aren't available.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_name}...")
    t0 = time.monotonic()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    try:
        # Try loading with AWQ support
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
    except (ImportError, Exception) as e:
        print(f"AWQ loading failed ({e}), trying Qwen2.5-1.5B-Instruct instead...")
        # Fall back to a smaller non-quantized model
        fallback = "Qwen/Qwen2.5-1.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(fallback, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            fallback,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

    model.eval()

    print(f"Model loaded in {time.monotonic() - t0:.1f}s")
    print(f"Model: {model.config._name_or_path}")
    print(f"Layers: {model.config.num_hidden_layers}")
    print(f"Hidden: {model.config.hidden_size}")
    print(f"Heads: {model.config.num_attention_heads}")

    return model, tokenizer


def extract_kv(model, tokenizer, text: str) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    """Forward pass to extract per-layer K,V tensors.

    Returns:
        (keys, values, logits) where keys[i] and values[i] are
        [1, n_heads, seq_len, head_dim] tensors for layer i.
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=False,
            use_cache=True,
            return_dict=True,
        )

    past_kv = outputs.past_key_values
    keys = [past_kv[i][0] for i in range(len(past_kv))]
    values = [past_kv[i][1] for i in range(len(past_kv))]

    return keys, values, outputs.logits


def compute_layer_deviations(
    donor_keys: list[torch.Tensor],
    donor_values: list[torch.Tensor],
    target_keys: list[torch.Tensor],
    target_values: list[torch.Tensor],
) -> list[LayerDeviation]:
    """Compute per-layer KV deviation between donor and target.

    Uses position-aligned comparison: for each position that exists in
    both sequences, compare the K,V vectors. This handles different
    sequence lengths by comparing only the overlapping prefix.
    """
    n_layers = len(donor_keys)
    deviations = []

    for layer_idx in range(n_layers):
        dk = donor_keys[layer_idx]   # [1, n_heads, donor_len, head_dim]
        tk = target_keys[layer_idx]  # [1, n_heads, target_len, head_dim]
        dv = donor_values[layer_idx]
        tv = target_values[layer_idx]

        # Compare overlapping positions
        min_len = min(dk.shape[2], tk.shape[2])
        if min_len == 0:
            deviations.append(LayerDeviation(layer_idx, 0, 0, 0, 0))
            continue

        dk_slice = dk[0, :, :min_len, :].reshape(-1, dk.shape[-1])  # [n_heads*min_len, head_dim]
        tk_slice = tk[0, :, :min_len, :].reshape(-1, tk.shape[-1])
        dv_slice = dv[0, :, :min_len, :].reshape(-1, dv.shape[-1])
        tv_slice = tv[0, :, :min_len, :].reshape(-1, tv.shape[-1])

        # Cosine similarity
        k_cos = F.cosine_similarity(dk_slice.float(), tk_slice.float(), dim=-1).mean().item()
        v_cos = F.cosine_similarity(dv_slice.float(), tv_slice.float(), dim=-1).mean().item()

        # L2 distance
        k_l2 = (dk_slice.float() - tk_slice.float()).norm(dim=-1).mean().item()
        v_l2 = (dv_slice.float() - tv_slice.float()).norm(dim=-1).mean().item()

        deviations.append(LayerDeviation(layer_idx, k_cos, v_cos, k_l2, v_l2))

    return deviations


def estimate_affine_correction(
    donor_kv: torch.Tensor,  # [n_heads, probe_len, head_dim]
    target_kv: torch.Tensor,  # [n_heads, probe_len, head_dim]
) -> torch.Tensor:
    """Estimate affine correction matrix W such that donor @ W ≈ target.

    Uses least-squares solution per attention head, then averages across heads.
    W is [head_dim, head_dim].

    The idea: if middle-layer KV deviation is approximately linear,
    then W = (D^T D)^{-1} D^T T gives the best linear mapping.
    """
    n_heads, probe_len, head_dim = donor_kv.shape

    # Solve per head then average (more stable than global solve)
    W_sum = torch.zeros(head_dim, head_dim, device=donor_kv.device, dtype=torch.float32)

    for h in range(n_heads):
        D = donor_kv[h].float()  # [probe_len, head_dim]
        T = target_kv[h].float()  # [probe_len, head_dim]

        # Least-squares: W = (D^T D)^{-1} D^T T
        # Use torch.linalg.lstsq for numerical stability
        result = torch.linalg.lstsq(D, T)
        W_h = result.solution  # [head_dim, head_dim]
        W_sum += W_h

    W = W_sum / n_heads
    return W


def apply_correction(
    donor_kv: torch.Tensor,  # [1, n_heads, seq_len, head_dim]
    W: torch.Tensor,         # [head_dim, head_dim]
) -> torch.Tensor:
    """Apply affine correction to donor KV: corrected = donor @ W."""
    # W is float32, donor_kv may be float16
    orig_dtype = donor_kv.dtype
    corrected = torch.matmul(donor_kv.float(), W.unsqueeze(0).unsqueeze(0))
    return corrected.to(orig_dtype)


def generate_with_kv(
    model, tokenizer, prompt_text: str,
    injected_keys: list[torch.Tensor] | None = None,
    injected_values: list[torch.Tensor] | None = None,
    max_new_tokens: int = 128,
    inject_layers: set[int] | None = None,
) -> tuple[str, float, float]:
    """Generate text, optionally injecting KV at specified layers.

    For layers in inject_layers, uses the provided KV instead of computing.
    For other layers, computes normally (recompute).

    Returns: (generated_text, ppl, time_ms)
    """
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]

    t0 = time.monotonic()

    if injected_keys is not None and inject_layers:
        # Build past_key_values with injected KV for selected layers
        # For layers NOT in inject_layers, we pass None (recompute)
        # Unfortunately, HF transformers doesn't support per-layer injection
        # directly — we need to use the full past_key_values tuple.
        #
        # Strategy: run a forward pass with the injected KV as past_key_values,
        # then generate from there. The model treats injected KV as "already
        # computed" and only processes the remaining tokens.
        #
        # For selective layer injection, we'd need model surgery. For the
        # prototype, inject ALL middle layers and recompute early/late by
        # NOT including their KV (setting them to zero-length tensors).
        #
        # Actually, HF past_key_values requires all layers. So we inject
        # for middle layers and use the cold-computed KV for early/late.

        # First, do a cold forward pass to get ground-truth KV for all layers
        with torch.no_grad():
            cold_out = model(**inputs, use_cache=True, return_dict=True)
        cold_kv = cold_out.past_key_values

        # Build hybrid past_key_values: cold for early/late, injected for middle
        n_layers = len(cold_kv)
        hybrid_kv = []
        for i in range(n_layers):
            if i in inject_layers and i < len(injected_keys):
                # Use injected (corrected donor) KV for this layer
                # Trim to match the target sequence length
                target_len = cold_kv[i][0].shape[2]
                ik = injected_keys[i][:, :, :target_len, :]
                iv = injected_values[i][:, :, :target_len, :]
                hybrid_kv.append((ik, iv))
            else:
                # Use cold-computed KV for early/late layers
                hybrid_kv.append(cold_kv[i])

        hybrid_kv = tuple(hybrid_kv)

        # Generate from the hybrid KV
        with torch.no_grad():
            gen_ids = model.generate(
                input_ids,
                past_key_values=hybrid_kv,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                return_dict_in_generate=True,
                output_scores=True,
            )

        time_ms = (time.monotonic() - t0) * 1000
        generated_ids = gen_ids.sequences[0][input_ids.shape[1]:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Compute PPL from scores
        scores = gen_ids.scores  # list of [1, vocab_size] logits
        total_log_prob = 0.0
        n_tokens = 0
        for i, score in enumerate(scores):
            if i >= len(generated_ids):
                break
            log_probs = F.log_softmax(score[0].float(), dim=-1)
            token_log_prob = log_probs[generated_ids[i]].item()
            total_log_prob += token_log_prob
            n_tokens += 1

        ppl = math.exp(-total_log_prob / max(n_tokens, 1)) if n_tokens > 0 else float('inf')

    else:
        # Cold prefill — standard generation
        with torch.no_grad():
            gen_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                return_dict_in_generate=True,
                output_scores=True,
            )

        time_ms = (time.monotonic() - t0) * 1000
        generated_ids = gen_ids.sequences[0][input_ids.shape[1]:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        scores = gen_ids.scores
        total_log_prob = 0.0
        n_tokens = 0
        for i, score in enumerate(scores):
            if i >= len(generated_ids):
                break
            log_probs = F.log_softmax(score[0].float(), dim=-1)
            token_log_prob = log_probs[generated_ids[i]].item()
            total_log_prob += token_log_prob
            n_tokens += 1

        ppl = math.exp(-total_log_prob / max(n_tokens, 1)) if n_tokens > 0 else float('inf')

    return text, ppl, time_ms


def check_answer(generated: str, reference: str) -> bool:
    """Check if generated text contains the reference answer."""
    if not reference or not generated:
        return False
    return reference.lower().strip() in generated.lower().strip()


def load_triviaqa_pairs(n: int) -> list[dict]:
    """Load TriviaQA with context + question + answer."""
    from datasets import load_dataset

    print(f"Loading {n} TriviaQA pairs...")
    ds = load_dataset("trivia_qa", "rc", split="validation")

    pairs = []
    for row in ds:
        if len(pairs) >= n:
            break
        entity_pages = row.get("entity_pages", {})
        wiki_contexts = entity_pages.get("wiki_context", [])
        if not wiki_contexts:
            continue
        context = max(wiki_contexts, key=len)
        if len(context) < 2000:
            continue
        # Truncate to ~2K tokens to fit in memory
        context = context[:8000]

        question = row.get("question", "")
        answer = row.get("answer", {})
        ans_text = ""
        if isinstance(answer, dict):
            aliases = answer.get("aliases", [])
            ans_text = answer.get("value", aliases[0] if aliases else "")
        else:
            ans_text = str(answer)

        if not question or not ans_text:
            continue

        pairs.append({
            "context": context,
            "question": question,
            "answer": ans_text,
        })

    print(f"Loaded {len(pairs)} pairs")
    return pairs


def run_experiment(n_pairs: int = 20, max_new_tokens: int = 64) -> dict:
    """Run the full Semantic KV Projection experiment."""
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    num_layers = model.config.num_hidden_layers

    # Define injection layers (middle layers only)
    early_layers = set(range(0, 4))            # 0-3: always recompute
    late_layers = set(range(num_layers - 3, num_layers))  # 25-27: always recompute
    inject_layers = set(range(4, num_layers - 3))  # 4-24: inject with correction

    print(f"\nLayer plan: recompute early {sorted(early_layers)}, "
          f"inject middle {min(inject_layers)}-{max(inject_layers)}, "
          f"recompute late {sorted(late_layers)}")

    pairs = load_triviaqa_pairs(n_pairs)
    import random
    rng = random.Random(42)

    results: list[SampleResult] = []

    for idx, pair in enumerate(pairs):
        # Pick a random instruction pair
        donor_instr, target_instr = INSTRUCTION_PAIRS[rng.randint(0, len(INSTRUCTION_PAIRS) - 1)]

        donor_prompt = PROMPT_TEMPLATE.format(
            instruction=donor_instr,
            context=pair["context"],
            question=pair["question"],
        )
        target_prompt = PROMPT_TEMPLATE.format(
            instruction=target_instr,
            context=pair["context"],
            question=pair["question"],
        )

        print(f"\n--- Sample {idx + 1}/{len(pairs)} ---")
        print(f"  Q: {pair['question'][:80]}...")
        print(f"  A: {pair['answer']}")

        # Step 1: Extract KV for both prompts
        donor_keys, donor_values, _ = extract_kv(model, tokenizer, donor_prompt)
        target_keys, target_values, _ = extract_kv(model, tokenizer, target_prompt)

        # Step 2: Measure per-layer deviation
        deviations = compute_layer_deviations(
            donor_keys, donor_values, target_keys, target_values,
        )

        # Print bathtub curve
        if idx == 0:
            print("\n  Per-layer KV cosine similarity (bathtub curve):")
            print(f"  {'Layer':>5} {'K_cos':>8} {'V_cos':>8} {'K_L2':>8} {'V_L2':>8}")
            for d in deviations:
                marker = " *" if d.layer_idx in inject_layers else ""
                print(f"  {d.layer_idx:>5} {d.k_cosine_sim:>8.4f} {d.v_cosine_sim:>8.4f} "
                      f"{d.k_l2_dist:>8.4f} {d.v_l2_dist:>8.4f}{marker}")

        # Step 3: Probe-then-inject
        # Use first N_PROBE tokens of the TARGET as probe
        N_PROBE = 32
        target_len = target_keys[0].shape[2]
        donor_len = donor_keys[0].shape[2]

        if target_len < N_PROBE + 10 or donor_len < N_PROBE + 10:
            print(f"  SKIP: sequences too short (donor={donor_len}, target={target_len})")
            continue

        t_probe = time.monotonic()
        corrections = []
        corrected_keys = []
        corrected_values = []

        for layer_idx in range(num_layers):
            dk = donor_keys[layer_idx]   # [1, n_heads, donor_len, head_dim]
            tk = target_keys[layer_idx]
            dv = donor_values[layer_idx]
            tv = target_values[layer_idx]

            if layer_idx in inject_layers:
                # Estimate W from probe tokens
                dk_probe = dk[0, :, :N_PROBE, :]  # [n_heads, N_PROBE, head_dim]
                tk_probe = tk[0, :, :N_PROBE, :]

                W_k = estimate_affine_correction(dk_probe, tk_probe)

                dv_probe = dv[0, :, :N_PROBE, :]
                tv_probe = tv[0, :, :N_PROBE, :]
                W_v = estimate_affine_correction(dv_probe, tv_probe)

                # Apply correction to full donor KV
                ck = apply_correction(dk, W_k)
                cv = apply_correction(dv, W_v)
                corrected_keys.append(ck)
                corrected_values.append(cv)

                # Measure improvement
                min_len = min(dk.shape[2], tk.shape[2])
                before_k = F.cosine_similarity(
                    dk[0, :, :min_len, :].reshape(-1, dk.shape[-1]).float(),
                    tk[0, :, :min_len, :].reshape(-1, tk.shape[-1]).float(),
                    dim=-1,
                ).mean().item()
                after_k = F.cosine_similarity(
                    ck[0, :, :min_len, :].reshape(-1, dk.shape[-1]).float(),
                    tk[0, :, :min_len, :].reshape(-1, tk.shape[-1]).float(),
                    dim=-1,
                ).mean().item()
                before_v = F.cosine_similarity(
                    dv[0, :, :min_len, :].reshape(-1, dv.shape[-1]).float(),
                    tv[0, :, :min_len, :].reshape(-1, tv.shape[-1]).float(),
                    dim=-1,
                ).mean().item()
                after_v = F.cosine_similarity(
                    cv[0, :, :min_len, :].reshape(-1, dv.shape[-1]).float(),
                    tv[0, :, :min_len, :].reshape(-1, tv.shape[-1]).float(),
                    dim=-1,
                ).mean().item()

                # Residual: how well does W fit on non-probe positions?
                post_probe = min(min_len, N_PROBE + 64)
                if post_probe > N_PROBE:
                    ck_check = ck[0, :, N_PROBE:post_probe, :].reshape(-1, dk.shape[-1]).float()
                    tk_check = tk[0, :, N_PROBE:post_probe, :].reshape(-1, tk.shape[-1]).float()
                    residual = (ck_check - tk_check).norm(dim=-1).mean().item()
                else:
                    residual = 0.0

                corrections.append(CorrectionResult(
                    layer_idx, before_k, after_k, before_v, after_v, residual,
                ))
            else:
                # Early/late layers: use target's own KV (recompute)
                corrected_keys.append(tk)
                corrected_values.append(tv)

        probe_time_ms = (time.monotonic() - t_probe) * 1000

        # Print correction effectiveness for first sample
        if idx == 0 and corrections:
            print(f"\n  Affine correction effectiveness (probe={N_PROBE} tokens):")
            print(f"  {'Layer':>5} {'K_before':>10} {'K_after':>10} {'V_before':>10} {'V_after':>10} {'Residual':>10}")
            for c in corrections[:10]:
                print(f"  {c.layer_idx:>5} {c.before_k_sim:>10.4f} {c.after_k_sim:>10.4f} "
                      f"{c.before_v_sim:>10.4f} {c.after_v_sim:>10.4f} {c.correction_residual:>10.4f}")

        # Step 4: Generate under three conditions
        # A) Cold prefill (target prompt, no injection)
        cold_text, cold_ppl, cold_time = generate_with_kv(
            model, tokenizer, target_prompt, max_new_tokens=max_new_tokens,
        )

        # B) Verbatim donor KV injection (no correction)
        verbatim_text, verbatim_ppl, _ = generate_with_kv(
            model, tokenizer, target_prompt,
            injected_keys=donor_keys,
            injected_values=donor_values,
            max_new_tokens=max_new_tokens,
            inject_layers=inject_layers,
        )

        # C) Corrected donor KV injection
        corrected_text, corrected_ppl, corrected_time = generate_with_kv(
            model, tokenizer, target_prompt,
            injected_keys=corrected_keys,
            injected_values=corrected_values,
            max_new_tokens=max_new_tokens,
            inject_layers=inject_layers,
        )

        cold_match = check_answer(cold_text, pair["answer"])
        verbatim_match = check_answer(verbatim_text, pair["answer"])
        corrected_match = check_answer(corrected_text, pair["answer"])

        print(f"  Cold PPL:      {cold_ppl:.2f}  match={cold_match}  text={cold_text[:60]}...")
        print(f"  Verbatim PPL:  {verbatim_ppl:.2f}  match={verbatim_match}  text={verbatim_text[:60]}...")
        print(f"  Corrected PPL: {corrected_ppl:.2f}  match={corrected_match}  text={corrected_text[:60]}...")
        print(f"  Probe time:    {probe_time_ms:.1f}ms")

        results.append(SampleResult(
            sample_id=f"triviaqa_{idx}",
            question=pair["question"],
            reference_answer=pair["answer"],
            cold_text=cold_text,
            verbatim_text=verbatim_text,
            corrected_text=corrected_text,
            cold_ppl=cold_ppl,
            verbatim_ppl=verbatim_ppl,
            corrected_ppl=corrected_ppl,
            cold_answer_match=cold_match,
            verbatim_answer_match=verbatim_match,
            corrected_answer_match=corrected_match,
            cold_time_ms=cold_time,
            corrected_time_ms=corrected_time,
            probe_time_ms=probe_time_ms,
            layer_deviations=deviations,
            corrections=corrections,
        ))

        # Free GPU memory
        del donor_keys, donor_values, target_keys, target_values
        del corrected_keys, corrected_values
        torch.cuda.empty_cache()

    # ==========================================
    # AGGREGATE RESULTS
    # ==========================================
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)

    if not results:
        print("No results!")
        return {}

    cold_ppls = [r.cold_ppl for r in results if r.cold_ppl < 100]
    verbatim_ppls = [r.verbatim_ppl for r in results if r.verbatim_ppl < 100]
    corrected_ppls = [r.corrected_ppl for r in results if r.corrected_ppl < 100]

    cold_matches = sum(1 for r in results if r.cold_answer_match)
    verbatim_matches = sum(1 for r in results if r.verbatim_answer_match)
    corrected_matches = sum(1 for r in results if r.corrected_answer_match)

    n = len(results)
    print(f"\nSamples: {n}")
    print(f"\n{'Condition':<20} {'Mean PPL':>10} {'QA Match':>10} {'Match Rate':>12}")
    print("-" * 55)
    print(f"{'Cold (baseline)':<20} {statistics.mean(cold_ppls):>10.3f} {cold_matches:>10} {cold_matches/n:>11.1%}")
    print(f"{'Verbatim inject':<20} {statistics.mean(verbatim_ppls):>10.3f} {verbatim_matches:>10} {verbatim_matches/n:>11.1%}")
    print(f"{'Corrected inject':<20} {statistics.mean(corrected_ppls):>10.3f} {corrected_matches:>10} {corrected_matches/n:>11.1%}")

    # PPL ratios
    verbatim_ratios = [v / c for v, c in zip(verbatim_ppls, cold_ppls) if c > 0]
    corrected_ratios = [co / c for co, c in zip(corrected_ppls, cold_ppls) if c > 0]

    print(f"\nPPL ratio (verbatim/cold):   {statistics.mean(verbatim_ratios):.4f}")
    print(f"PPL ratio (corrected/cold):  {statistics.mean(corrected_ratios):.4f}")

    # Correction improvement
    if verbatim_ratios and corrected_ratios:
        improvement = statistics.mean(verbatim_ratios) - statistics.mean(corrected_ratios)
        print(f"Correction improvement:      {improvement:+.4f} PPL ratio")

    # Per-layer deviation summary (from first sample)
    if results[0].layer_deviations:
        print("\n--- Average per-layer K cosine similarity ---")
        for d in results[0].layer_deviations:
            bar = "#" * int(d.k_cosine_sim * 40)
            inject = " [INJECT]" if d.layer_idx in inject_layers else ""
            print(f"  L{d.layer_idx:>2}: {d.k_cosine_sim:.4f} {bar}{inject}")

    # Correction effectiveness summary
    if results[0].corrections:
        k_before = statistics.mean(c.before_k_sim for c in results[0].corrections)
        k_after = statistics.mean(c.after_k_sim for c in results[0].corrections)
        v_before = statistics.mean(c.before_v_sim for c in results[0].corrections)
        v_after = statistics.mean(c.after_v_sim for c in results[0].corrections)
        print("\nAffine correction (middle layers):")
        print(f"  K similarity: {k_before:.4f} → {k_after:.4f} ({k_after - k_before:+.4f})")
        print(f"  V similarity: {v_before:.4f} → {v_after:.4f} ({v_after - v_before:+.4f})")

    # Timing
    cold_times = [r.cold_time_ms for r in results]
    corrected_times = [r.corrected_time_ms for r in results]
    probe_times = [r.probe_time_ms for r in results]
    print("\nTiming:")
    print(f"  Cold prefill:  {statistics.mean(cold_times):.0f}ms")
    print(f"  Corrected:     {statistics.mean(corrected_times):.0f}ms")
    print(f"  Probe overhead:{statistics.mean(probe_times):.0f}ms")
    if statistics.mean(corrected_times) > 0:
        speedup = statistics.mean(cold_times) / statistics.mean(corrected_times)
        print(f"  Speedup:       {speedup:.2f}x")

    # Verdict
    print("\n--- VERDICT ---")
    if corrected_ratios:
        mean_ratio = statistics.mean(corrected_ratios)
        qa_delta = (cold_matches - corrected_matches) / n
        print(f"PPL ratio:     {mean_ratio:.4f} {'PASS' if mean_ratio <= 1.065 else 'FAIL'} (threshold <= 1.065)")
        print(f"QA degradation:{qa_delta:+.1%} {'PASS' if qa_delta <= 0.05 else 'FAIL'} (threshold <= 5%)")
        if mean_ratio <= 1.065 and qa_delta <= 0.05:
            print("SEMANTIC KV PROJECTION: FEASIBLE")
        else:
            print("SEMANTIC KV PROJECTION: NEEDS WORK")

    return {
        "n_samples": n,
        "cold_mean_ppl": statistics.mean(cold_ppls),
        "verbatim_mean_ppl": statistics.mean(verbatim_ppls),
        "corrected_mean_ppl": statistics.mean(corrected_ppls),
        "verbatim_ppl_ratio": statistics.mean(verbatim_ratios) if verbatim_ratios else 0,
        "corrected_ppl_ratio": statistics.mean(corrected_ratios) if corrected_ratios else 0,
        "cold_qa_match": cold_matches / n,
        "verbatim_qa_match": verbatim_matches / n,
        "corrected_qa_match": corrected_matches / n,
        "mean_probe_time_ms": statistics.mean(probe_times),
    }


def main():
    parser = argparse.ArgumentParser(description="Semantic KV Projection experiment")
    parser.add_argument("--n-pairs", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results = run_experiment(n_pairs=args.n_pairs, max_new_tokens=args.max_tokens)

    if args.output and results:
        from pathlib import Path
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()

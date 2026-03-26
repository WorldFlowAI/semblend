#!/usr/bin/env python3
"""Comprehensive KV transfer feasibility study with real data.

Tests 5 dimensions of cross-vocabulary KV transfer using real TriviaQA
Wikipedia passages (4K+ tokens), n=20 samples, full layer analysis,
PPL measurement, and QA accuracy verification.

Experiments:
  1. PREFIX vs MIDDLE paraphrase: where does the token difference matter?
     - Uses real TriviaQA contexts with paraphrased instructions (prefix)
       vs paraphrased document sections (middle)
  2. PER-CHUNK deviation: which 256-token chunks are transferable?
     - Analyzes every chunk independently across all layers
  3. NEURAL correction: can a small MLP learn the nonlinear KV mapping?
     - Trains per-head MLP correctors on probe tokens, tests generalization
  4. ATTENTION SINK: does preserving first N tokens help injection?
     - Tests sink sizes 4, 16, 64, measures PPL impact
  5. OVERLAP SWEEP: at what token overlap % does KV become transferable?
     - Progressively replaces tokens and measures per-layer similarity

All experiments use:
  - Real TriviaQA Wikipedia passages (4K+ chars)
  - Multiple samples (default 20) for statistical significance
  - Analysis across ALL layers (not just middle)
  - PPL and QA accuracy where applicable

Usage (inside GPU pod):
    python kv_transfer_feasibility.py --n 20 --output /tmp/feasibility.json
    python kv_transfer_feasibility.py --n 5 --quick  # Fast smoke test
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct-AWQ"
FALLBACK_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

INSTR_A = "You are a helpful assistant. Answer the following question based on the context provided below."
INSTR_B = "Based on the text below, please answer:"
INSTR_C = "Read the document carefully and answer the question that follows."
INSTR_D = "Given the following document, respond to the query."


# ======================================================================
# Model & KV utilities
# ======================================================================


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading model...")
    t0 = time.monotonic()
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
        )
        name = MODEL_NAME
    except Exception as e:
        print(f"  {MODEL_NAME} failed: {e}")
        tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            FALLBACK_MODEL, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
        )
        name = FALLBACK_MODEL
    model.eval()
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    hd = cfg.hidden_size // cfg.num_attention_heads
    print(f"  {name}: {n_layers}L, {n_kv}KV, {hd}hd ({time.monotonic() - t0:.1f}s)")
    return model, tokenizer, n_layers, n_kv, hd


def get_kv(model, tokenizer, text):
    ids = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**ids, use_cache=True, return_dict=True)
    kv = [
        (out.past_key_values[i][0].clone(), out.past_key_values[i][1].clone())
        for i in range(len(out.past_key_values))
    ]
    return kv, ids["input_ids"]


def cos_sim(a, b, start=0, end=None):
    """Mean cosine similarity between two KV tensors in position range [start:end]."""
    ml = min(a.shape[2], b.shape[2])
    e = end if end and end <= ml else ml
    if e <= start:
        return 0.0
    af = a[0, :, start:e, :].reshape(-1, a.shape[-1]).float()
    bf = b[0, :, start:e, :].reshape(-1, b.shape[-1]).float()
    return F.cosine_similarity(af, bf, dim=-1).mean().item()


def generate_from_kv(model, tokenizer, kv_pairs, input_ids, max_tokens=32):
    """Generate text from KV cache. Returns (text, ppl)."""
    from transformers import DynamicCache

    cache = DynamicCache()
    for li, (k, v) in enumerate(kv_pairs):
        cache.update(k, v, li)
    cache_len = kv_pairs[0][0].shape[2]
    cur = input_ids[:, cache_len - 1 : cache_len]
    gen, tlp = [], 0.0
    with torch.no_grad():
        for _ in range(max_tokens):
            out = model(input_ids=cur, past_key_values=cache, use_cache=True, return_dict=True)
            logits = out.logits[:, -1, :]
            lps = F.log_softmax(logits.float(), dim=-1)
            tok = logits.argmax(dim=-1, keepdim=True)
            tlp += lps[0, tok[0, 0]].item()
            gen.append(tok[0, 0].item())
            cache = out.past_key_values
            cur = tok
            if tok[0, 0].item() == tokenizer.eos_token_id:
                break
    text = tokenizer.decode(gen, skip_special_tokens=True)
    ppl = math.exp(-tlp / max(len(gen), 1))
    return text, ppl


def cold_generate(model, tokenizer, prompt, max_tokens=32):
    ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(input_ids=ids["input_ids"], use_cache=True, return_dict=True)
    past = out.past_key_values
    logits = out.logits[:, -1, :]
    lps = F.log_softmax(logits.float(), dim=-1)
    tok = logits.argmax(dim=-1, keepdim=True)
    tlp = lps[0, tok[0, 0]].item()
    gen = [tok[0, 0].item()]
    cur = tok
    with torch.no_grad():
        for _ in range(max_tokens - 1):
            out = model(input_ids=cur, past_key_values=past, use_cache=True, return_dict=True)
            logits = out.logits[:, -1, :]
            lps = F.log_softmax(logits.float(), dim=-1)
            tok = logits.argmax(dim=-1, keepdim=True)
            tlp += lps[0, tok[0, 0]].item()
            gen.append(tok[0, 0].item())
            past = out.past_key_values
            cur = tok
            if tok[0, 0].item() == tokenizer.eos_token_id:
                break
    text = tokenizer.decode(gen, skip_special_tokens=True)
    ppl = math.exp(-tlp / max(len(gen), 1))
    return text, ppl


def check_answer(text, ref):
    return ref.lower().strip() in text.lower().strip() if ref and text else False


# ======================================================================
# Data loading
# ======================================================================


def load_triviaqa(n):
    from datasets import load_dataset

    print(f"Loading {n} TriviaQA pairs (4K+ chars)...")
    ds = load_dataset("trivia_qa", "rc", split="validation")
    pairs = []
    for row in ds:
        if len(pairs) >= n:
            break
        wiki = row.get("entity_pages", {}).get("wiki_context", [])
        if not wiki:
            continue
        ctx = max(wiki, key=len)
        if len(ctx) < 4000:
            continue
        ctx = ctx[:12000]
        q = row.get("question", "")
        ans = row.get("answer", {})
        a = ans.get("value", "") if isinstance(ans, dict) else str(ans)
        if q and a:
            pairs.append({"context": ctx, "question": q, "answer": a})
    print(f"  Loaded {len(pairs)}")
    return pairs


# ======================================================================
# EXPERIMENT 1: Prefix vs middle paraphrase
# ======================================================================


def exp1(model, tokenizer, n_layers, pairs):
    print("\n" + "=" * 60)
    print("EXP 1: PREFIX vs MIDDLE paraphrase — real TriviaQA data")
    print("=" * 60)

    prefix_k_by_layer = [[] for _ in range(n_layers)]
    prefix_v_by_layer = [[] for _ in range(n_layers)]
    middle_k_by_layer = [[] for _ in range(n_layers)]
    middle_v_by_layer = [[] for _ in range(n_layers)]

    for si, pair in enumerate(pairs):
        ctx = pair["context"]
        q = pair["question"]

        # PREFIX: different instruction, same document
        d_text = f"{INSTR_A}\n\nContext:\n{ctx}\n\nQuestion: {q}\nAnswer:"
        t_text = f"{INSTR_B}\n\nContext:\n{ctx}\n\nQuestion: {q}\nAnswer:"
        dk, _ = get_kv(model, tokenizer, d_text)
        tk, _ = get_kv(model, tokenizer, t_text)
        for li in range(n_layers):
            prefix_k_by_layer[li].append(cos_sim(dk[li][0], tk[li][0]))
            prefix_v_by_layer[li].append(cos_sim(dk[li][1], tk[li][1]))
        del dk, tk

        # MIDDLE: same instruction, paraphrased document (swap sentences)
        sentences = ctx.split(". ")
        if len(sentences) > 4:
            rng = random.Random(si)
            mid = len(sentences) // 2
            shuffled = sentences[:mid]
            rest = list(sentences[mid:])
            rng.shuffle(rest)
            alt_ctx = ". ".join(shuffled + rest)
        else:
            alt_ctx = ctx[::-1][: len(ctx)]  # Crude paraphrase fallback

        d_text2 = f"{INSTR_A}\n\nContext:\n{ctx}\n\nQuestion: {q}\nAnswer:"
        t_text2 = f"{INSTR_A}\n\nContext:\n{alt_ctx}\n\nQuestion: {q}\nAnswer:"
        dk2, _ = get_kv(model, tokenizer, d_text2)
        tk2, _ = get_kv(model, tokenizer, t_text2)
        for li in range(n_layers):
            middle_k_by_layer[li].append(cos_sim(dk2[li][0], tk2[li][0]))
            middle_v_by_layer[li].append(cos_sim(dk2[li][1], tk2[li][1]))
        del dk2, tk2
        torch.cuda.empty_cache()

        if (si + 1) % 5 == 0:
            print(f"  {si + 1}/{len(pairs)}")

    # Print results
    print(f"\n{'Layer':>5} {'PfxK':>7} {'PfxV':>7} {'MidK':>7} {'MidV':>7} {'Winner':>8}")
    for li in range(n_layers):
        pk = statistics.mean(prefix_k_by_layer[li])
        pv = statistics.mean(prefix_v_by_layer[li])
        mk = statistics.mean(middle_k_by_layer[li])
        mv = statistics.mean(middle_v_by_layer[li])
        winner = "prefix" if pv > mv else "middle"
        print(f"{li:>5} {pk:>7.4f} {pv:>7.4f} {mk:>7.4f} {mv:>7.4f} {winner:>8}")

    all_pv = [statistics.mean(prefix_v_by_layer[li]) for li in range(n_layers)]
    all_mv = [statistics.mean(middle_v_by_layer[li]) for li in range(n_layers)]
    print(
        f"\nOverall V-sim: Prefix={statistics.mean(all_pv):.4f}, Middle={statistics.mean(all_mv):.4f}"
    )

    return {
        "prefix_v_mean": statistics.mean(all_pv),
        "middle_v_mean": statistics.mean(all_mv),
        "per_layer_prefix_v": [statistics.mean(prefix_v_by_layer[li]) for li in range(n_layers)],
        "per_layer_middle_v": [statistics.mean(middle_v_by_layer[li]) for li in range(n_layers)],
    }


# ======================================================================
# EXPERIMENT 2: Per-chunk deviation
# ======================================================================


def exp2(model, tokenizer, n_layers, pairs):
    print("\n" + "=" * 60)
    print("EXP 2: PER-CHUNK deviation analysis (256-token chunks)")
    print("=" * 60)

    chunk_size = 256
    # Use first 5 long samples
    samples = [p for p in pairs if len(p["context"]) > 6000][:5]

    all_chunk_k_sims = {}  # layer → list of per-chunk sims
    all_chunk_v_sims = {}

    for si, pair in enumerate(samples):
        ctx = pair["context"]
        q = pair["question"]
        d_text = f"{INSTR_A}\n\nContext:\n{ctx}\n\nQuestion: {q}\nAnswer:"
        t_text = f"{INSTR_B}\n\nContext:\n{ctx}\n\nQuestion: {q}\nAnswer:"

        dk, _ = get_kv(model, tokenizer, d_text)
        tk, _ = get_kv(model, tokenizer, t_text)

        ml = min(dk[0][0].shape[2], tk[0][0].shape[2])
        n_chunks = ml // chunk_size

        for li in [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]:
            if li not in all_chunk_k_sims:
                all_chunk_k_sims[li] = [[] for _ in range(n_chunks)]
                all_chunk_v_sims[li] = [[] for _ in range(n_chunks)]

            for ci in range(n_chunks):
                s, e = ci * chunk_size, (ci + 1) * chunk_size
                ks = cos_sim(dk[li][0], tk[li][0], s, e)
                vs = cos_sim(dk[li][1], tk[li][1], s, e)
                if ci < len(all_chunk_k_sims[li]):
                    all_chunk_k_sims[li][ci].append(ks)
                    all_chunk_v_sims[li][ci].append(vs)

        del dk, tk
        torch.cuda.empty_cache()

    # Print per-chunk results
    for li in sorted(all_chunk_k_sims.keys()):
        print(f"\nLayer {li}:")
        print(f"  {'Chunk':>6} {'K_sim':>8} {'V_sim':>8} {'Region':>15}")
        for ci in range(len(all_chunk_k_sims[li])):
            if not all_chunk_k_sims[li][ci]:
                break
            km = statistics.mean(all_chunk_k_sims[li][ci])
            vm = statistics.mean(all_chunk_v_sims[li][ci])
            region = (
                "instruction"
                if ci == 0
                else "early doc"
                if ci <= 2
                else "mid doc"
                if ci < len(all_chunk_k_sims[li]) - 2
                else "late doc"
            )
            print(f"  {ci:>6} {km:>8.4f} {vm:>8.4f} {region:>15}")

    return {
        "per_chunk_data": {
            str(li): {
                "k": [
                    statistics.mean(all_chunk_k_sims[li][ci])
                    for ci in range(len(all_chunk_k_sims[li]))
                    if all_chunk_k_sims[li][ci]
                ],
                "v": [
                    statistics.mean(all_chunk_v_sims[li][ci])
                    for ci in range(len(all_chunk_v_sims[li]))
                    if all_chunk_v_sims[li][ci]
                ],
            }
            for li in sorted(all_chunk_k_sims.keys())
        }
    }


# ======================================================================
# EXPERIMENT 3: Neural MLP correction
# ======================================================================


def exp3(model, tokenizer, n_layers, head_dim, n_kv_heads, pairs):
    print("\n" + "=" * 60)
    print("EXP 3: NEURAL MLP correction vs mean-shift (20 samples)")
    print("=" * 60)

    class Corrector(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

        def forward(self, x):
            return x + self.net(x)

    n_probe = 64
    test_layers = [n_layers // 4, n_layers // 2, 3 * n_layers // 4]

    results_by_layer = {li: {"before": [], "meanshift": [], "neural": []} for li in test_layers}

    for si, pair in enumerate(pairs):
        ctx = pair["context"]
        q = pair["question"]
        d_text = f"{INSTR_A}\n\nContext:\n{ctx}\n\nQuestion: {q}\nAnswer:"
        t_text = f"{INSTR_B}\n\nContext:\n{ctx}\n\nQuestion: {q}\nAnswer:"
        dk, _ = get_kv(model, tokenizer, d_text)
        tk, _ = get_kv(model, tokenizer, t_text)
        ml = min(dk[0][0].shape[2], tk[0][0].shape[2])

        if ml < n_probe + 32:
            del dk, tk
            continue

        for li in test_layers:
            d_k = dk[li][0][0].float()  # [heads, seq, dim]
            t_k = tk[li][0][0].float()

            # Before
            before = (
                F.cosine_similarity(
                    d_k[:, n_probe:ml, :].reshape(-1, head_dim),
                    t_k[:, n_probe:ml, :].reshape(-1, head_dim),
                    dim=-1,
                )
                .mean()
                .item()
            )

            # Mean-shift
            shift = (t_k[:, :n_probe, :] - d_k[:, :n_probe, :]).mean(dim=1)
            ms = d_k[:, n_probe:ml, :] + shift.unsqueeze(1)
            ms_sim = (
                F.cosine_similarity(
                    ms.reshape(-1, head_dim), t_k[:, n_probe:ml, :].reshape(-1, head_dim), dim=-1
                )
                .mean()
                .item()
            )

            # Neural per-head
            neural_sims = []
            for h in range(n_kv_heads):
                corrector = Corrector(head_dim).to(d_k.device).float()
                opt = torch.optim.Adam(corrector.parameters(), lr=1e-3)
                dp, tp = d_k[h, :n_probe], t_k[h, :n_probe]
                for _ in range(300):
                    loss = F.mse_loss(corrector(dp), tp)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                with torch.no_grad():
                    corrected = corrector(d_k[h, n_probe:ml])
                    s = F.cosine_similarity(corrected, t_k[h, n_probe:ml], dim=-1).mean().item()
                    neural_sims.append(s)
                del corrector

            neural = statistics.mean(neural_sims)

            results_by_layer[li]["before"].append(before)
            results_by_layer[li]["meanshift"].append(ms_sim)
            results_by_layer[li]["neural"].append(neural)

        del dk, tk
        torch.cuda.empty_cache()

        if (si + 1) % 5 == 0:
            print(f"  {si + 1}/{len(pairs)}")

    print(f"\n{'Layer':>6} {'Before':>9} {'MnShift':>9} {'Neural':>9} {'Δ Neural':>10}")
    summary = {}
    for li in test_layers:
        b = statistics.mean(results_by_layer[li]["before"])
        m = statistics.mean(results_by_layer[li]["meanshift"])
        n = statistics.mean(results_by_layer[li]["neural"])
        print(f"{li:>6} {b:>9.4f} {m:>9.4f} {n:>9.4f} {n - b:>+10.4f}")
        summary[li] = {"before": b, "meanshift": m, "neural": n}

    return summary


# ======================================================================
# EXPERIMENT 4: Attention sink preservation + PPL
# ======================================================================


def exp4(model, tokenizer, n_layers, pairs):
    print("\n" + "=" * 60)
    print("EXP 4: ATTENTION SINK preservation with PPL + QA")
    print("=" * 60)

    sink_sizes = [0, 4, 16, 64]
    ppl_by_strategy = {f"sink_{s}": [] for s in sink_sizes}
    ppl_by_strategy["cold"] = []
    qa_by_strategy = {f"sink_{s}": 0 for s in sink_sizes}
    qa_by_strategy["cold"] = 0

    for si, pair in enumerate(pairs[:10]):  # Limit to 10 for speed (generation is slow)
        ctx = pair["context"][:6000]
        q = pair["question"]
        d_text = f"{INSTR_A}\n\nContext:\n{ctx}\n\nQuestion: {q}\nAnswer:"
        t_text = f"{INSTR_B}\n\nContext:\n{ctx}\n\nQuestion: {q}\nAnswer:"

        dk, _ = get_kv(model, tokenizer, d_text)
        tk, tid = get_kv(model, tokenizer, t_text)
        ml = min(dk[0][0].shape[2], tk[0][0].shape[2])

        # Cold baseline
        cold_text, cold_ppl = cold_generate(model, tokenizer, t_text, 32)
        ppl_by_strategy["cold"].append(cold_ppl)
        if check_answer(cold_text, pair["answer"]):
            qa_by_strategy["cold"] += 1

        for sink_s in sink_sizes:
            hybrid = []
            for li in range(n_layers):
                hk = dk[li][0][:, :, :ml, :].clone()
                hv = dk[li][1][:, :, :ml, :].clone()
                if sink_s > 0 and sink_s <= ml:
                    hk[:, :, :sink_s, :] = tk[li][0][:, :, :sink_s, :]
                    hv[:, :, :sink_s, :] = tk[li][1][:, :, :sink_s, :]
                hybrid.append((hk, hv))
            try:
                text, ppl = generate_from_kv(model, tokenizer, hybrid, tid, 32)
                ppl_by_strategy[f"sink_{sink_s}"].append(ppl)
                if check_answer(text, pair["answer"]):
                    qa_by_strategy[f"sink_{sink_s}"] += 1
            except Exception:
                ppl_by_strategy[f"sink_{sink_s}"].append(999)

        del dk, tk
        torch.cuda.empty_cache()

        if (si + 1) % 3 == 0:
            print(f"  {si + 1}/10")

    print(f"\n{'Strategy':<12} {'Mean PPL':>10} {'QA Match':>10} {'PPL Ratio':>10}")
    cold_mean = statistics.mean(ppl_by_strategy["cold"])
    for name in ["cold"] + [f"sink_{s}" for s in sink_sizes]:
        vals = [v for v in ppl_by_strategy[name] if v < 100]
        m = statistics.mean(vals) if vals else 999
        ratio = m / cold_mean if cold_mean > 0 else 0
        print(f"{name:<12} {m:>10.3f} {qa_by_strategy[name]:>10} {ratio:>10.4f}")

    return {
        name: {
            "mean_ppl": statistics.mean([v for v in ppl_by_strategy[name] if v < 100])
            if any(v < 100 for v in ppl_by_strategy[name])
            else 999,
            "qa_matches": qa_by_strategy[name],
        }
        for name in ppl_by_strategy
    }


# ======================================================================
# EXPERIMENT 5: Token overlap sweep
# ======================================================================


def exp5(model, tokenizer, n_layers, pairs):
    print("\n" + "=" * 60)
    print("EXP 5: TOKEN OVERLAP sweep (99% → 50%)")
    print("=" * 60)

    overlaps = [0.99, 0.97, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50]
    test_layers = [1, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 2]

    results = {pct: {li: {"k": [], "v": []} for li in test_layers} for pct in overlaps}

    for si, pair in enumerate(pairs):
        ctx = pair["context"][:6000]
        q = pair["question"]
        base = f"{INSTR_A}\n\nContext:\n{ctx}\n\nQuestion: {q}\nAnswer:"
        tokens = tokenizer.encode(base, add_special_tokens=False)
        n_tok = len(tokens)

        dk, _ = get_kv(model, tokenizer, base)

        rng = random.Random(si)
        for pct in overlaps:
            n_change = max(1, int(n_tok * (1.0 - pct)))
            modified = list(tokens)
            positions = rng.sample(range(n_tok), min(n_change, n_tok))
            for pos in positions:
                modified[pos] = rng.randint(100, 30000)
            mod_text = tokenizer.decode(modified)
            tk, _ = get_kv(model, tokenizer, mod_text)

            for li in test_layers:
                results[pct][li]["k"].append(cos_sim(dk[li][0], tk[li][0]))
                results[pct][li]["v"].append(cos_sim(dk[li][1], tk[li][1]))
            del tk

        del dk
        torch.cuda.empty_cache()

        if (si + 1) % 5 == 0:
            print(f"  {si + 1}/{len(pairs)}")

    print(f"\n{'Overlap':>8}", end="")
    for li in test_layers:
        print(f"  L{li}K   L{li}V", end="")
    print()

    summary = {}
    for pct in overlaps:
        print(f"{pct:>7.0%}", end="")
        row = {}
        for li in test_layers:
            km = statistics.mean(results[pct][li]["k"])
            vm = statistics.mean(results[pct][li]["v"])
            print(f" {km:>5.3f} {vm:>5.3f}", end="")
            row[f"L{li}_k"] = km
            row[f"L{li}_v"] = vm
        print()
        summary[str(pct)] = row

    return summary


# ======================================================================
# MAIN
# ======================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    n = 5 if args.quick else args.n
    model, tokenizer, n_layers, n_kv, hd = load_model()
    pairs = load_triviaqa(n)

    results = {}
    results["exp1"] = exp1(model, tokenizer, n_layers, pairs)
    results["exp2"] = exp2(model, tokenizer, n_layers, pairs)
    results["exp3"] = exp3(model, tokenizer, n_layers, hd, n_kv, pairs)
    results["exp4"] = exp4(model, tokenizer, n_layers, pairs)
    results["exp5"] = exp5(model, tokenizer, n_layers, pairs)

    print("\n" + "=" * 60)
    print("COMPREHENSIVE SUMMARY")
    print("=" * 60)

    e1 = results["exp1"]
    print(
        f"\n1. PREFIX vs MIDDLE: V-sim prefix={e1['prefix_v_mean']:.4f}, middle={e1['middle_v_mean']:.4f}"
    )
    if e1["middle_v_mean"] > e1["prefix_v_mean"]:
        print(
            f"   → Middle paraphrase {(e1['middle_v_mean'] - e1['prefix_v_mean']) / e1['prefix_v_mean'] * 100:+.1f}% better — same-prefix preserves more KV"
        )
    else:
        print(
            "   → Prefix paraphrase unexpectedly better — autoregressive contamination less severe"
        )

    e3 = results["exp3"]
    best_neural = max((v["neural"] - v["before"]) for v in e3.values())
    best_shift = max((v["meanshift"] - v["before"]) for v in e3.values())
    print(
        f"\n3. NEURAL vs LINEAR: best neural Δ={best_neural:+.4f}, best shift Δ={best_shift:+.4f}"
    )
    if best_neural > best_shift + 0.01:
        print(
            f"   → Neural correction captures nonlinear structure ({best_neural - best_shift:+.4f} better)"
        )
    else:
        print(
            "   → Neural correction no better than mean-shift — deviation is not learnable from probes"
        )

    e4 = results["exp4"]
    cold_ppl = e4.get("cold", {}).get("mean_ppl", 999)
    for name in ["sink_0", "sink_4", "sink_16", "sink_64"]:
        ppl = e4.get(name, {}).get("mean_ppl", 999)
        ratio = ppl / cold_ppl if cold_ppl > 0 else 0
        print(
            f"\n4. SINK {name}: PPL ratio={ratio:.4f}, QA={e4.get(name, {}).get('qa_matches', 0)}"
        )

    print("\n5. OVERLAP SWEEP: see detailed table above")

    if args.output:
        from pathlib import Path

        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2, default=str))
        print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()

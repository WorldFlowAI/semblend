# Semantic KV Cache Reuse for SGLang — Provider Design

**Status**: Design draft — response to ibifrost/SGLang `Draft_Prefix_Matching.md` (Chenxin Wu, 2026-04-22).

**Owners**: Zach Bennett (semblend), Chenxin Wu (SGLang).

**Linked artifacts**:
- SGLang draft: https://github.com/ibifrost/sglang/blob/feature/support_fuzzy_prefix_match_opensource/Draft_Prefix_Matching.md
- SemBlend paper: `autoresearch-semblend/paper/semblend.pdf`
- Reference implementation: `semblend.integration.sglang.provider.SemBlendProviderAdapter` (this repo)

## 1. Summary

Chenxin's `FuzzyMatchProvider` ABC is exactly the right shape to plug a semantic provider into SGLang. We propose **three additive, optional extensions** to `FuzzyMatchResult` so that a semantic provider can express what the SemBlend paper does: (a) N:M and non-contiguous token alignment across possibly multiple donors, (b) per-layer selective recomputation (the bathtub curve), and (c) an explicit quality-signals contract. `TokenBlockMatchProvider` is untouched — all new fields default to `None` and `_correct_fuzzy_kv_rope` falls back to the existing contiguous-segment flow when they are absent.

The semantic provider itself, `SemanticEmbeddingProvider`, is a thin wrapper in SGLang that delegates to `semblend.integration.sglang.provider.SemBlendProviderAdapter` in the open-source `semblend` pip package. SemBlend is **process-local**: in-process MiniLM embedding and a numpy donor store. All pipeline machinery (embedding, ANN search, alignment, bathtub curve, partial-attention planning) lives in `semblend_core/` and is reused unchanged. There is no service dependency, no network hop, no daemon — opt-in via `fuzzy_match_provider="SemanticEmbedding"` and disabled means zero cost.

## 2. Alignment with Chenxin's design principles

| Principle | How we preserve it |
|---|---|
| **Non-pollution** of the exact-match RadixTree | Semantic provider never touches `root_node` or `TreeNode.value`. All state lives in its own in-process store. |
| **Config-driven** opt-in | `fuzzy_match_provider="SemanticEmbedding"` plus new fields on `FuzzyMatchConfig`. Disabled → zero cost. |
| **Position-aware** RoPE correction | Reuse the exact `_correct_fuzzy_kv_rope` allocator path. For N:M segments we iterate per-segment with the same reverse-then-reapply kernel. |
| **No double-counting** of pool indices | `FuzzyMatchSegment` carries `donor_node_id` + `donor_offset` + `length` (a NodeRef), not a tensor of pool indices. Same NodeRef pattern Chenxin already designed into `NonPrefixKVStore`. Pool indices only live in the radix tree. |

## 3. Why the current `FuzzyMatchResult` isn't enough for semantic

SemBlend's paper result is PPL ratio ≤ 1.065 across CNN/DailyMail, XSum, MultiNews, WikiHow with hit rate up to ~74% at 8K tokens (§ 4.3, Table 7). That number depends on three mechanisms the current `FuzzyMatchResult` can't describe:

### 3.1 Non-contiguous, N:M alignment (multi-segment, multi-donor)

For REORDER and PARAPHRASE matches, the donor and target don't share a single contiguous block. They share several disjoint spans, potentially at different offsets in the donor, and in rare cases across multiple donors. SemBlend already produces this as `PipelineResult.slot_actions` (`semblend_core/pipeline.py:92`) and `composite_plan` (`pipeline.py:102`) — we need a way to return it.

Today's field: `kv_cache_indices: torch.Tensor` + single `cached_start_pos`. This can only describe a single contiguous span.

### 3.2 Per-layer selective recomputation (bathtub curve)

SemBlend paper § 3.4 (Eq. 4): early and late transformer layers are more sensitive to position changes than middle layers. Reusing all layers unconditionally when similarity is below 1.0 causes the accuracy degradation that Chenxin is seeing in `TokenBlockMatchProvider`. SemBlend mitigates this by computing `LayerDeviation` per layer (`semblend_core/bathtub.py:233`) and recomputing flagged layers fresh. Presets are model-specific: `llama` (σ_e=0.45, σ_l=0.15), `qwen2.5-7b` (σ_e=0.15, σ_l=0.35) — see `PRESETS` at `bathtub.py:99`.

Today's field: no mechanism to tell the model_runner "reuse K/V for these matched tokens on layers X, Y, but recompute on layers Z".

### 3.3 Quality gating

Chenxin's draft has `fuzzy_min_match_length` as the only guardrail. SemBlend also uses:
- Cosine similarity threshold τ=0.60 (paper § 3.1, ablation Table 9)
- Chunk-hash gating (exact LMCache block match at chunk boundaries — paper Figure 4 shows PPL=1.000±0.001 when this gate fires)
- Minimum reuse ratio — below which we'd pay RoPE correction cost for marginal token savings

These signals are currently hidden inside the provider. Exposing them on the result object lets SGLang log/monitor/trace them at the scheduler level and lets debuggers understand why a match fired (or didn't).

## 4. Proposed interface deltas

All additions are optional. `TokenBlockMatchProvider` never sets them and sees no behavioral change.

```python
# In sglang/srt/mem_cache/fuzzy_match/provider.py

@dataclass
class FuzzyMatchResult:
    # --- existing fields (unchanged) ---
    cached_token_count: int
    cached_token_ids: List[int]
    prompt_token_count: int
    kv_cache_indices: torch.Tensor    # legacy, single contiguous span
    position_offset: int
    cached_start_pos: int = 0
    _match_entry: Any = None

    # --- NEW (optional, provider-dependent) ---
    segments: Optional[List["FuzzyMatchSegment"]] = None
    layer_recompute_mask: Optional[List[bool]] = None
    quality_signals: Optional["QualitySignals"] = None

    # --- NEW (lifetime management for fuzzy donors) ---
    # ID of the donor's TreeNode in radix_tree._node_registry. RadixCache
    # inc_lock_ref's the donor at match time and dec_lock_ref's at
    # cache_finished_req so its KV slots aren't LRU-evicted while the
    # recipient request is consuming them.
    donor_last_node_id: Optional[int] = None


@dataclass
class FuzzyMatchSegment:
    """One contiguous span of matched tokens in an N:M alignment plan.

    Addressed via NodeRef rather than raw pool indices, preserving Chenxin's
    "no double-counting" principle: the radix tree is the single owner of
    pool indices. The model_runner resolves a segment to slots at consume
    time via:

        node = radix_tree._node_registry[donor_node_id]
        donor_kv_slots = node.value[donor_offset : donor_offset + length]

    Multiple segments may reference different donors (multi-donor scatter).
    """
    target_positions: torch.Tensor   # absolute positions in the new prompt
    donor_positions: torch.Tensor    # source positions in the donor (for RoPE delta)

    # NodeRef-based addressing — preferred. Resolves at consume time so the
    # donor TreeNode's lifetime governs slot validity (paired with
    # FuzzyMatchResult.donor_last_node_id inc_lock_ref protection).
    donor_node_id: Optional[int] = None
    donor_offset: Optional[int] = None
    length: Optional[int] = None

    # Legacy: raw pool-indices tensor. Used by TokenBlockMatch's existing
    # contiguous path. New providers populate donor_node_id instead.
    donor_kv_indices: Optional[torch.Tensor] = None

    donor_req_id: Optional[str] = None
    layer_recompute_mask: Optional[List[bool]] = None  # per-segment override


@dataclass
class QualitySignals:
    cosine_similarity: float
    reuse_ratio: float
    confidence_tier: str              # "exact" | "fuzzy" | "recompute"
    passed_quality_gate: bool
    rejection_reason: Optional[str] = None
```

### 4.1 `_correct_fuzzy_kv_rope` changes

```python
def _correct_fuzzy_kv_rope(self, forward_batch, req):
    result = req.fuzzy_match_result

    if result.segments is None:
        # existing Chenxin path: single contiguous span
        self._rope_correct_contiguous(req, result)
    else:
        # new semantic path: iterate segments
        for seg in result.segments:
            donor_locs = self._resolve_segment_kv(seg)   # NodeRef -> tensor
            self._rope_correct_segment(req, seg, donor_locs)

    if result.layer_recompute_mask is not None:
        req.layers_to_recompute = [
            i for i, m in enumerate(result.layer_recompute_mask) if m
        ]
```

`_resolve_segment_kv` is one dict lookup + one tensor slice; resolution failure (donor evicted despite our lock_ref) returns None and the segment is skipped with a warning. `_rope_correct_segment` is the same allocator + reverse-RoPE + apply-RoPE sequence Chenxin already wrote, scoped to one segment's `donor_positions → target_positions`. The RoPE delta math is identical.

### 4.2 Scheduler / forward_extend changes

`forward_batch_info.py` already plumbs `req.fuzzy_match_result.layer_recompute_mask` and `req.fuzzy_match_result.segments` onto the `ForwardBatch` (`fuzzy_layer_recompute_mask`, `fuzzy_segments`). `model_runner._correct_fuzzy_kv_rope_segments` consumes them. `rope_correction.py:_copy_kv_with_rope_correction` zeroes layers flagged by the mask before reapplying the new positions' RoPE — see § 9 below for the answer to "does this exist?"

## 5. `SemanticEmbeddingProvider` — implementation sketch

### 5.1 SGLang side (thin wrapper, in the SGLang PR)

```python
# python/sglang/srt/mem_cache/fuzzy_match/semantic_embedding.py

class SemanticEmbeddingProvider(FuzzyMatchProvider):
    def __init__(self, config: FuzzyMatchConfig):
        super().__init__(config)
        try:
            from semblend.integration.sglang.provider import SemBlendProviderAdapter
            from semblend.integration.sglang.config import SemBlendProviderConfig
        except ImportError as e:
            raise ImportError(
                "fuzzy_match_provider='SemanticEmbedding' requires the "
                "`semblend` package. Install with: pip install semblend"
            ) from e
        self._adapter = SemBlendProviderAdapter(SemBlendProviderConfig.from_dict(...))
```

The SGLang PR picks up a runtime dependency on `semblend` only when `fuzzy_match_provider="SemanticEmbedding"` is set. We import lazily so SGLang users who don't enable semantic matching never load semblend.

### 5.2 SemBlend side (heavy lift, in `semblend-release`)

`semblend/integration/sglang/provider.py`:

```python
class SemBlendProviderAdapter:
    """Adapts SemBlendPipeline to SGLang's FuzzyMatchProvider contract.

    Process-local: in-process MiniLM embedding, numpy donor store. No
    service dependency.
    """

    def __init__(self, config):
        from semblend_core.pipeline import SemBlendPipeline
        from semblend_core.bathtub import RecomputeConfig

        self._pipeline = SemBlendPipeline(
            max_donors=config.max_entries,
            min_similarity=config.min_similarity,
            min_reuse_ratio=config.min_reuse_ratio,
            embedder_type="onnx-gpu" if config.embedding_use_gpu else "minilm",
            model_name=config.model_arch,          # bathtub preset selector
            chunk_size=config.block_size,
            recompute_config=RecomputeConfig.from_env(),
        )

    def register_donor(self, request_id, token_ids, kv_cache, start, end, *,
                        prompt_text=None, radix_tree=None):
        # embed + insert into DonorStore + record _DonorKVHandle
        ...

    def on_donor_inserted(self, request_id, donor_last_node_id):
        # Called by RadixCache.cache_finished_req after insert. Records the
        # donor's TreeNode id so match results can surface it for inc_lock_ref.
        ...

    def match(self, prompt_token_ids, already_matched_len, *, prompt_text=None):
        result = self._pipeline.find_donor(...)
        if not result.found:
            return None
        segments = self._build_segments(result)         # NodeRef-based
        layer_mask = self._build_layer_mask(result)     # bathtub
        return FuzzyMatchResult(
            cached_token_count=...,
            cached_token_ids=result.donor_tokens,
            prompt_token_count=...,
            kv_cache_indices=...,         # legacy, single-segment fallback
            position_offset=already_matched_len,
            cached_start_pos=...,
            segments=segments if len(segments) > 1 else None,
            layer_recompute_mask=layer_mask,
            donor_last_node_id=handle.last_node_id,
            quality_signals=QualitySignals(
                cosine_similarity=result.similarity,
                reuse_ratio=result.reuse_ratio,
                confidence_tier=result.confidence_tier,
                passed_quality_gate=True,
            ),
        )
```

### 5.3 Module reuse — nothing new in the hot path

- `semblend_core/pipeline.py:107` — `SemBlendPipeline` (5-stage orchestrator, `find_donor` at line ≈252)
- `semblend_core/pipeline.py:84` — `PipelineResult` already carries `slot_actions`, `layer_deviations`, `position_map`, `composite_plan`, `multi_donor_position_map`
- `semblend_core/embedder.py` — `MiniLMEmbedder` (CPU/GPU auto-detect, ~3–5 ms)
- `semblend_core/donor_store.py:60` — `DonorStore` with numpy cosine search, LRU eviction, chunk index
- `semblend_core/alignment.py` — chunk-hash + fuzzy + Levenshtein
- `semblend_core/bathtub.py:99` — `PRESETS` (llama, qwen2.5-7b, qwen2.5-1.5b, default) and `compute_layer_deviations` at line 233
- `semblend_core/partial_attention.py` — `build_attention_plan` produces `PartialAttentionPlan` → we translate to `List[FuzzyMatchSegment]`
- `semblend_core/multi_donor_alignment.py` — already handles multi-donor for the `composite_plan` path

## 6. Quality floor — evidence this won't regress accuracy

Chenxin's concern in his message is accuracy degradation in his experimental `TokenBlockMatchProvider`. SemBlend's paper addresses this directly:

- **PPL ratio** (paper § 4.3, Table 7, primary quality metric): 11 of 12 measurements within 3% of 1.0; range 0.988–1.042. Target: PPL ratio ≤ 1.065.
- **Variation-sensitivity test** (paper Figure 4): PPL = 1.000 ± 0.001 across EXACT, REORDER, PARTIAL_40, PARTIAL_80, PARAPHRASE, DIVERSE, ENTITY_SWAP. This is the quality floor guarantee — whenever an injection occurs, the bathtub + chunk-hash gate keeps PPL at baseline.
- **Ablations** (paper Table 9): at τ=0.40, rare bad hits appear (PPL up to 1.27); at τ=0.60, never observed. We ship τ=0.60 as the default.

We reproduce these numbers on SGLang specifically (not just on vLLM/LMCache where the paper measured them) as part of validation — see § 8.

## 7. Configuration schema

```python
@dataclass
class FuzzyMatchConfig:
    # Existing Chenxin fields (unchanged defaults)
    enable_fuzzy_match: bool = False
    fuzzy_min_match_length: int = 128
    fuzzy_semantic_threshold: float = 0.60   # cosine τ (paper § 3.1)
    fuzzy_match_provider: str = "TokenBlockMatch"   # or "SemanticEmbedding"
    cache_fuzzy_results: bool = True
    fuzzy_non_prefix_max_entries: int = 10000
    fuzzy_block_size: int = 32
    embedding_model_name: str = "all-MiniLM-L6-v2"

    # NEW for SemanticEmbeddingProvider
    embedding_use_gpu: bool = True           # MiniLMEmbedder auto-detects if False
    model_arch: Optional[str] = None         # "llama" | "qwen2.5-7b" | ... (bathtub preset)
    enable_bathtub: bool = True              # disabled silently if model_arch is None (warn once)
    fuzzy_top_k: int = 5
    fuzzy_min_reuse_ratio: float = 0.50      # paper § 3.3 default
    quality_gate_ppl_threshold: float = 1.065  # informational; used by monitoring
    discovery_only: bool = False             # measure-only mode; useful when an upstream
                                              # RadixCache has the pool-leak bug fixed in
                                              # commit ec4c41e of this branch
```

## 8. Validation plan (before the PR is opened)

Per our internal process, the PR to `ibifrost/sglang` is blocked on benchmarks. Results are committed under `autoresearch-semblend/benchmarks/results/v0.3.x/` and linked from the PR description.

Acceptance criteria:
- n ≥ 200 per dataset, 4K / 8K / 16K token lengths
- PPL ratio ≤ 1.065 with 95% CIs
- ROUGE-L and QA-hallucination included alongside PPL
- Hit rate ≥ 40% at 8K (paper § 4.3 floor)
- TTFT improvement ≥ 2× vs cold prefill at 8K (paper Table 5)
- `TokenBlockMatchProvider` baseline unchanged — no regression on Chenxin's existing tests
- Fresh pod per length, warmup included, bench-runner in-cluster (no port-forward for 16K)

See § 12 for the runnable recipe.

## 9. Decisions on the four open questions

These were drafted as open questions in an earlier revision; below are the chosen answers grounded in the current branch state.

### 9.1 Layer recompute micro-pass — **already exists on our branch**

`req.layers_to_recompute` is implemented as `forward_batch.fuzzy_layer_recompute_mask` (`forward_batch_info.py:447`), populated from `req.fuzzy_match_result.layer_recompute_mask` at batch-prep time. `model_runner._correct_fuzzy_kv_rope_segments` consumes it (`model_runner.py:2864`), and `rope_correction.py:_copy_kv_with_rope_correction` zeroes flagged layers' K/V before reapplying the new positions' RoPE (`rope_correction.py:97-98`). This was added in our SemanticEmbedding feature commit (`fa226d8`); the original Chenxin baseline (`41003a4`) does not include it. We ship it as part of the SemanticEmbedding PR.

### 9.2 Segment indices representation — **NodeRef**

`FuzzyMatchSegment.donor_kv_indices` is preserved as a legacy field for `TokenBlockMatchProvider`'s contiguous-span path (which has no donor-lifetime concerns). New providers populate `donor_node_id` + `donor_offset` + `length` instead.

Reasoning: Chenxin's design principle #4 ("no double-counting") puts the radix tree as the single owner of pool indices. Storing pool-index tensors directly inside `FuzzyMatchSegment` violates this — the same indices end up in both the donor's `TreeNode.value` *and* the segment's tensor, and a donor eviction silently invalidates the segment. NodeRef makes the dependency structural: at consume time the model_runner resolves `radix_tree._node_registry[donor_node_id].value[donor_offset:donor_offset+length]`, fail-fast if the node is gone, and the donor's lifetime is the slots' lifetime. Cost on the hot path is one dict lookup and one slice — microseconds. The natural pairing with `donor_last_node_id` inc_lock_ref protection (§ 4 / commit `ec4c41e`) closes the loop: lock the node at match time, resolve at consume time, dec_lock at cache_finished_req.

### 9.3 Provider registry — **setuptools entry points**

`fuzzy_match/__init__.py` keeps a small built-in factory for `TokenBlockMatch` and `SemanticEmbedding` (so SGLang has a sensible default install with no external entry points), and additionally consults the `sglang.fuzzy_match_providers` entry-point group:

```toml
# pyproject.toml in third-party packages (e.g. semblend)
[project.entry-points."sglang.fuzzy_match_providers"]
SemanticEmbedding = "semblend.integration.sglang.semantic_embedding:SemanticEmbeddingProvider"
```

A third party can register `MyProvider` without patching SGLang. The factory checks built-ins first, then walks discovered entry points; collision raises a clear error.

### 9.4 Bathtub when `model_arch` is unknown — **log warning, skip bathtub**

If `enable_bathtub=True` but `model_arch` is unset (or unrecognized), the adapter logs a single warning at startup and then disables bathtub for that provider lifetime (`layer_recompute_mask` is always `None`, all layers reuse). We do *not* fall back to `PRESETS["default"]` because conservative-middle-layer reuse without calibration risks the exact accuracy regression Chenxin is concerned about. We do *not* refuse to start because that's a worse user experience than degrading to "fuzzy without bathtub" which is still better than no fuzzy at all. Operators who want bathtub on a new model run the offline calibration procedure (`semblend_core/bathtub_calibrate.py`) and pass the resulting σ values via `model_arch` or a custom preset file.

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Our segments path touches `_correct_fuzzy_kv_rope` which is actively iterating | PR atop current feature-branch HEAD; land interface extensions in a small first commit, provider in a second |
| Bathtub calibration unknown for a given model | Disable bathtub + warn (§ 9.4); operator opts in to `enable_bathtub` per deployment |
| `TokenBlockMatchProvider` regressions | All new fields are `Optional`; the contiguous path is byte-identical when `segments is None`. CI runs Chenxin's existing `test_fuzzy_match` suite unchanged. |
| Donor evicted while a recipient request is consuming its KV (pool-leak SIGQUIT) | `donor_last_node_id` plumbed through to `RadixCache.match_prefix`, which `inc_lock_ref`s the donor TreeNode; `dec_lock_ref` paired in `cache_finished_req`. Validated by 398/596 clean samples on the 2026-04-29 A10G run (commit `ec4c41e` / `ace900c`). See § 12. |
| Cumulative pool counter drift detected at first idle window (overlap-overcount, currently open) | Workaround: serve one dataset per server lifetime in benchmarks. Long-term fix: walk donor `inc_lock_ref` only up to the LCA of donor and recipient `last_node` to avoid double-walking shared ancestors. Tracked separately. |
| vLLM-calibrated bathtub numbers don't translate to SGLang | Validation (§ 8) is SGLang-specific; won't ship if PPL ratio > 1.065 on SGLang |

## 11. Out of scope (v1)

- CacheBlend-style segment-based matching (Chenxin's doc explicitly leaves this out; we agree).
- Cross-tenant donor sharing (handled by the `extra_key` namespace already in the ABC).
- Speculative decoding integration — orthogonal, future work.
- Automated bathtub calibration for new models — offline procedure, not in the provider.
- Fleet-level cross-instance donor sharing — process-local SemBlend is the v1; multi-instance is a future research direction.

## 12. Reproducibility — running the SemBlend bench yourself

Everything below assumes a single-node EKS cluster with 1 GPU and the autoresearch-semblend repo checked out alongside this one. Image builds happen in CI; you do not build locally.

### 12.1 One-shot reproduction (recommended)

```bash
cd autoresearch-semblend

# Trigger the cloud build (writes to ECR as sglang-semblend:<sha>)
gh workflow run build-sglang-semblend.yml \
  --ref feature/semblend-sglang-integration \
  -f semblend_sha=<this-repo-sha> \
  -f sglang_sha=<sglang-fork-sha>

# Wait for the build, then run the bench
SEMBLEND_SHA=<this-repo-sha> SGLANG_SHA=<sglang-fork-sha> \
  ./benchmarks/scripts/run_sglang_semblend.sh \
    --gpu a100 \
    --model qwen2.5-7b-instruct \
    --datasets triviaqa,longeval_4096,scbench \
    --variants vanilla,tokenblock,semblend \
    --n-per-dataset 200 \
    --restart-between-datasets   # works around the pending overlap-overcount fix

# Results land in benchmarks/results/v0.3.x/<run-tag>/COMPARISON.md
```

`run_sglang_semblend.sh` deploys a fresh SGLang pod via Helm, runs the bench-runner pod in-cluster, captures TTFT / PPL / hit rate / CI per variant, tears down the pod, and emits a single `COMPARISON.md`. `--restart-between-datasets` cuts the server between datasets so the cumulative pool drift hypothesized in § 10 row 5 cannot accumulate across dataset boundaries.

### 12.2 Server flags reference

```bash
# baseline (RadixCache exact-match only)
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --port 8000

# TokenBlockMatch (Chenxin's reference)
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --enable-fuzzy-match \
  --fuzzy-match-provider TokenBlockMatch \
  --port 8000

# SemBlend semantic
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --enable-fuzzy-match \
  --fuzzy-match-provider SemanticEmbedding \
  --fuzzy-semantic-threshold 0.60 \
  --fuzzy-min-reuse-ratio 0.50 \
  --fuzzy-min-match-length 128 \
  --fuzzy-non-prefix-max-entries 10000 \
  --fuzzy-block-size 32 \
  --embedding-use-gpu \
  --model-arch qwen2.5-7b \
  --enable-bathtub \
  --port 8000
```

### 12.3 Environment expectations

- 1× A100-40GB or larger (A10G works for signal-only runs; not paper-grade).
- `pip install semblend` (this repo, when SemanticEmbedding is selected; otherwise no dependency).
- HuggingFace token mounted as `HF_TOKEN` so dataset pulls don't get rate-limited.
- bench-runner pod runs in-cluster — no port-forwarding for 8K+ contexts.

### 12.4 Latest validation snapshot

Most recent reference run: A10G + Qwen2.5-1.5B-Instruct, 2026-04-29 (`autoresearch-semblend@36f6498:benchmarks/results/v0.3.x/a10g-leakfix-20260429-183007/COMPARISON.md`).

| Variant | Dataset | N served | HIT% | Mean speedup |
|---|---|---|---|---|
| Vanilla (RadixCache only) | triviaqa | 200/200 | 1.0% | 1.03× |
| TokenBlockMatch | triviaqa | 200/200 | 1.5% | 1.03× |
| **SemBlend** | **triviaqa** | **200/200** | **13.0% [8.5, 18.0]** | **1.25× [1.09, 1.46]** |
| Vanilla | longeval_4096 | 198/198 | 0.0% | 0.17× (harness artifact) |
| TokenBlockMatch | longeval_4096 | 75/198 | 0.0% | crashed |
| **SemBlend** | **longeval_4096** | **198/198** | **20.2% [14.6, 25.8]** | **1.20× [0.70, 1.85]** |

Best individual hit: triviaqa sample 81, **14.35× TTFT speedup** (cold 7705 ms → warm 537 ms). Hit-only mean speedup across both datasets: 1.79×. Paper-grade A100 + Qwen-7B numbers are pending GPU capacity.

---

*Questions or concerns — reply in the Slack thread or comment on this doc.*

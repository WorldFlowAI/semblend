# Semantic KV Cache Reuse for SGLang — Provider Design

**Status**: Design draft — response to ibifrost/SGLang `Draft_Prefix_Matching.md` (Chenxin Wu, 2026-04-22).
**Owners**: Zach Bennett (semblend), Chenxin Wu (SGLang), Zhangheng.
**Linked artifacts**:
- SGLang draft: https://github.com/ibifrost/sglang/blob/feature/support_fuzzy_prefix_match_opensource/Draft_Prefix_Matching.md
- SemBlend paper: `autoresearch-semblend/paper/semblend.pdf`
- Existing SemBlend SGLang prototype: `semblend/integration/sglang/radix_backend.py`

## 1. Summary

Chenxin's `FuzzyMatchProvider` ABC is exactly the right shape to plug SemBlend into SGLang. We're proposing **three additive, optional extensions** to `FuzzyMatchResult` so that a semantic provider can express what the paper actually does: (a) N:M and non-contiguous token alignment across possibly multiple donors, (b) per-layer selective recomputation (the bathtub curve), and (c) an explicit quality-signals contract. `TokenBlockMatchProvider` is untouched — all new fields default to `None` and `_correct_fuzzy_kv_rope` falls back to the existing contiguous-segment flow when they are absent.

The semantic provider itself, `SemanticEmbeddingProvider`, is a thin wrapper in SGLang that delegates to `semblend.integration.sglang.provider.SemBlendProviderAdapter` in the `semblend` pip package. All pipeline machinery (embedding, ANN search, alignment, bathtub curve, partial-attention planning) lives in `semblend_core/` today and is reused unchanged.

## 2. Alignment with Chenxin's design principles

| Principle | How we preserve it |
|---|---|
| **Non-pollution** of the exact-match RadixTree | Semantic provider never touches `root_node` or `TreeNode.value`. All state lives in its own in-process store (or optional Synapse Gateway service). |
| **Config-driven** opt-in | `fuzzy_match_provider="SemanticEmbedding"` plus new fields on `FuzzyMatchConfig`. Disabled → zero cost. |
| **Position-aware** RoPE correction | Reuse the exact `_correct_fuzzy_kv_rope` allocator path. For N:M segments we iterate per-segment with the same reverse-then-reapply kernel. |
| **No double-counting** of pool indices | Semantic provider stores tuples of (donor_req_id, node_id, offset, length) — same NodeRef pattern Chenxin already designed into `NonPrefixKVStore`. Pool indices only live in the radix tree. |

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
    kv_cache_indices: torch.Tensor
    position_offset: int
    cached_start_pos: int = 0
    _match_entry: Any = None

    # --- NEW (optional, provider-dependent) ---
    segments: Optional[List["FuzzyMatchSegment"]] = None
    layer_recompute_mask: Optional[List[bool]] = None
    quality_signals: Optional["QualitySignals"] = None


@dataclass
class FuzzyMatchSegment:
    """One contiguous span of matched tokens (may be part of an N:M plan).

    When present, supersedes kv_cache_indices + cached_start_pos. Multiple
    segments may reference different donors (multi-donor scatter).
    """
    donor_kv_indices: torch.Tensor   # pool indices on donor side
    target_positions: torch.Tensor   # absolute positions in the new prompt
    donor_positions: torch.Tensor    # source positions in the donor (for RoPE delta)
    donor_req_id: Optional[str] = None       # identifies the donor (for multi-donor)
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
            self._rope_correct_segment(req, seg)

    if result.layer_recompute_mask is not None:
        req.layers_to_recompute = [
            i for i, m in enumerate(result.layer_recompute_mask) if m
        ]
```

`_rope_correct_segment` is the same allocator + reverse-RoPE + apply-RoPE sequence Chenxin already wrote, scoped to one segment's `donor_positions → target_positions`. The RoPE delta math is identical.

### 4.2 Scheduler / forward_extend changes

When `req.layers_to_recompute` is non-empty, after the normal `forward_extend` completes, run a second micro-pass on flagged layers only. The matched tokens on those layers bypass KV injection and attend fresh. Middle layers still reuse. This is mechanically similar to the split-layer support SGLang already has for some multi-pass scheduling patterns; we need to verify it exists on Chenxin's branch (Open Question #1).

## 5. `SemanticEmbeddingProvider` — implementation sketch

### 5.1 SGLang side (thin wrapper, in the SGLang PR)

```python
# python/sglang/srt/mem_cache/fuzzy_match/semantic_embedding.py

class SemanticEmbeddingProvider(FuzzyMatchProvider):
    def __init__(self, config: FuzzyMatchConfig):
        super().__init__(config)
        from semblend.integration.sglang.provider import SemBlendProviderAdapter
        self._adapter = SemBlendProviderAdapter(config)

    def cache_on_request_finished(self, request, token_ids, kv_cache,
                                   cache_start_pos, cache_end_pos, radix_tree=None):
        return self._adapter.register_donor(
            request, token_ids, kv_cache,
            cache_start_pos, cache_end_pos, radix_tree,
        )

    def match_on_prefix_miss(self, prompt_token_ids, already_matched_len, extra_key=None):
        return self._adapter.match(prompt_token_ids, already_matched_len, extra_key)
```

The SGLang PR picks up a runtime dependency on `semblend` only when `fuzzy_match_provider="SemanticEmbedding"` is set. We import lazily so SGLang users who don't enable semantic matching never load semblend.

### 5.2 SemBlend side (heavy lift, in `semblend-release`)

`semblend/integration/sglang/provider.py` (new file):

```python
class SemBlendProviderAdapter:
    """Adapts SemBlendPipeline to SGLang's FuzzyMatchProvider contract."""

    def __init__(self, config):
        from semblend_core.pipeline import SemBlendPipeline
        from semblend_core.bathtub import RecomputeConfig

        self._pipeline = SemBlendPipeline(
            max_donors=config.fuzzy_non_prefix_max_entries,
            min_similarity=config.fuzzy_semantic_threshold,
            min_reuse_ratio=config.fuzzy_min_reuse_ratio,
            embedder_type="onnx-gpu" if config.embedding_use_gpu else "minilm",
            model_name=config.model_arch,          # bathtub preset selector
            chunk_size=config.fuzzy_block_size,
            recompute_config=RecomputeConfig.from_env(),
        )
        self._backend_mode = config.embedding_backend     # "local" | "gateway"
        if self._backend_mode == "gateway":
            from semblend.integration.sglang.gateway_client import SynapseGatewayClient
            self._gateway = SynapseGatewayClient(config.gateway_url)

    def register_donor(self, req, token_ids, kv_cache, start, end, radix_tree):
        # Extract NodeRefs from the radix tree for this request's tokens
        # (mirrors Chenxin's _create_node_refs_from_tree; avoids double-counting)
        ...
        return self._pipeline.donor_store.add_donor(donor_node)

    def match(self, prompt_token_ids, already_matched_len, extra_key):
        remaining = prompt_token_ids[already_matched_len:]
        result = self._pipeline.find_donor(
            token_ids=remaining,
            prompt_text=self._decode_tokens(remaining),
            top_k=self._top_k,
        )
        if not result.found:
            return None

        segments = self._build_segments(result)         # from composite_plan + position_map
        layer_mask = self._build_layer_mask(result)     # from bathtub LayerDeviations
        return FuzzyMatchResult(
            cached_token_count=len(segments[0].target_positions) if segments else 0,
            cached_token_ids=result.donor_tokens,
            prompt_token_count=...,
            kv_cache_indices=segments[0].donor_kv_indices if len(segments)==1 else torch.empty(0),
            position_offset=already_matched_len,
            cached_start_pos=int(result.position_map.donor_positions[0]),
            segments=segments if len(segments) > 1 else None,
            layer_recompute_mask=layer_mask,
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

## 6. Storage: local vs gateway

`FuzzyMatchConfig.embedding_backend`:

- **`"local"` (default)**: embed + ANN search in-process, donor metadata in a numpy `DonorStore`. Matches Chenxin's `TokenBlockMatchProvider` deployment model. Target: single-node workloads up to ~10K donors.
- **`"gateway"`**: HTTPS/gRPC call to Synapse Gateway (existing Rust service, cuVS CAGRA index, 1M+ donors). The gateway returns candidate donor IDs; KV fetch still happens via the chunk-hash → pool-index path on the inference node. Target: fleet-level multi-instance deployments.

Both modes use the identical `FuzzyMatchResult` shape. The gateway mode is async with a 3ms deadline and falls back to local on timeout so it never blocks the match path.

## 7. Quality floor — evidence this won't regress accuracy

Chenxin's concern in his message is accuracy degradation in his experimental `TokenBlockMatchProvider`. SemBlend's paper addresses this directly:

- **PPL ratio** (paper § 4.3, Table 7, primary quality metric): 11 of 12 measurements within 3% of 1.0; range 0.988–1.042. Target: PPL ratio ≤ 1.065.
- **Variation-sensitivity test** (paper Figure 4): PPL = 1.000 ± 0.001 across EXACT, REORDER, PARTIAL_40, PARTIAL_80, PARAPHRASE, DIVERSE, ENTITY_SWAP. This is the quality floor guarantee — whenever an injection occurs, the bathtub + chunk-hash gate keeps PPL at baseline.
- **Ablations** (paper Table 9): at τ=0.40, rare bad hits appear (PPL up to 1.27); at τ=0.60, never observed. We ship τ=0.60 as the default.

We will reproduce these numbers on SGLang specifically (not just on vLLM/LMCache where the paper measured them) as part of validation — see § 9.

## 8. Configuration schema (proposed)

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
    embedding_backend: str = "local"         # "local" | "gateway"
    gateway_url: Optional[str] = None
    gateway_timeout_ms: int = 3
    embedding_use_gpu: bool = True           # MiniLMEmbedder auto-detects if False
    model_arch: Optional[str] = None         # "llama" | "qwen2.5-7b" | ... (bathtub preset)
    enable_bathtub: bool = True
    fuzzy_top_k: int = 5
    fuzzy_min_reuse_ratio: float = 0.50      # paper § 3.3 default
    quality_gate_ppl_threshold: float = 1.065  # (informational; used by monitoring)
```

## 9. Validation plan (before the PR is opened)

Per our internal process, the PR to `ibifrost/sglang` is blocked on benchmarks. Results will be committed to `autoresearch-semblend/benchmarks/results/v0.2.0/sglang_semantic_fuzzy.json` and linked from the PR description.

- Build SGLang+SemBlend image via `autoresearch-semblend/.github/workflows/build-sglang-semblend.yml` (new workflow) — cloud build only, per project policy.
- Deploy on A100 EKS via `infra/values-autoresearch-sglang.yaml` (new values file).
- Run `python -m benchmarks.suite.reproduce --priority p0 --engine sglang --provider semantic` against the existing paper-table harness.

Acceptance criteria:
- n ≥ 200 per dataset (CNN/DailyMail, XSum, MultiNews, WikiHow), 4K / 8K / 16K token lengths
- PPL ratio ≤ 1.065 with 95% CIs
- ROUGE-L and QA-hallucination included alongside PPL
- Hit rate ≥ 40% at 8K (paper § 4.3 floor)
- TTFT improvement ≥ 2× vs cold prefill at 8K (paper Table 5)
- `TokenBlockMatchProvider` baseline unchanged — no regression on Chenxin's existing tests
- Fresh pod per length, warmup included, bench-runner in-cluster (no port-forward for 16K)

## 10. Open questions for Chenxin

1. **Layer recompute micro-pass** — does `req.layers_to_recompute` (or equivalent) already exist in `model_runner` on the feature branch? If not, this is a small addition to `forward_extend` we'd include in the PR.
2. **Segment indices representation** — should `FuzzyMatchSegment.donor_kv_indices` be pool indices (like the current `kv_cache_indices` tensor) or a NodeRef list (like the `NonPrefixKVStore` entries you already designed)? NodeRef is cleaner for bookkeeping but adds a resolve step on the hot path.
3. **Provider registry** — do you want the factory in `fuzzy_match/__init__.py` to hard-code providers, or should we use setuptools entry-points so third parties can register providers without touching SGLang source? (We're agnostic; happy to go with whatever you prefer.)
4. **Bathtub when `model_arch` is unknown** — if the operator hasn't set `model_arch`, do we want to disable bathtub silently, emit a warning, or refuse to start? Defaulting to `PRESETS["default"]` is safe (conservative middle-layer reuse) but not optimal.

## 11. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Our segments path touches `_correct_fuzzy_kv_rope` which is actively iterating | PR atop current feature-branch HEAD; land interface extensions in a small first commit, provider in a second |
| Bathtub calibration unknown for a given model | Fall back to `PRESETS["default"]` + warn; operator opts in to `enable_bathtub` per deployment |
| Gateway mode adds a network hop | `gateway_timeout_ms=3`, fall back to local on timeout; never blocks match path |
| `TokenBlockMatchProvider` regressions | All new fields are `Optional`; the contiguous path is byte-identical when `segments is None`. CI runs Chenxin's existing `test_fuzzy_match` suite unchanged. |
| vLLM-calibrated bathtub numbers don't translate to SGLang | Validation (§ 9) is SGLang-specific; won't ship if PPL ratio > 1.065 on SGLang |

## 12. Out of scope (v1)

- CacheBlend-style segment-based matching (Chenxin's doc explicitly leaves this out; we agree).
- Cross-tenant donor sharing (handled by the `extra_key` namespace already in the ABC).
- Speculative decoding integration — orthogonal, future work.
- Automated bathtub calibration for new models — offline procedure, not in the provider.

---

*Questions or concerns — reply in the Slack thread or comment on this doc.*

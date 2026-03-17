"""SemBlend Connector — Semantic KV-Donor Discovery for vLLM.

Extends LMCache's KV connector with semantic donor discovery.
When LMCache's chunk-hash lookup misses (no exact prefix match),
SemBlend finds semantically similar cached prompts and injects the
donor's KV via LMCache's CacheBlend recomputation pipeline.

Architecture:
    vLLM scheduler
        → SemBlendConnectorV1.get_num_new_matched_tokens()
            → LMCache normal lookup (chunk-hash)
            → on miss: SemBlend semantic donor search
                → find best donor via token-set Jaccard or embedding cosine
                → monkeypatch LMCache lookup to return donor's hit count
                → mark request for CacheBlend recomputation
        → LMCache start_load_kv() loads DONOR's KV
        → CacheBlend blender selectively recomputes deviated layers

Usage:
    --kv-transfer-config '{
        "kv_connector": "SemBlendConnectorV1",
        "kv_connector_module_path": "synapse_kv_connector.semblend_connector",
        "kv_role": "kv_both"
    }'

    Env vars:
        SEMBLEND_ENABLED=1              Enable semantic donor search
        SEMBLEND_GATEWAY_URL=...        Synapse gateway for embeddings
        SEMBLEND_MIN_SIMILARITY=0.60    Min Jaccard/cosine for donors
        SEMBLEND_DISABLE_ROPE_CORRECTION=1  Skip RoPE correction (ablation only)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = logging.getLogger("synapse.semblend")
# Ensure our logs reach stdout (vLLM may not propagate our logger)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


@dataclass
class KVFingerprint:
    """Per-layer KV cache fingerprint for deviation prediction.

    For each transformer layer, stores a lightweight summary of the
    K matrix that can predict KV deviation without full recomputation.
    Uses L2 norm of the K matrix per layer as a cheap but effective
    proxy for the KV state.

    Option C from the SemBlend architecture: per-layer KV fingerprints
    encode the contextual history per layer, directly addressing the
    history dependence problem.
    """

    layer_norms: list[float]
    """L2 norm of K matrix per layer — captures scale of KV state."""

    layer_mean_keys: list[list[float]]
    """Mean K vector per layer (head_dim floats) — captures direction."""


@dataclass
class DonorEntry:
    """Cached donor prompt for semantic matching."""

    request_id: str
    token_ids: list[int]
    prompt_text: str
    embedding: list[float]
    timestamp: float
    num_tokens: int
    kv_fingerprint: KVFingerprint | None = None


@dataclass
class LayerDeviation:
    """Predicted per-layer deviation between donor and query."""

    layer_idx: int
    norm_distance: float
    direction_similarity: float
    should_recompute: bool


@dataclass
class SemanticMatch:
    """Result of semantic donor search."""

    donor: DonorEntry
    similarity: float
    common_prefix_len: int
    token_overlap_ratio: float
    layer_deviations: list[LayerDeviation] | None = None


class SemBlendDonorStore:
    """In-process semantic donor index (LRU-bounded)."""

    def __init__(
        self,
        max_entries: int = 1000,
        min_similarity: float = 0.60,
        gateway_url: str | None = None,
    ) -> None:
        self._entries: OrderedDict[str, DonorEntry] = OrderedDict()
        self._max_entries = max_entries
        self._min_similarity = min_similarity
        self._gateway_url = gateway_url

    @property
    def size(self) -> int:
        return len(self._entries)

    def add_donor(
        self,
        request_id: str,
        token_ids: list[int],
        prompt_text: str,
        embedding: list[float] | None = None,
        kv_fingerprint: KVFingerprint | None = None,
    ) -> None:
        if request_id in self._entries:
            self._entries.move_to_end(request_id)
            return
        self._entries[request_id] = DonorEntry(
            request_id=request_id,
            token_ids=token_ids,
            prompt_text=prompt_text,
            embedding=embedding or [],
            timestamp=time.monotonic(),
            num_tokens=len(token_ids),
            kv_fingerprint=kv_fingerprint,
        )
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def find_donor(
        self,
        query_token_ids: list[int],
        query_embedding: list[float] | None = None,
    ) -> SemanticMatch | None:
        if not self._entries:
            return None

        best: SemanticMatch | None = None
        best_score = self._min_similarity
        query_set = set(query_token_ids)
        method_used = "none"

        for entry in self._entries.values():
            if entry.token_ids == query_token_ids:
                continue

            # Similarity: embedding cosine if available, else Jaccard
            if query_embedding and entry.embedding:
                sim = _cosine_similarity(query_embedding, entry.embedding)
                this_method = "cosine"
            else:
                entry_set = set(entry.token_ids)
                inter = len(query_set & entry_set)
                union = len(query_set | entry_set)
                sim = inter / union if union > 0 else 0.0
                this_method = "jaccard"

            if sim < best_score:
                continue

            prefix_len = _common_prefix_length(
                query_token_ids, entry.token_ids
            )
            overlap = _token_overlap_count(query_token_ids, entry.token_ids)
            ratio = overlap / max(len(query_token_ids), 1)

            best_score = sim
            method_used = this_method
            best = SemanticMatch(
                donor=entry,
                similarity=sim,
                common_prefix_len=prefix_len,
                token_overlap_ratio=ratio,
                layer_deviations=None,
            )

        if best is not None:
            print(
                f"[SemBlend] donor found via {method_used}: "
                f"sim={best.similarity:.3f}, "
                f"has_query_emb={query_embedding is not None}, "
                f"has_donor_emb={bool(best.donor.embedding)}",
                file=sys.stderr, flush=True,
            )

        return best

    def get_embedding(self, text: str) -> list[float] | None:
        """Fetch embedding from jina-v4 TEI endpoint.

        Tries direct TEI /embed first (most reliable), then gateway fallback.
        """
        if not text or len(text.strip()) < 10:
            return None

        jina_url = os.environ.get(
            "SEMBLEND_JINA_URL",
            "http://synapse-staging-jina-v4:8080",
        )

        try:
            import requests as req_lib

            # TEI /embed endpoint — expects {"inputs": ["text"]} or {"inputs": "text"}
            t0 = time.monotonic()
            resp = req_lib.post(
                f"{jina_url}/embed",
                json={"inputs": [text[:2000]]},
                timeout=10.0,
            )
            elapsed_ms = (time.monotonic() - t0) * 1000

            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and data:
                    emb = data[0] if isinstance(data[0], list) else data
                    print(
                        f"[SemBlend] embedding OK: dim={len(emb)}, "
                        f"time={elapsed_ms:.0f}ms, text={len(text)}ch",
                        file=sys.stderr, flush=True,
                    )
                    return emb
            else:
                print(
                    f"[SemBlend] embedding FAIL: status={resp.status_code}, "
                    f"url={jina_url}/embed, body={resp.text[:200]}",
                    file=sys.stderr, flush=True,
                )

            # Fallback: gateway embedding endpoint
            if self._gateway_url:
                resp = req_lib.post(
                    f"{self._gateway_url}/api/v1/embeddings",
                    json={"text": text[:2000]},
                    timeout=5.0,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    emb = data.get("embedding") or data.get("embeddings")
                    if isinstance(emb, list) and emb:
                        result = emb if isinstance(emb[0], float) else emb[0]
                        print(
                            f"[SemBlend] embedding OK (gateway): dim={len(result)}",
                            file=sys.stderr, flush=True,
                        )
                        return result

        except Exception as e:
            print(
                f"[SemBlend] embedding ERROR (remote): {e}",
                file=sys.stderr, flush=True,
            )

        # Fallback: local MiniLM via sentence-transformers
        try:
            if not hasattr(self, "_local_model"):
                from sentence_transformers import SentenceTransformer
                self._local_model = SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2"
                )
                print(
                    "[SemBlend] loaded local MiniLM-L6-v2 for embeddings",
                    file=sys.stderr, flush=True,
                )
            t0 = time.monotonic()
            emb = self._local_model.encode(
                text[:2000], convert_to_numpy=True
            ).tolist()
            elapsed_ms = (time.monotonic() - t0) * 1000
            print(
                f"[SemBlend] embedding OK (local MiniLM): dim={len(emb)}, "
                f"time={elapsed_ms:.0f}ms",
                file=sys.stderr, flush=True,
            )
            return emb
        except Exception as e2:
            print(
                f"[SemBlend] embedding ERROR (local fallback): {e2}",
                file=sys.stderr, flush=True,
            )
        return None


class SemBlendConnectorV1(KVConnectorBase_V1):
    """vLLM KV connector with semantic KV-donor discovery + CacheBlend.

    Inherits from KVConnectorBase_V1 so vLLM's isinstance() checks pass.
    Wraps LMCacheConnectorV1 and adds semantic donor search on cache miss.
    Unknown attributes are delegated to the wrapped connector via __getattr__.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: Any,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        # Initialize base class so vLLM's isinstance checks pass and
        # base attributes (_vllm_config, _role, etc.) are set
        super().__init__(vllm_config, role, kv_cache_config)

        # Initialize underlying LMCache connector
        from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import (
            LMCacheConnectorV1,
        )

        self._lmcache = LMCacheConnectorV1(
            vllm_config, role, kv_cache_config
        )

        # SemBlend config
        self._enabled = os.environ.get("SEMBLEND_ENABLED", "1") == "1"
        gateway_url = os.environ.get(
            "SEMBLEND_GATEWAY_URL",
            os.environ.get("SYNAPSE_GATEWAY_URL", ""),
        )

        # New in-process pipeline (Phase 2 refactor)
        # Falls back to legacy SemBlendDonorStore if pipeline init fails
        self._pipeline = None
        self._use_pipeline = (
            os.environ.get("SEMBLEND_USE_PIPELINE", "1") == "1"
        )
        if self._use_pipeline:
            try:
                from synapse_kv_connector.pipeline import create_vllm_pipeline
                model_name = None
                try:
                    mc = getattr(vllm_config, "model_config", None)
                    if mc:
                        model_name = getattr(mc, "model", None)
                except Exception:
                    pass
                self._pipeline = create_vllm_pipeline(
                    max_donors=int(
                        os.environ.get("SEMBLEND_MAX_DONORS", "10000")
                    ),
                    min_similarity=float(
                        os.environ.get("SEMBLEND_MIN_SIMILARITY", "0.60")
                    ),
                    min_reuse_ratio=float(
                        os.environ.get("SEMBLEND_MIN_REUSE_RATIO", "0.50")
                    ),
                    embedder_type=os.environ.get("SEMBLEND_EMBEDDER"),
                    model_name=model_name,
                )
            except Exception as e:
                print(
                    f"[SemBlend] pipeline init failed, falling back to "
                    f"legacy store: {e}",
                    file=sys.stderr, flush=True,
                )

        # Legacy donor store (fallback when pipeline unavailable)
        self._donor_store = SemBlendDonorStore(
            max_entries=int(os.environ.get("SEMBLEND_MAX_DONORS", "1000")),
            min_similarity=float(
                os.environ.get("SEMBLEND_MIN_SIMILARITY", "0.60")
            ),
            gateway_url=gateway_url or None,
        )

        # Donor token substitution map: req_id → donor_token_ids
        # Used to swap tokens during build_connector_meta so that
        # LMCache retrieves the donor's KV chunks
        self._donor_token_map: dict[str, list[int]] = {}

        # RoPE correction position maps: req_id → PositionMapping
        # When donor_pos != target_pos, K needs RoPE(delta) correction
        # after LMCache loads the donor KV into paged cache
        self._position_maps: dict[str, object] = {}

        # Track requests that used donor injection (can_save=False) —
        # these should NOT be registered as donors because their KV
        # was never saved to LMCache
        self._donor_matched_reqs: set[str] = set()

        # Per-layer KV fingerprint accumulator (Option C)
        # Collects per-layer K statistics during save_kv_layer calls
        self._pending_fingerprints: dict[str, dict[int, dict]] = {}
        self._fingerprint_enabled = (
            os.environ.get("SEMBLEND_KV_FINGERPRINT", "1") == "1"
        )

        # PartialAttention via Triton kernels (Phase 5)
        # When enabled, uses paged KV scatter + partial attention instead
        # of LMCache's CacheBlend blender for donor KV injection
        self._use_partial_attn = (
            os.environ.get("SEMBLEND_USE_PARTIAL_ATTN", "0") == "1"
        )
        self._active_hook = None  # Set during get_num_new_matched_tokens
        self._model_runner_patched = False

        # RoPE correction disable flag (for ablation benchmarks)
        # When set, skips RoPE delta correction after donor KV injection.
        # This causes quality degradation for REORDER scenarios where
        # donor tokens are at different positions than target tokens.
        self._disable_rope_correction = (
            os.environ.get("SEMBLEND_DISABLE_ROPE_CORRECTION", "0") == "1"
        )
        if self._disable_rope_correction:
            print(
                "[SemBlend] WARNING: RoPE correction DISABLED "
                "(SEMBLEND_DISABLE_ROPE_CORRECTION=1). "
                "Quality will degrade for reordered donor KV.",
                file=sys.stderr, flush=True,
            )

        self._stats = {
            "lmcache_hits": 0,
            "semblend_hits": 0,
            "semblend_misses": 0,
            "total_lookups": 0,
            "total_saves": 0,
            "partial_attn_applied": 0,
        }

        msg = (
            f"SemBlend connector initialized: enabled={self._enabled}, "
            f"min_similarity={self._donor_store._min_similarity:.2f}, "
            f"gateway={gateway_url or '(none)'}"
        )
        logger.info(msg)
        print(f"[SemBlend] {msg}", file=sys.stderr, flush=True)

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: Any,
        metric_types: Any,
        labelnames: Any,
        per_engine_labelvalues: Any,
    ) -> None:
        """Prometheus metrics — delegate to LMCache's implementation."""
        from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import (
            LMCacheConnectorV1,
        )

        return LMCacheConnectorV1.build_prom_metrics(
            vllm_config, metric_types, labelnames, per_engine_labelvalues
        )

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the wrapped LMCache connector.

        vLLM's kv_connector_model_runner_mixin accesses properties like
        prefer_cross_layer_blocks on the connector. Rather than manually
        proxying every property/method from KVConnectorBase_V1, we
        delegate to the underlying LMCacheConnectorV1 instance.
        """
        # Avoid infinite recursion during __init__ before _lmcache is set
        if name == "_lmcache":
            raise AttributeError(name)
        return getattr(self._lmcache, name)

    # ==============================
    # Metadata binding — propagate to inner LMCache connector
    # ==============================

    def bind_connector_metadata(self, connector_metadata: Any) -> None:
        """Propagate metadata to both self and inner LMCache connector.

        vLLM's model runner calls this on the outermost connector.
        LMCache's engine calls _parent._get_connector_metadata() on the
        inner LMCacheConnectorV1, so we must set it there too.
        """
        super().bind_connector_metadata(connector_metadata)
        self._lmcache.bind_connector_metadata(connector_metadata)

    def clear_connector_metadata(self) -> None:
        super().clear_connector_metadata()
        self._lmcache.clear_connector_metadata()

    # ==============================
    # Pass-through worker-side methods
    # ==============================

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        # Register the vLLM model with LMCache's VLLMModelTracker so that
        # CacheBlend's blender can access model weights for selective
        # layer recomputation. This call site is ideal because it runs
        # during gpu_model_runner.initialize_kv_cache() when self.model
        # is already loaded.
        self._register_model_for_cacheblend()

        # Patch model runner for PartialAttention (Phase 5)
        if self._use_partial_attn and not self._model_runner_patched:
            self._try_patch_model_runner()
        self._lmcache.register_kv_caches(kv_caches)

    def start_load_kv(
        self, forward_context: "ForwardContext", **kwargs: Any
    ) -> None:
        # Log what the worker side sees for active loads
        engine = getattr(self._lmcache, "_lmcache_engine", None)
        if engine is not None:
            meta = getattr(engine, "_parent", self._lmcache)
            cmeta = getattr(meta, "_connector_metadata", None)
            if cmeta is not None:
                for req in getattr(cmeta, "requests", []):
                    load_spec = getattr(req, "load_spec", None)
                    if load_spec is not None:
                        req_id = getattr(req, "request_id", getattr(req, "req_id", "?"))
                        print(
                            f"[SemBlend] WORKER start_load: req={req_id}, "
                            f"load_spec.lmcache_cached={load_spec.lmcache_cached_tokens}, "
                            f"load_spec.can_load={load_spec.can_load}, "
                            f"tok_len={len(getattr(req, 'token_ids', []))}",
                            file=sys.stderr, flush=True,
                        )
        t0 = time.monotonic()
        self._lmcache.start_load_kv(forward_context, **kwargs)
        elapsed = (time.monotonic() - t0) * 1000
        if elapsed > 5:
            print(
                f"[SemBlend] WORKER start_load_kv took {elapsed:.0f}ms",
                file=sys.stderr, flush=True,
            )
        # Apply RoPE correction AFTER LMCache loads donor KV.
        import synapse_kv_connector.semblend_connector as _mod_rope
        hook = getattr(_mod_rope, "_semblend_active_hook", None)
        if hook is not None and not hook.executed:
            self._apply_rope_after_load(hook, forward_context)

    def _apply_rope_after_load(self, hook, forward_context):
        """Apply RoPE correction after LMCache loads donor KV."""
        try:
            import time as _t
            t0 = _t.monotonic()
            mr = getattr(self, '_model_runner_ref', None)
            kv_caches = getattr(mr, 'kv_caches', None) if mr else None
            if kv_caches is None:
                print("[SemBlend] RoPE: no kv_caches", file=sys.stderr, flush=True)
                return
            num_layers = len(kv_caches)
            for li in range(num_layers):
                try:
                    self._lmcache.wait_for_layer_load(f"model.layers.{li}.self_attn")
                except Exception:
                    pass
            wait_ms = (_t.monotonic() - t0) * 1000
            block_table = None
            attn_meta = getattr(forward_context, "attn_metadata", None)
            if isinstance(attn_meta, dict):
                for _, meta in attn_meta.items():
                    bt = getattr(meta, "block_table", None)
                    if bt is None:
                        bt = getattr(meta, "block_table_tensor", None)
                    if bt is not None and isinstance(bt, torch.Tensor):
                        block_table = bt[0] if bt.dim() > 1 else bt
                        break
            if block_table is None and mr is not None:
                ib = getattr(mr, 'input_batch', None)
                if ib is not None:
                    bt = getattr(ib, 'block_table', None)
                    if bt is not None:
                        try:
                            block_table = bt[0]
                        except (IndexError, KeyError, TypeError):
                            if isinstance(bt, torch.Tensor):
                                block_table = bt
            if block_table is None:
                print("[SemBlend] RoPE: no block_table", file=sys.stderr, flush=True)
                return
            kv_list = []
            if isinstance(kv_caches, dict):
                fk = next(iter(kv_caches), "")
                fmt = "model.layers.{}.self_attn.attn" if "self_attn.attn" in fk else "model.layers.{}.self_attn"
                for li in range(num_layers):
                    k = fmt.format(li)
                    if k in kv_caches:
                        kv_list.append(kv_caches[k])
                    else:
                        break
            else:
                kv_list = list(kv_caches)
            if not kv_list:
                print("[SemBlend] RoPE: empty kv_list", file=sys.stderr, flush=True)
                return
            print(
                f"[SemBlend] RoPE: applying, bt={block_table.shape}, "
                f"layers={len(kv_list)}, kv={kv_list[0].shape}, wait={wait_ms:.0f}ms",
                file=sys.stderr, flush=True,
            )
            result = hook.apply_rope_correction(kv_list, block_table)
            total = (_t.monotonic() - t0) * 1000
            if result.get("corrected"):
                print(
                    f"[SemBlend] RoPE APPLIED: {result['layers_corrected']}/{len(kv_list)} layers, "
                    f"{result['correction_pairs']} pairs, {result['time_ms']:.1f}ms, total={total:.0f}ms",
                    file=sys.stderr, flush=True,
                )
            else:
                print(f"[SemBlend] RoPE skip: {result.get('reason')}", file=sys.stderr, flush=True)

            # SEMBLEND_FORCE_DELTA: artificially corrupt K positions to
            # simulate Δ≠0 for E2E validation of RoPE correction.
            # Reads from /tmp/semblend_force_delta.json at runtime so
            # conditions can change between requests without restart.
            force_delta = int(os.environ.get("SEMBLEND_FORCE_DELTA", "0"))
            force_correct = os.environ.get("SEMBLEND_FORCE_DELTA_CORRECT", "0").strip() == "1"
            _fd_cfg = "/tmp/semblend_force_delta.json"
            try:
                import json as _json
                with open(_fd_cfg) as _f:
                    _cfg = _json.load(_f)
                force_delta = int(_cfg.get("delta", force_delta))
                force_correct = bool(_cfg.get("correct", force_correct))
            except (FileNotFoundError, ValueError, KeyError):
                pass
            if force_delta != 0:
                from synapse_kv_connector.rope_correction import apply_rope_delta_inplace
                pos_map = hook._position_map
                num_matched = pos_map.num_pairs if hasattr(pos_map, 'num_pairs') else 0
                positions = list(range(num_matched))
                fd_t0 = _t.monotonic()
                fd_modified = 0
                for layer_kv in kv_list:
                    fd_modified += apply_rope_delta_inplace(
                        layer_kv, block_table, positions, force_delta,
                    )
                if force_correct:
                    for layer_kv in kv_list:
                        apply_rope_delta_inplace(
                            layer_kv, block_table, positions, -force_delta,
                        )
                fd_ms = (_t.monotonic() - fd_t0) * 1000
                print(
                    f"[SemBlend] FORCE_DELTA={force_delta} applied: "
                    f"{fd_modified} K positions corrupted across {len(kv_list)} layers, "
                    f"correct={force_correct}, {fd_ms:.1f}ms, "
                    f"req={hook._request_id}",
                    file=sys.stderr, flush=True,
                )
        except Exception:
            import traceback
            print(f"[SemBlend] RoPE FAILED:\n{traceback.format_exc()}", file=sys.stderr, flush=True)
        finally:
            import synapse_kv_connector.semblend_connector as _m
            _m._semblend_active_hook = None

    def wait_for_layer_load(self, layer_name: str) -> None:
        t0 = time.monotonic()
        self._lmcache.wait_for_layer_load(layer_name)
        elapsed = (time.monotonic() - t0) * 1000
        if elapsed > 10:
            print(
                f"[SemBlend] WORKER wait_for_layer {layer_name}: {elapsed:.0f}ms",
                file=sys.stderr, flush=True,
            )

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: Any,
        **kwargs: Any,
    ) -> None:
        # Capture KV fingerprint for donor registration (Option C)
        self._capture_kv_fingerprint(layer_name, kv_layer)

        # Per-layer KV deviation logging for bathtub curve calibration
        # When donor KV was loaded, compare freshly computed KV against
        # the loaded donor KV to measure actual per-layer deviation.
        if self._use_partial_attn and self._donor_matched_reqs:
            self._log_layer_deviation(layer_name, kv_layer)

        self._lmcache.save_kv_layer(
            layer_name, kv_layer, attn_metadata, **kwargs
        )

    def wait_for_save(self) -> None:
        self._lmcache.wait_for_save()
        # Flush accumulated KV fingerprints to disk for SCHEDULER to read
        if self._fingerprint_enabled:
            self._flush_fingerprints_to_disk()

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        return self._lmcache.get_finished(finished_req_ids)

    def get_block_ids_with_load_errors(self) -> set[int]:
        fn = getattr(self._lmcache, "get_block_ids_with_load_errors", None)
        result = fn() if callable(fn) else set()
        if result:
            print(
                f"[SemBlend] LOAD ERRORS in blocks: {result}",
                file=sys.stderr, flush=True,
            )
        return result

    def shutdown(self) -> None:
        fn = getattr(self._lmcache, "shutdown", None)
        if callable(fn):
            fn()

    # ==============================
    # Scheduler-side methods
    # ==============================

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        self._lmcache.update_state_after_alloc(
            request, blocks, num_external_tokens
        )

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> Any:
        # Log scheduling details for donor-matched requests
        if self._donor_token_map:
            for req in getattr(scheduler_output, "scheduled_new_reqs", []):
                req_id = getattr(req, "req_id", None)
                if req_id and req_id in self._donor_token_map:
                    sched_tokens = (
                        scheduler_output.num_scheduled_tokens.get(req_id, -1)
                        if hasattr(scheduler_output, "num_scheduled_tokens")
                        else -1
                    )
                    num_computed = getattr(req, "num_computed_tokens", -1)
                    num_total = getattr(req, "num_tokens", -1)
                    print(
                        f"[SemBlend] build_meta: req={req_id}, "
                        f"sched_tokens={sched_tokens}, "
                        f"computed={num_computed}, total={num_total}",
                        file=sys.stderr, flush=True,
                    )

        # Build metadata via LMCache
        meta = self._lmcache.build_connector_meta(scheduler_output)

        # After metadata is built, swap token IDs in the ReqMeta entries
        # for donor-matched requests so LMCache retrieves the DONOR's KV.
        # Also skip saving for these requests (donor's tokens shouldn't be
        # cached under the new request's chunk hashes).
        if self._donor_token_map and meta is not None:
            requests = getattr(meta, "requests", [])
            for req_meta in requests:
                req_id = getattr(req_meta, "request_id", None) or getattr(req_meta, "req_id", None)
                if req_id and req_id in self._donor_token_map:
                    donor_tokens = self._donor_token_map.pop(req_id)
                    original_tokens = getattr(req_meta, "token_ids", None)
                    if original_tokens is not None:
                        # Replace donor-covered portion of tokens while
                        # maintaining original length. LMCache asserts
                        # len(tokens) == len(slot_mapping), so we must
                        # not change the token list size.
                        new_tokens = list(original_tokens)
                        swap_len = min(len(new_tokens), len(donor_tokens))
                        new_tokens[:swap_len] = donor_tokens[:swap_len]
                        req_meta.token_ids = new_tokens
                        logger.info(
                            "SemBlend: swapped %d tokens in metadata "
                            "for req=%s",
                            swap_len,
                            req_id,
                        )
                    # Skip saving — the donor's KV is already cached
                    # under the donor's chunk hashes. We don't want to
                    # save under the swapped tokens.
                    save_spec = getattr(req_meta, "save_spec", None)
                    if save_spec is not None:
                        save_spec.can_save = False

        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        result = self._lmcache.request_finished(request, block_ids)

        if self._enabled:
            # Only register as donor if this request's KV was actually
            # saved to LMCache. Donor-matched requests have can_save=False
            # so their KV is NOT in LMCache's cache.
            req_id = request.request_id
            if req_id in self._donor_matched_reqs:
                self._donor_matched_reqs.discard(req_id)
                print(
                    f"[SemBlend] skip donor reg for donor-matched "
                    f"req={req_id}",
                    file=sys.stderr, flush=True,
                )
            else:
                self._register_donor(request)
                store_size = (
                    self._pipeline.donor_count
                    if self._pipeline is not None
                    else self._donor_store.size
                )
                print(
                    f"[SemBlend] registered donor req={req_id}, "
                    f"store_size={store_size}",
                    file=sys.stderr, flush=True,
                )

        return result

    # ==============================
    # SemBlend: semantic donor discovery
    # ==============================

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """Token matching with semantic donor fallback."""
        self._stats["total_lookups"] += 1
        prompt_len = len(list(request.all_token_ids))
        print(
            f"[SemBlend] get_matched: req={request.request_id}, "
            f"num_computed={num_computed_tokens}, prompt_len={prompt_len}, "
            f"call_count={self._stats['total_lookups']}",
            file=sys.stderr, flush=True,
        )

        # Step 1: Normal LMCache lookup
        lmcache_result = self._lmcache.get_num_new_matched_tokens(
            request, num_computed_tokens
        )

        if isinstance(lmcache_result, tuple):
            num_matched, do_remote = lmcache_result
        else:
            num_matched = lmcache_result
            do_remote = False

        # If LMCache found a hit, use it
        if num_matched is not None and num_matched > 0:
            self._stats["lmcache_hits"] += 1
            return (num_matched, do_remote)

        # Step 2: Semantic donor search + KV injection
        # Only trigger when BOTH LMCache and vLLM prefix cache miss
        if not self._enabled:
            return (num_matched or 0, False)

        token_ids = list(request.all_token_ids)
        prompt_len = len(token_ids)
        if prompt_len < 256:
            return (num_matched or 0, False)

        # Skip if vLLM prefix cache already matched most tokens.
        # Use 50% of prompt as threshold — if prefix cache covers more
        # than half, let vLLM handle it natively.
        prefix_threshold = max(prompt_len // 2, 256)
        if num_computed_tokens and num_computed_tokens > prefix_threshold:
            logger.debug(
                "SemBlend: skip (prefix cache hit=%d/%d), req=%s",
                num_computed_tokens,
                prompt_len,
                request.request_id,
            )
            return (num_matched or 0, False)

        # Get prompt text for embedding
        prompt = self._get_prompt_text(request)

        # --- New pipeline path (Phase 2) ---
        # Uses multi-candidate approach: get ranked candidates from pipeline,
        # try each one's KV in LMCache until we find one that's available.
        # This handles LMCache CPU buffer eviction gracefully.
        if self._pipeline is not None:
            candidates = self._pipeline.find_donor_candidates(
                token_ids=token_ids,
                prompt_text=prompt,
            )

            # Check if any candidates were found
            if not candidates or not candidates[0].found:
                self._stats["semblend_misses"] += 1
                timings_str = ""
                reason = "no_donor_match"
                if candidates:
                    timings_str = f", timings={candidates[0].timings.total_ms:.1f}ms"
                    reason = candidates[0].rejection_reason or reason
                print(
                    f"[SemBlend] MISS req={request.request_id}, "
                    f"store_size={self._pipeline.donor_count}, "
                    f"prompt_len={prompt_len}, reason={reason}{timings_str}",
                    file=sys.stderr, flush=True,
                )
                return (num_matched or 0, False)

            # Try each candidate's KV in LMCache (most recent first)
            engine_impl = self._get_lmcache_engine_impl()
            lookup_client = None
            if engine_impl is not None:
                lookup_client = getattr(engine_impl, "lookup_client", None)

            pipeline_result = None
            donor_token_ids = None

            for ci, candidate in enumerate(candidates):
                if not candidate.found:
                    continue

                # Check if this donor's KV is in LMCache
                if lookup_client is not None:
                    reqs_status = getattr(lookup_client, "reqs_status", {})
                    reqs_status.pop(request.request_id, None)
                    try:
                        kv_hit = lookup_client.lookup(
                            candidate.donor_tokens,
                            lookup_id=request.request_id,
                        )
                    except Exception:
                        kv_hit = None

                    if kv_hit is None or kv_hit <= 0:
                        print(
                            f"[SemBlend] candidate {ci+1}/{len(candidates)} "
                            f"donor={candidate.donor_id} KV NOT in cache, "
                            f"trying next...",
                            file=sys.stderr, flush=True,
                        )
                        continue

                # This candidate has KV available (or no lookup_client)
                pipeline_result = candidate
                donor_token_ids = candidate.donor_tokens
                break

            if pipeline_result is None or donor_token_ids is None:
                self._stats["semblend_misses"] += 1
                print(
                    f"[SemBlend] all {len(candidates)} candidates KV evicted, "
                    f"req={request.request_id}",
                    file=sys.stderr, flush=True,
                )
                return (num_matched or 0, False)

            self._stats["semblend_hits"] += 1
            print(
                f"[SemBlend] HIT req={request.request_id} → "
                f"donor={pipeline_result.donor_id} "
                f"(candidate {ci+1}/{len(candidates)}), "
                f"sim={pipeline_result.similarity:.3f}, "
                f"reuse={pipeline_result.reuse_ratio:.2f}, "
                f"timings={pipeline_result.timings.total_ms:.1f}ms "
                f"(emb={pipeline_result.timings.embed_ms:.1f} "
                f"lkp={pipeline_result.timings.lookup_ms:.1f})",
                file=sys.stderr, flush=True,
            )

            # Store position map for RoPE correction (used by worker-side
            # paged scatter after LMCache loads donor KV).
            # Skipped when SEMBLEND_DISABLE_ROPE_CORRECTION=1 (ablation).
            #
            # Also create hook when SEMBLEND_FORCE_DELTA is set (E2E validation):
            # artificially applies RoPE(Δ) to K after load to simulate Δ≠0,
            # then optionally corrects with RoPE(-Δ).
            _force_delta = int(os.environ.get("SEMBLEND_FORCE_DELTA", "0"))
            _need_hook = (
                (pipeline_result.position_map.needs_correction and not self._disable_rope_correction)
                or _force_delta != 0
            )
            if _need_hook:
                self._position_maps[request.request_id] = pipeline_result.position_map
                # Create RoPE hook and share with worker via module global
                try:
                    from synapse_kv_connector.model_runner_hook import RoPECorrectionHook
                    _rope_hook = RoPECorrectionHook(
                        position_map=pipeline_result.position_map,
                        plan=None,
                        request_id=request.request_id,
                    )
                    self._active_hook = _rope_hook
                    import synapse_kv_connector.semblend_connector as _mod_hook
                    _mod_hook._semblend_active_hook = _rope_hook
                except Exception as _rope_err:
                    print(f"[SemBlend] RoPE hook creation failed: {_rope_err}", file=sys.stderr, flush=True)
                _reason = "force_delta" if _force_delta != 0 else "position_correction"
                print(
                    f"[SemBlend] RoPE hook created ({_reason}): "
                    f"{pipeline_result.position_map.num_pairs} position pairs, "
                    f"force_delta={_force_delta}, req={request.request_id}",
                    file=sys.stderr, flush=True,
                )
            elif (
                pipeline_result.position_map.needs_correction
                and self._disable_rope_correction
            ):
                print(
                    f"[SemBlend] RoPE correction SKIPPED (disabled): "
                    f"{pipeline_result.position_map.num_pairs} position pairs, "
                    f"req={request.request_id}",
                    file=sys.stderr, flush=True,
                )

            # Build PartialAttention plan if enabled (Phase 5)
            # RoPE correction hook is only created when correction is enabled.
            if (
                self._use_partial_attn
                and self._model_runner_patched
                and not self._disable_rope_correction
            ):
                plan = self._pipeline.build_partial_attention_plan(
                    pipeline_result
                )
                if plan is not None:
                    try:
                        from synapse_kv_connector.model_runner_hook import (
                            RoPECorrectionHook,
                        )
                        self._active_plan = plan
                        # Create lightweight RoPE correction hook
                        # After LMCache loads donor KV, this corrects K
                        # position encoding in the paged cache in-place.
                        self._active_hook = RoPECorrectionHook(
                            position_map=pipeline_result.position_map,
                            plan=plan,
                            request_id=request.request_id,
                        )
                        import synapse_kv_connector.semblend_connector as _mod_hook2
                        _mod_hook2._semblend_active_hook = self._active_hook
                        self._stats["partial_attn_applied"] += 1
                        print(
                            f"[SemBlend] PartialAttention plan built: "
                            f"reuse={plan.num_reuse_positions}, "
                            f"partial={plan.num_partial_positions}, "
                            f"comp_ratio={plan.computation_ratio:.2f}, "
                            f"rope_pairs={pipeline_result.position_map.num_pairs}",
                            file=sys.stderr, flush=True,
                        )
                    except Exception:
                        logger.warning(
                            "PartialAttention hook creation failed",
                            exc_info=True,
                        )

            # Fall through to LMCache injection below
            # (donor_token_ids is set from pipeline result)

        # --- Legacy path (fallback) ---
        else:
            embedding = None
            if prompt:
                embedding = self._donor_store.get_embedding(prompt[:2000])
                if embedding is None:
                    print(
                        f"[SemBlend] embedding fetch returned None for "
                        f"req={request.request_id}, prompt_len={len(prompt)}ch",
                        file=sys.stderr, flush=True,
                    )
            else:
                print(
                    f"[SemBlend] no prompt text for req={request.request_id}, "
                    f"falling back to Jaccard",
                    file=sys.stderr, flush=True,
                )

            match = self._donor_store.find_donor(token_ids, embedding)
            if match is None:
                self._stats["semblend_misses"] += 1
                print(
                    f"[SemBlend] MISS req={request.request_id}, "
                    f"store_size={self._donor_store.size}, "
                    f"prompt_len={prompt_len}",
                    file=sys.stderr, flush=True,
                )
                return (num_matched or 0, False)

            self._stats["semblend_hits"] += 1
            donor = match.donor
            donor_token_ids = donor.token_ids

            # Predict per-layer deviations (Option C)
            layer_devs = self._predict_layer_deviations(
                donor.kv_fingerprint, None
            ) if donor.kv_fingerprint is not None else None

            if layer_devs is None:
                num_layers = 28
                try:
                    mc = getattr(self, "_vllm_config", None)
                    if mc:
                        mc2 = getattr(mc, "model_config", None)
                        if mc2:
                            hf = getattr(mc2, "hf_config", None)
                            if hf:
                                num_layers = getattr(
                                    hf, "num_hidden_layers", 28
                                )
                except Exception:
                    pass
                layer_devs = self._predict_layer_deviations_bathtub(num_layers)

            match = SemanticMatch(
                donor=match.donor,
                similarity=match.similarity,
                common_prefix_len=match.common_prefix_len,
                token_overlap_ratio=match.token_overlap_ratio,
                layer_deviations=layer_devs,
            )
            recompute_layers = sum(
                1 for d in match.layer_deviations if d.should_recompute
            )
            total_layers = len(match.layer_deviations)
            reuse_ratio = 1.0 - (recompute_layers / max(total_layers, 1))

            print(
                f"[SemBlend] HIT req={request.request_id} → "
                f"donor={donor.request_id}, sim={match.similarity:.3f}, "
                f"prefix={match.common_prefix_len}, "
                f"overlap={match.token_overlap_ratio:.2f}, "
                f"donor_tok={donor.num_tokens}, "
                f"kv_fp={'yes' if donor.kv_fingerprint else 'no'}, "
                f"recompute_layers={recompute_layers}/{total_layers}, "
                f"reuse_ratio={reuse_ratio:.2f}",
                file=sys.stderr, flush=True,
            )

        # Attempt to inject donor's KV via LMCache
        # donor_token_ids is set by either pipeline or legacy path above
        engine_impl = self._get_lmcache_engine_impl()
        if engine_impl is None:
            return (num_matched or 0, False)

        lookup_client = getattr(engine_impl, "lookup_client", None)
        if lookup_client is None:
            logger.warning("SemBlend: no lookup_client")
            return (num_matched or 0, False)

        try:
            # Clear stale cached result for this request
            reqs_status = getattr(lookup_client, "reqs_status", {})
            reqs_status.pop(request.request_id, None)

            # Lookup using donor's tokens — finds donor's cached KV.
            # For pipeline path, this re-lookup gets the exact hit count
            # (the candidate selection only checked > 0).
            donor_hit = lookup_client.lookup(
                donor_token_ids,
                lookup_id=request.request_id,
            )
            if donor_hit is None or donor_hit <= 0:
                print(
                    f"[SemBlend] donor KV NOT in cache (re-lookup), "
                    f"req={request.request_id}",
                    file=sys.stderr, flush=True,
                )
                return (num_matched or 0, False)

            # Cap donor_hit to fit within query prompt and align to
            # LMCache chunk boundary (256 tokens). LMCache stores KV in
            # 256-token chunks; requesting a non-aligned count causes
            # "retrieved tokens < expected" errors and falls back to
            # full recomputation.
            from synapse_kv_connector.alignment import LMCACHE_CHUNK_SIZE
            chunk_size = LMCACHE_CHUNK_SIZE
            # Leave at least one full chunk (256 tokens) for the model to
            # process fresh. Injecting right up to the prompt boundary causes
            # EOS-collapse: the model sees the donor's sentence-ending KV as
            # its context, "thinks" the answer is complete, and emits EOS
            # immediately (observed 25% collapse rate at 8K). With 256 fresh
            # tokens the model processes real tail context before generating.
            min_fresh_tokens = chunk_size
            max_usable = max(chunk_size, prompt_len - min_fresh_tokens)
            # Align to chunk boundary (floor)
            donor_hit = min(donor_hit, max_usable)
            donor_hit = (donor_hit // chunk_size) * chunk_size
            if donor_hit <= 0:
                print(
                    f"[SemBlend] donor_hit aligned to 0, skip "
                    f"(prompt={prompt_len})",
                    file=sys.stderr, flush=True,
                )
                return (num_matched or 0, False)

            print(
                f"[SemBlend] donor hit={donor_hit} tokens for "
                f"req={request.request_id} (prompt={prompt_len}, "
                f"aligned to {chunk_size}-tok chunks)",
                file=sys.stderr, flush=True,
            )

            # Create a LoadSpec so LMCache's update_state_after_alloc
            # assertion passes. The load_spec must be consistent:
            # num_external_tokens = lmcache_cached - vllm_cached - recalc
            load_specs = getattr(engine_impl, "load_specs", None)
            if load_specs is not None:
                from dataclasses import dataclass as _dc

                # Import LoadSpec from LMCache
                LoadSpec = type(next(iter(load_specs.values()))) if load_specs else None
                if LoadSpec is None:
                    # Construct LoadSpec manually
                    try:
                        from lmcache.integration.vllm.vllm_v1_adapter import LoadSpec
                    except ImportError:
                        pass

                if LoadSpec is not None:
                    load_specs[request.request_id] = LoadSpec(
                        vllm_cached_tokens=num_computed_tokens or 0,
                        lmcache_cached_tokens=donor_hit,
                        can_load=False,
                    )
                    logger.debug(
                        "SemBlend: set load_spec for req=%s: "
                        "lmcache=%d, vllm=%d",
                        request.request_id,
                        donor_hit,
                        num_computed_tokens or 0,
                    )

            # Store donor tokens for token substitution in
            # build_connector_meta. Also track that this request used
            # donor injection so we don't register it as a donor later
            # (its KV won't be saved to LMCache).
            self._donor_token_map[request.request_id] = donor_token_ids
            self._donor_matched_reqs.add(request.request_id)

            # FORCE_DELTA hook (legacy path): create hook when env var or
            # config file requests forced RoPE corruption.
            _force_delta_legacy = int(os.environ.get("SEMBLEND_FORCE_DELTA", "0"))
            try:
                import json as _json_fd
                with open("/tmp/semblend_force_delta.json") as _ffd:
                    _fd_cfg = _json_fd.load(_ffd)
                _force_delta_legacy = int(_fd_cfg.get("delta", _force_delta_legacy))
            except (FileNotFoundError, ValueError, KeyError):
                pass
            if _force_delta_legacy != 0:
                try:
                    from semblend_core.pipeline import PositionMapping
                    from synapse_kv_connector.model_runner_hook import RoPECorrectionHook
                    _pos_map = PositionMapping(
                        donor_positions=list(range(donor_hit)),
                        target_positions=list(range(donor_hit)),
                    )
                    _rope_hook = RoPECorrectionHook(
                        position_map=_pos_map,
                        plan=None,
                        request_id=request.request_id,
                    )
                    self._active_hook = _rope_hook
                    import synapse_kv_connector.semblend_connector as _mod_fd
                    _mod_fd._semblend_active_hook = _rope_hook
                    print(
                        f"[SemBlend] FORCE_DELTA hook created (legacy): "
                        f"delta={_force_delta_legacy}, positions={donor_hit}, "
                        f"req={request.request_id}",
                        file=sys.stderr, flush=True,
                    )
                except Exception as _fd_err:
                    print(
                        f"[SemBlend] FORCE_DELTA hook FAILED (legacy): {_fd_err}",
                        file=sys.stderr, flush=True,
                    )

            # Return value must be need_to_allocate =
            # donor_hit - num_computed_tokens (matching LMCache's
            # convention). vLLM adds this to prefix cache hits to get
            # total computed tokens.
            # Also: must leave at least 1 token for vLLM to compute.
            need = donor_hit - (num_computed_tokens or 0)
            if donor_hit == prompt_len:
                need -= 1  # Full hit: recompute last token for logits
            if need <= 0:
                # Donor doesn't add value beyond prefix cache
                self._donor_token_map.pop(request.request_id, None)
                return (num_matched or 0, False)
            return (need, False)

        except Exception:
            logger.warning(
                "SemBlend: donor injection failed for req=%s",
                request.request_id,
                exc_info=True,
            )
            return (num_matched or 0, False)

    def _capture_kv_fingerprint(
        self, layer_name: str, kv_layer: torch.Tensor
    ) -> None:
        """Capture per-layer KV fingerprint during save_kv_layer (WORKER side).

        Extracts lightweight statistics from the K matrix:
        - L2 norm (captures scale/magnitude of KV state)
        - Mean K vector (captures direction, head_dim floats)

        Since save_kv_layer runs on the WORKER process and request_finished
        runs on the SCHEDULER process, we persist fingerprints to /tmp
        for cross-process transfer.

        Args:
            layer_name: Layer identifier (e.g., "model.layers.0.self_attn").
            kv_layer: KV tensor for this layer.
        """
        if not self._fingerprint_enabled:
            return

        try:
            layer_idx = _parse_layer_index(layer_name)
            if layer_idx is None:
                return

            if kv_layer.dim() < 2:
                return

            with torch.no_grad():
                if kv_layer.dim() == 4:
                    k_data = kv_layer[:, 0, :, :].float()
                elif kv_layer.dim() == 3:
                    k_data = kv_layer[0].float()
                else:
                    k_data = kv_layer.float()

                k_norm = torch.norm(k_data).item()
                k_flat = k_data.reshape(-1, k_data.shape[-1])
                k_mean = k_flat.mean(dim=0).cpu().tolist()

            if "__global__" not in self._pending_fingerprints:
                self._pending_fingerprints["__global__"] = {}
            self._pending_fingerprints["__global__"][layer_idx] = {
                "norm": k_norm,
                "mean_k": k_mean[:32],
            }
        except Exception:
            pass  # Fingerprinting is best-effort

    def _log_layer_deviation(
        self, layer_name: str, kv_layer: torch.Tensor
    ) -> None:
        """Log per-layer KV deviation for bathtub curve calibration.

        Computes the K-norm deviation at this layer and writes it to a
        JSON-lines file at /tmp/semblend_deviations.jsonl. Post-hoc
        analysis of this file fits the bathtub curve parameters.

        This is lightweight: only computes L2 norm per layer, ~10μs.
        """
        try:
            layer_idx = _parse_layer_index(layer_name)
            if layer_idx is None:
                return

            with torch.no_grad():
                if kv_layer.dim() == 4:
                    k_data = kv_layer[:, 0, :, :].float()
                elif kv_layer.dim() == 3:
                    k_data = kv_layer[0].float()
                else:
                    k_data = kv_layer.float()

                k_norm = torch.norm(k_data).item()
                k_mean_norm = torch.norm(k_data.mean(dim=-2)).item()

            # Accumulate deviations in memory, flush on wait_for_save
            if "__deviations__" not in self._pending_fingerprints:
                self._pending_fingerprints["__deviations__"] = []
            self._pending_fingerprints["__deviations__"].append({
                "layer_idx": layer_idx,
                "k_norm": k_norm,
                "k_mean_norm": k_mean_norm,
            })
        except Exception:
            pass

    def _flush_fingerprints_to_disk(self) -> None:
        """Write accumulated fingerprints to /tmp for SCHEDULER to read.

        Called after wait_for_save() completes (WORKER side).
        The SCHEDULER reads these during request_finished → _register_donor.
        """
        layer_data = self._pending_fingerprints.pop("__global__", {})
        deviation_data = self._pending_fingerprints.pop("__deviations__", [])

        try:
            fp_dir = Path("/tmp/semblend_fingerprints")
            fp_dir.mkdir(exist_ok=True)

            if layer_data:
                fp_path = fp_dir / "latest.json"
                fp_obj = {}
                for idx, data in sorted(layer_data.items()):
                    fp_obj[str(idx)] = data
                fp_path.write_text(json.dumps(fp_obj))
                print(
                    f"[SemBlend] WORKER: flushed fingerprint "
                    f"({len(fp_obj)} layers) to {fp_path}",
                    file=sys.stderr, flush=True,
                )

            # Flush bathtub deviation data (JSONL for post-hoc analysis)
            if deviation_data:
                import time as _time
                dev_path = fp_dir / "deviations.jsonl"
                with open(dev_path, "a") as f:
                    entry = {
                        "timestamp": _time.time(),
                        "layers": deviation_data,
                    }
                    f.write(json.dumps(entry) + "\n")
                print(
                    f"[SemBlend] WORKER: flushed {len(deviation_data)} "
                    f"layer deviations to {dev_path}",
                    file=sys.stderr, flush=True,
                )
        except Exception as e:
            print(
                f"[SemBlend] fingerprint flush failed: {e}",
                file=sys.stderr, flush=True,
            )

    def _build_kv_fingerprint(self) -> KVFingerprint | None:
        """Build KVFingerprint from disk (cross-process from WORKER).

        The WORKER writes fingerprints to /tmp/semblend_fingerprints/latest.json
        during wait_for_save(). The SCHEDULER reads them here during
        request_finished → _register_donor.
        """
        # First try in-process data (if WORKER and SCHEDULER are same process)
        layer_data = self._pending_fingerprints.pop("__global__", {})

        # If no in-process data, read from disk (cross-process IPC)
        if not layer_data:
            try:
                fp_path = Path("/tmp/semblend_fingerprints/latest.json")
                if fp_path.exists():
                    raw = json.loads(fp_path.read_text())
                    layer_data = {int(k): v for k, v in raw.items()}
                    # Consume the file so the next request doesn't reuse it
                    fp_path.unlink(missing_ok=True)
                    print(
                        f"[SemBlend] SCHEDULER: read fingerprint from disk "
                        f"({len(layer_data)} layers)",
                        file=sys.stderr, flush=True,
                    )
            except Exception:
                pass

        if not layer_data:
            return None

        max_layer = max(layer_data.keys()) + 1
        norms = [0.0] * max_layer
        means = [[]] * max_layer

        for idx, data in sorted(layer_data.items()):
            norms[idx] = data["norm"]
            means[idx] = data["mean_k"]

        return KVFingerprint(layer_norms=norms, layer_mean_keys=means)

    def _predict_layer_deviations(
        self,
        donor_fp: KVFingerprint,
        query_fp: KVFingerprint | None,
    ) -> list[LayerDeviation]:
        """Predict which layers will deviate between donor and query.

        Uses KV fingerprint comparison to predict deviation without
        computing full KV. This implements the bathtub curve insight:
        early and late layers deviate more, middle layers are stable.

        Args:
            donor_fp: Donor's per-layer KV fingerprint.
            query_fp: Query's fingerprint (if available from a probe).

        Returns:
            Per-layer deviation predictions.
        """
        deviations = []
        num_layers = len(donor_fp.layer_norms)

        # Deviation threshold — layers above this should be recomputed
        # Bathtub curve heuristic: layers 0-3 and last 4 are high-deviation
        recompute_threshold = float(
            os.environ.get("SEMBLEND_LAYER_DEVIATION_THRESHOLD", "0.3")
        )

        for i in range(num_layers):
            # Without query fingerprint, use bathtub heuristic
            if query_fp is None or i >= len(query_fp.layer_norms):
                # Bathtub: first 12.5% and last 12.5% of layers deviate
                frac = i / max(num_layers - 1, 1)
                is_edge = frac < 0.125 or frac > 0.875
                deviations.append(
                    LayerDeviation(
                        layer_idx=i,
                        norm_distance=1.0 if is_edge else 0.1,
                        direction_similarity=0.5 if is_edge else 0.95,
                        should_recompute=is_edge,
                    )
                )
                continue

            # Compare norms
            d_norm = donor_fp.layer_norms[i]
            q_norm = query_fp.layer_norms[i]
            norm_dist = abs(d_norm - q_norm) / max(d_norm, q_norm, 1e-6)

            # Compare mean K direction
            d_mean = donor_fp.layer_mean_keys[i]
            q_mean = query_fp.layer_mean_keys[i]
            dir_sim = _cosine_similarity(d_mean, q_mean) if d_mean and q_mean else 0.0

            should_recompute = (
                norm_dist > recompute_threshold or dir_sim < (1.0 - recompute_threshold)
            )

            deviations.append(
                LayerDeviation(
                    layer_idx=i,
                    norm_distance=norm_dist,
                    direction_similarity=dir_sim,
                    should_recompute=should_recompute,
                )
            )

        return deviations

    def _predict_layer_deviations_bathtub(
        self, num_layers: int
    ) -> list[LayerDeviation]:
        """Bathtub curve heuristic: first/last 12.5% of layers deviate most.

        Applied when no real KV fingerprints are available. Based on the
        empirical finding that early layers (input parsing) and late layers
        (output generation) diverge most between semantically similar prompts,
        while middle layers (abstract reasoning) are highly shareable.
        """
        deviations = []
        for i in range(num_layers):
            frac = i / max(num_layers - 1, 1)
            is_edge = frac < 0.125 or frac > 0.875
            deviations.append(
                LayerDeviation(
                    layer_idx=i,
                    norm_distance=1.0 if is_edge else 0.1,
                    direction_similarity=0.5 if is_edge else 0.95,
                    should_recompute=is_edge,
                )
            )
        return deviations

    def _try_patch_model_runner(self) -> None:
        """Attempt to patch the model runner for PartialAttention.

        Walks the call stack to find GPUModelRunnerV1, then applies the
        monkey-patch for intercepting prefill with donor KV scatter.
        Called during register_kv_caches (worker-side).
        """
        try:
            import inspect

            from synapse_kv_connector.model_runner_hook import patch_model_runner

            for frame_info in inspect.stack():
                local_self = frame_info.frame.f_locals.get("self")
                if (
                    local_self is not None
                    and hasattr(local_self, "execute_model")
                    and hasattr(local_self, "model")
                ):
                    if patch_model_runner(local_self, self):
                        self._model_runner_patched = True
                        print(
                            "[SemBlend] model runner patched for PartialAttention",
                            file=sys.stderr, flush=True,
                        )
                    break
        except Exception:
            logger.warning(
                "PartialAttention: model runner patch failed",
                exc_info=True,
            )

    def _register_model_for_cacheblend(self) -> None:
        """Register vLLM model with LMCache VLLMModelTracker for CacheBlend.

        CacheBlend's blender needs access to the loaded model weights to
        compute layer-level deviation scores. LMCache expects the model
        to be registered via VLLMModelTracker.register_model() before the
        blender is created. vLLM v0.14.1 never calls this, so we do it
        here during register_kv_caches() when the model is available.

        We find the model by inspecting the call stack — register_kv_caches
        is called from GPUModelRunnerV1.initialize_kv_cache_group() which
        has self.model available.
        """
        try:
            from lmcache.integration.vllm.utils import ENGINE_NAME
            from lmcache.v1.compute.models.utils import VLLMModelTracker

            # Walk the stack to find the GPUModelRunnerV1 instance
            import inspect

            model = None
            for frame_info in inspect.stack():
                local_self = frame_info.frame.f_locals.get("self")
                if local_self is not None and hasattr(local_self, "model"):
                    candidate = getattr(local_self, "model", None)
                    if candidate is not None and hasattr(candidate, "parameters"):
                        model = candidate
                        break

            if model is not None:
                VLLMModelTracker.register_model(ENGINE_NAME, model)
                logger.info(
                    "CacheBlend: registered model %s with VLLMModelTracker",
                    type(model).__name__,
                )
            else:
                logger.warning(
                    "CacheBlend: could not find model in call stack"
                )
        except ImportError:
            logger.debug("LMCache CacheBlend imports not available")
        except Exception:
            logger.warning(
                "CacheBlend: model registration failed", exc_info=True
            )

    def _get_lmcache_engine_impl(self) -> Any:
        """Get the inner LMCacheConnectorV1Impl from the LMCache connector.

        The chain is: SemBlend._lmcache (LMCacheConnectorV1)
            → _lmcache_engine (LMCacheConnectorV1Impl)
        The Impl has load_specs, lookup_client, etc.
        """
        engine = getattr(self._lmcache, "_lmcache_engine", None)
        if engine is None:
            logger.warning("SemBlend: no LMCache engine")
        return engine

    def _get_prompt_text(self, request: "Request") -> str:
        """Extract prompt text from request, with tokenizer decode fallback.

        vLLM's Request object may or may not have the prompt text on the
        scheduler side. If not, decode from token IDs using the cached
        tokenizer (loaded once, reused for all requests).
        """
        # Try direct prompt attribute
        prompt = getattr(request, "prompt", None) or ""
        if prompt:
            return prompt

        # Try prompt_text
        prompt = getattr(request, "prompt_text", None) or ""
        if prompt:
            return prompt

        # Decode from token IDs using cached tokenizer
        try:
            token_ids = list(request.all_token_ids)
            tokenizer = self._get_tokenizer()
            if tokenizer is not None:
                t0 = time.monotonic()
                n = len(token_ids)
                # For long prompts, sample beginning + middle + end so the
                # MiniLM embedding (512-token window) sees representative content
                # from the full document rather than just the first 24% at 8K.
                _MAX_DECODE = 2000
                if n <= _MAX_DECODE:
                    sample = token_ids
                else:
                    # 40% beginning (instruction + doc start), 30% middle, 30% end
                    head = int(_MAX_DECODE * 0.40)
                    mid_w = int(_MAX_DECODE * 0.30)
                    tail = _MAX_DECODE - head - mid_w
                    mid_start = (n - mid_w) // 2
                    sample = (
                        token_ids[:head]
                        + token_ids[mid_start: mid_start + mid_w]
                        + token_ids[n - tail:]
                    )
                prompt = tokenizer.decode(sample, skip_special_tokens=True)
                elapsed = (time.monotonic() - t0) * 1000
                if prompt:
                    print(
                        f"[SemBlend] decoded prompt: "
                        f"{len(prompt)}ch from {n}tok "
                        f"(sample={len(sample)}) in {elapsed:.0f}ms",
                        file=sys.stderr, flush=True,
                    )
                    return prompt
        except Exception as e:
            print(
                f"[SemBlend] tokenizer decode failed: {e}",
                file=sys.stderr, flush=True,
            )

        return ""

    def _get_tokenizer(self) -> Any:
        """Get or create cached tokenizer instance."""
        if hasattr(self, "_cached_tokenizer"):
            return self._cached_tokenizer

        self._cached_tokenizer = None
        try:
            if hasattr(self, "_vllm_config") and self._vllm_config:
                model_config = getattr(self._vllm_config, "model_config", None)
                if model_config:
                    tok_name = getattr(
                        model_config, "tokenizer", None
                    ) or getattr(model_config, "model", None)
                    if tok_name:
                        from transformers import AutoTokenizer

                        t0 = time.monotonic()
                        self._cached_tokenizer = AutoTokenizer.from_pretrained(
                            tok_name, trust_remote_code=True
                        )
                        elapsed = (time.monotonic() - t0) * 1000
                        print(
                            f"[SemBlend] tokenizer loaded: {tok_name} "
                            f"in {elapsed:.0f}ms",
                            file=sys.stderr, flush=True,
                        )
        except Exception as e:
            print(
                f"[SemBlend] tokenizer load failed: {e}",
                file=sys.stderr, flush=True,
            )

        return self._cached_tokenizer

    def _register_donor(self, request: "Request") -> None:
        """Register completed request as a potential donor."""
        # Use only prompt tokens for registration (not generated output).
        # all_token_ids includes output which changes the embedding and
        # causes lookup mismatches (output text shifts sorted sentences).
        num_prompt = (
            getattr(request, "num_prompt_tokens", None)
            or getattr(request, "num_tokens", None)
        )
        all_ids = list(request.all_token_ids)
        if num_prompt and 0 < num_prompt < len(all_ids):
            token_ids = all_ids[:num_prompt]
        else:
            token_ids = all_ids
        if len(token_ids) < 256:
            return

        self._stats["total_saves"] += 1

        # Decode prompt-only tokens for embedding (skip output tokens).
        # Use sliding window sampling for long prompts so MiniLM sees
        # representative content from beginning + middle + end of document.
        prompt = ""
        try:
            tokenizer = self._get_tokenizer()
            if tokenizer is not None:
                n = len(token_ids)
                _MAX_DECODE = 2000
                if n <= _MAX_DECODE:
                    sample = token_ids
                else:
                    head = int(_MAX_DECODE * 0.40)
                    mid_w = int(_MAX_DECODE * 0.30)
                    tail = _MAX_DECODE - head - mid_w
                    mid_start = (n - mid_w) // 2
                    sample = (
                        token_ids[:head]
                        + token_ids[mid_start: mid_start + mid_w]
                        + token_ids[n - tail:]
                    )
                prompt = tokenizer.decode(sample, skip_special_tokens=True)
        except Exception:
            pass
        if not prompt:
            prompt = self._get_prompt_text(request)

        # New pipeline path
        if self._pipeline is not None:
            self._pipeline.register_donor(
                request_id=request.request_id,
                token_ids=token_ids,
                prompt_text=prompt,
            )
            print(
                f"[SemBlend] donor reg (pipeline): req={request.request_id}, "
                f"tok={len(token_ids)}/{len(all_ids)}, "
                f"num_prompt={num_prompt}, prompt={len(prompt)}ch",
                file=sys.stderr, flush=True,
            )
            return

        # Legacy path
        embedding = None
        if prompt:
            embedding = self._donor_store.get_embedding(prompt[:2000])

        kv_fp = self._build_kv_fingerprint() if self._fingerprint_enabled else None

        self._donor_store.add_donor(
            request_id=request.request_id,
            token_ids=token_ids,
            prompt_text=prompt,
            embedding=embedding,
            kv_fingerprint=kv_fp,
        )

        print(
            f"[SemBlend] donor reg: req={request.request_id}, "
            f"tok={len(token_ids)}, emb={'yes' if embedding else 'no'}, "
            f"kv_fp={'yes' if kv_fp else 'no'}, "
            f"prompt={len(prompt)}ch",
            file=sys.stderr, flush=True,
        )

    def get_stats(self) -> dict[str, Any]:
        donor_size = (
            self._pipeline.donor_count
            if self._pipeline is not None
            else self._donor_store.size
        )
        return {
            **self._stats,
            "donor_store_size": donor_size,
            "pending_swaps": len(self._donor_token_map),
            "use_pipeline": self._pipeline is not None,
        }


# ==============================
# Utility functions
# ==============================


def _parse_layer_index(layer_name: str) -> int | None:
    """Extract layer index from vLLM layer name like 'model.layers.5.self_attn'."""
    import re

    match = re.search(r"layers\.(\d+)", layer_name)
    return int(match.group(1)) if match else None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _common_prefix_length(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def _token_overlap_count(a: list[int], b: list[int]) -> int:
    ca = Counter(a)
    cb = Counter(b)
    return sum((ca & cb).values())

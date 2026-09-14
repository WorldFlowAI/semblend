"""SemBlend TRT-LLM PyTorch backend -- SemBlendBackend implementation.

Implements the 4 abstract methods from semblend_core.backend.SemBlendBackend
for TRT-LLM's PyTorch backend (tensorrt_llm.torch).

The PyTorch backend exposes KVCacheManager with:
    - get_buffers(layer_idx) -> [num_blocks, 2, tokens_per_block, num_kv_heads, head_dim]
    - get_batch_cache_indices(requests) -> block IDs per request
    - allocate_blocks() / free_blocks() -> block lifecycle

This backend uses the existing SemBlend pipeline (embedding, donor lookup,
alignment) and the existing Triton kernels (RoPE correction, KV scatter)
with TRT-LLM-specific stride computation.

Requires: tensorrt_llm, torch (install with: pip install semblend[trtllm])
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any

from semblend.integration.trtllm.namespace import (
    NO_CACHE_SALT_NAMESPACE,
    bind_cache_salt,
    build_cache_namespace,
    cache_salt_namespace,
    namespace_key,
)

logger = logging.getLogger("semblend.trtllm.backend")

# One-time operator signal for a process where no salt ever arrives.
#
# The salt is supplied by callers outside this package (the engine, the
# serving layer, the deployment's own request shaping). A process that never
# receives one keys every donor into the no-cache-salt namespace: correct for
# a single tenant, a cross-tenant leak for anyone else, and nothing inside
# the process can tell those two deployments apart. So the default stays as
# the upstream contracts expect and the operator is told once instead.
UNSALTED_DONOR_WARN_AFTER_ENV = "SEMBLEND_UNSALTED_DONOR_WARN_AFTER"

_DEFAULT_UNSALTED_DONOR_WARN_AFTER = 32


def unsalted_donor_warn_after() -> int:
    """Unsalted registrations a component tolerates before it warns once."""
    raw = os.environ.get(UNSALTED_DONOR_WARN_AFTER_ENV, "").strip()
    if not raw:
        return _DEFAULT_UNSALTED_DONOR_WARN_AFTER
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "%s=%r is not an integer -- falling back to %d",
            UNSALTED_DONOR_WARN_AFTER_ENV,
            raw,
            _DEFAULT_UNSALTED_DONOR_WARN_AFTER,
        )
        return _DEFAULT_UNSALTED_DONOR_WARN_AFTER


class SaltIsolationWatch:
    """Counts donor registrations by whether the request carried a salt.

    Shared by this backend and both TensorRT-LLM providers so the signal
    reads the same whichever component a deployment drives.

    Warns at most once, and only while not a single salted registration has
    been seen: a deployment that salts some requests is configured, so the
    unsalted ones there are a deliberate shared namespace rather than an
    isolation that was never switched on.

    Args:
        component: Name of the provider/backend, so an operator running
            several knows which one is unisolated.
        config_path: Where that component reads the salt from, so the
            warning says what to set rather than only what is wrong.
        warn_after: Threshold override; defaults to the env-tunable value.
        log: Logger to warn through, so the record carries the component's
            own logger name.
    """

    def __init__(
        self,
        component: str,
        config_path: str,
        *,
        warn_after: int | None = None,
        log: logging.Logger | None = None,
    ) -> None:
        self._component = component
        self._config_path = config_path
        self._warn_after = unsalted_donor_warn_after() if warn_after is None else warn_after
        self._log = log or logger
        self._unsalted = 0
        self._salted = 0
        self._warned = False

    def record(self, salted: bool) -> None:
        """Count one donor registration, warning if this process never salts."""
        if salted:
            self._salted += 1
            return

        self._unsalted += 1
        if self._warned or self._salted or self._warn_after <= 0:
            return
        if self._unsalted < self._warn_after:
            return

        self._warned = True
        self._log.warning(
            "%s: %d donors registered and not one carried a cache_salt, so every "
            "request in this process can reuse every other request's KV. Correct "
            "for a single tenant; otherwise set a per-tenant salt via %s. "
            "Threshold: %s.",
            self._component,
            self._unsalted,
            self._config_path,
            UNSALTED_DONOR_WARN_AFTER_ENV,
        )

    def stats(self) -> dict:
        """Counters for the owning component's get_stats()."""
        return {
            "unsalted_donors_registered": self._unsalted,
            "salted_donors_registered": self._salted,
            "unsalted_namespace_warned": self._warned,
        }


class TRTLLMPyTorchBackend:
    """SemBlend backend adapter for TRT-LLM's PyTorch backend.

    Bridges the SemBlend pipeline with TRT-LLM's KVCacheManager,
    providing KV injection, donor registration, and RoPE correction.

    This backend follows the token substitution pattern (Approach B):
    on semantic hit, donor token IDs are substituted before request
    submission so TRT-LLM's radix tree matches the donor's cached KV.
    After load, RoPE correction adjusts K positions.

    Args:
        kv_cache_manager: TRT-LLM KVCacheManager instance from the
            PyTorch ModelEngine. Must expose get_buffers(layer_idx).
        model_config: Dict with model parameters. Expected keys:
            - num_layers: Number of transformer layers.
            - num_kv_heads: Number of KV attention heads.
            - head_dim: Dimension per attention head.
            - tokens_per_block: Tokens per KV block (default 128).
            - rope_base: RoPE base frequency (default 10000.0).
            - model_name: Model name for bathtub preset lookup.
        tokenizer: Optional tokenizer for token-to-text decoding.
            If None, uses SEMBLEND_MODEL_NAME env var to load one.
    """

    def __init__(
        self,
        kv_cache_manager: Any,
        model_config: dict,
        tokenizer: Any | None = None,
    ) -> None:
        self._kv_mgr = kv_cache_manager
        self._config = dict(model_config)
        self._tokenizer = tokenizer
        self._tokenizer_load_attempted = False

        # Extract config with defaults
        self._tokens_per_block = self._config.get("tokens_per_block", 128)
        self._num_layers = self._config.get("num_layers", 28)
        self._num_kv_heads = self._config.get("num_kv_heads", 4)
        self._head_dim = self._config.get("head_dim", 128)
        self._rope_base = self._config.get("rope_base", 10000.0)
        self._model_name = self._config.get("model_name", "")

        # Donor isolation namespace. Every donor registered through this
        # backend is keyed by this namespace plus the registering request's
        # cache_salt, and every lookup is keyed the same way, so KV produced
        # under one salt is unreachable from another. Requests that carry no
        # salt share the no-cache-salt sentinel, which is what keeps
        # single-tenant deployments reusing exactly as before.
        self._namespace = build_cache_namespace(
            model=self._model_name,
            block_size=self._tokens_per_block,
        )

        # Pipeline (lazy init to avoid import overhead at module load)
        self._pipeline = None

        # Stats
        self._stats = {
            "injections": 0,
            "donors_registered": 0,
            "rope_corrections": 0,
            "semantic_hits": 0,
            "misses": 0,
            # Donors withheld from a lookup because their cache_salt
            # namespace differed from the requesting tenant's. Non-zero here
            # is isolation doing its job, not an error.
            "cross_namespace_rejected": 0,
        }

        # Isolation here is only as good as the salt the launcher hook hands
        # down, and that comes from out-of-repo request objects, so a
        # deployment that never sets one is reported rather than guessed at.
        self._salt_watch = SaltIsolationWatch(
            "TRTLLMPyTorchBackend (semblend-trtllm launcher hook)",
            "cache_salt on the request, or kv_metadata['cache_salt'] at registration",
        )

        logger.info(
            "TRTLLMPyTorchBackend initialized: "
            "tokens_per_block=%d, num_layers=%d, num_kv_heads=%d, "
            "head_dim=%d, rope_base=%.1f",
            self._tokens_per_block,
            self._num_layers,
            self._num_kv_heads,
            self._head_dim,
            self._rope_base,
        )

    # ------------------------------------------------------------------
    # SemBlendBackend ABC implementation
    # ------------------------------------------------------------------

    def get_kv_block_size(self) -> int:
        """Return the KV cache block size in tokens.

        TRT-LLM PyTorch backend typically uses 128 tokens per block,
        configurable via KvCacheConfig(tokens_per_block=...).
        """
        return self._tokens_per_block

    def inject_donor_kv(
        self,
        donor_kv: Any,
        block_table: Any,
        token_mapping: list[tuple[int, int]],
        layer_idx: int,
    ) -> None:
        """Write donor KV tensors into TRT-LLM's paged KV cache.

        Uses the existing rope_correct_scatter_paged Triton kernel with
        TRT-LLM-specific strides for zero-copy layout handling.

        Args:
            donor_kv: Donor KV tensor [2, num_kv_heads, donor_seq_len, head_dim].
            block_table: Block table mapping logical to physical blocks.
            token_mapping: List of (donor_pos, target_pos) pairs.
            layer_idx: Transformer layer index.
        """
        import torch

        if not token_mapping:
            return

        kv_buffer = self._kv_mgr.get_buffers(layer_idx)

        donor_positions = torch.tensor(
            [p[0] for p in token_mapping],
            dtype=torch.int32,
            device=kv_buffer.device,
        )
        target_positions = torch.tensor(
            [p[1] for p in token_mapping],
            dtype=torch.int32,
            device=kv_buffer.device,
        )

        from semblend_core.rope_correction import rope_correct_scatter_paged

        rope_correct_scatter_paged(
            kv_cache=kv_buffer,
            donor_kv=donor_kv,
            block_table=block_table,
            donor_positions=donor_positions,
            target_positions=target_positions,
            rope_base=self._rope_base,
            head_dim=self._head_dim,
        )

        self._stats["injections"] += 1

    def register_donor(
        self,
        request_id: str,
        token_ids: list[int],
        kv_metadata: dict,
        *,
        cache_salt: Any = None,
    ) -> None:
        """Register a completed request as a donor for future reuse.

        Lightweight path: stores (embedding, token_ids, timestamp) only.
        Does NOT extract KV tensors -- relies on TRT-LLM's radix tree
        to still have the blocks cached. On semantic hit, we check if
        the donor's tokens are in the radix tree; if evicted, it's a miss.

        This is the SGLang pattern (radix_backend.py:_lookup_donor_in_tree).

        Args:
            request_id: Unique request identifier.
            token_ids: Token IDs of the completed request.
            kv_metadata: Engine-specific metadata. A ``cache_salt`` entry is
                read from it when the keyword below is not supplied, so a
                caller that already threads metadata needs no new argument.
            cache_salt: Isolation salt of the completed request. The donor is
                reachable only from lookups carrying the same salt; omitting
                it registers into the no-cache-salt namespace.
        """
        pipeline = self._get_pipeline()
        if pipeline is None:
            return

        prompt_text = self._tokens_to_text(token_ids)
        if prompt_text is None:
            return

        salt = cache_salt if cache_salt is not None else _metadata_cache_salt(kv_metadata)
        pipeline.register_donor(
            request_id=request_id,
            token_ids=token_ids,
            prompt_text=prompt_text,
            extra_key=self._request_namespace(salt),
            # The raw salt as well as the namespace derived from it: the
            # published tenant key is a function of the salt alone, so a
            # router that set the salt can select this donor by tenant.
            cache_salt=salt,
        )
        self._stats["donors_registered"] += 1
        # Reached only with the pipeline live, so the count reflects donors
        # this process actually made reusable.
        self._salt_watch.record(cache_salt_namespace(salt) != NO_CACHE_SALT_NAMESPACE)

    def apply_rope_correction(
        self,
        kv_cache: Any,
        position_deltas: Any,
        rope_config: dict,
    ) -> None:
        """Apply RoPE position correction on injected KV tensors.

        Uses the existing permute_paged_kv_with_rope from semblend_core
        with TRT-LLM stride computation for layout adaptation.

        Args:
            kv_cache: Per-layer KV cache tensor from get_buffers().
            position_deltas: List of (donor_pos, target_pos) pairs,
                or a PositionMapping object.
            rope_config: Dict with rope_base and head_dim.
        """
        if not position_deltas:
            return

        # Convert PositionMapping to list of tuples if needed
        if hasattr(position_deltas, "donor_positions"):
            pairs = list(
                zip(
                    position_deltas.donor_positions,
                    position_deltas.target_positions,
                )
            )
        else:
            pairs = list(position_deltas)

        if not pairs:
            return

        rope_base = rope_config.get("rope_base", self._rope_base)

        from semblend_core.rope_correction import permute_paged_kv_with_rope

        # permute_paged_kv_with_rope expects [num_blocks, 2, num_heads, block_size, head_dim]
        # TRT-LLM layout has pos and head swapped -- we need block_table
        # For direct in-place correction, use apply_rope_delta_inplace
        # which auto-detects layout

        block_table = rope_config.get("block_table")
        if block_table is not None:
            permute_paged_kv_with_rope(
                kv_cache=kv_cache,
                block_table=block_table,
                permutation=pairs,
                rope_base=rope_base,
            )
            self._stats["rope_corrections"] += 1

    def get_model_config(self) -> dict:
        """Return model configuration for pipeline decisions."""
        return {
            "num_layers": self._num_layers,
            "num_heads": self._num_kv_heads,
            "head_dim": self._head_dim,
            "model_name": self._model_name,
            "rope_base": self._rope_base,
            "max_seq_len": self._config.get("max_seq_len", 32768),
        }

    # ------------------------------------------------------------------
    # Semantic donor discovery
    # ------------------------------------------------------------------

    def find_semantic_donor(
        self,
        token_ids: list[int],
        prompt_text: str | None = None,
        *,
        cache_salt: Any = None,
    ) -> dict | None:
        """Run the SemBlend pipeline to find a semantic donor.

        Called before request submission to TRT-LLM. If a donor is found,
        the caller should substitute donor token IDs for the prompt prefix
        to trigger a radix tree hit in TRT-LLM.

        Args:
            token_ids: Token IDs of the incoming request.
            prompt_text: Decoded prompt text. If None, decodes from token_ids.
            cache_salt: Isolation salt of the incoming request. Only donors
                registered under the same salt are candidates; a caller that
                omits it sees the no-cache-salt namespace only, never another
                tenant's donors.

        Returns:
            Dict with donor info on hit, None on miss:
            {
                "donor_id": str,
                "donor_tokens": list[int],
                "similarity": float,
                "reuse_ratio": float,
                "position_map": PositionMapping,
                "timings": PipelineTimings,
            }
        """
        pipeline = self._get_pipeline()
        if pipeline is None:
            self._stats["misses"] += 1
            return None

        if prompt_text is None:
            prompt_text = self._tokens_to_text(token_ids)
        if prompt_text is None:
            self._stats["misses"] += 1
            return None

        extra_key = self._request_namespace(cache_salt)
        result = pipeline.find_donor(
            token_ids=token_ids,
            prompt_text=prompt_text,
            extra_key=extra_key,
        )

        if not result.found:
            self._stats["misses"] += 1
            self._stats["cross_namespace_rejected"] += self._withheld_donor_count(extra_key)
            return None

        if not self._namespace_allows(result.donor_id or "", extra_key):
            self._stats["misses"] += 1
            return None

        self._stats["semantic_hits"] += 1
        logger.info(
            "SemBlend semantic hit: donor=%s, sim=%.3f, reuse=%.2f "
            "(embed=%.1fms, lookup=%.1fms, total=%.1fms)",
            result.donor_id,
            result.similarity,
            result.reuse_ratio,
            result.timings.embed_ms,
            result.timings.lookup_ms,
            result.timings.total_ms,
        )

        return {
            "donor_id": result.donor_id,
            "donor_tokens": result.donor_tokens,
            "similarity": result.similarity,
            "reuse_ratio": result.reuse_ratio,
            "position_map": result.position_map,
            "timings": result.timings,
        }

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return SemBlend-specific statistics."""
        pipeline = self._pipeline
        donor_count = pipeline.donor_count if pipeline is not None else 0
        return {
            **self._stats,
            **self._salt_watch.stats(),
            "donor_store_size": donor_count,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _request_namespace(self, cache_salt: Any) -> str:
        """Donor-store isolation key: the backend namespace plus this salt.

        The salt is hashed on the way in, so neither the donor record nor a
        log line ever carries the tenant-identifying value itself.
        """
        return namespace_key(bind_cache_salt(self._namespace, cache_salt))

    def _namespace_allows(self, donor_id: str, extra_key: str) -> bool:
        """Defense in depth over the donor store's own extra_key filter.

        Fails closed: a donor whose namespace cannot be resolved is
        rejected. The store already filters on extra_key, so this only fires
        when the store and the request disagree — which is exactly the case
        that must never be served.
        """
        donor_namespace = self._donor_namespace(donor_id)
        if donor_namespace == extra_key:
            return True
        self._stats["cross_namespace_rejected"] += 1
        logger.warning(
            "namespace reject: donor=%s donor_ns=%s request_ns=%s",
            donor_id,
            donor_namespace,
            extra_key,
        )
        return False

    def _donor_namespace(self, donor_id: str) -> str | None:
        """Namespace key a stored donor was registered under, None if unknown."""
        pipeline = self._pipeline
        if pipeline is None or not donor_id:
            return None
        try:
            node = pipeline._donor_store.get_donor(donor_id)
        except Exception:
            return None
        if node is None:
            return None
        return getattr(node, "extra_key", None)

    def _withheld_donor_count(self, extra_key: str) -> int:
        """Donors in the store this namespace is not allowed to see.

        semblend_core filters on extra_key before it proposes candidates, so
        a foreign donor never surfaces as a rejected candidate and the
        operator-visible counter would read zero while isolation was in fact
        doing its job. Only called on a miss, after the lookup has already
        paid for an embedding.
        """
        pipeline = self._pipeline
        if pipeline is None:
            return 0
        try:
            entries = pipeline._donor_store._entries
            return sum(
                1 for node in entries.values() if getattr(node, "extra_key", None) != extra_key
            )
        except Exception:
            return 0

    def _get_pipeline(self):
        """Lazily initialize the SemBlend pipeline."""
        if self._pipeline is not None:
            return self._pipeline

        enabled = os.environ.get("SEMBLEND_ENABLED", "1") == "1"
        if not enabled:
            return None

        from semblend_core.pipeline import SemBlendPipeline

        min_similarity = float(os.environ.get("SEMBLEND_MIN_SIMILARITY", "0.60"))
        max_donors = int(os.environ.get("SEMBLEND_MAX_DONORS", "1000"))
        embedder_type = os.environ.get("SEMBLEND_EMBEDDER", "minilm")
        min_reuse = float(os.environ.get("SEMBLEND_MIN_REUSE_RATIO", "0.50"))

        self._pipeline = SemBlendPipeline(
            max_donors=max_donors,
            min_similarity=min_similarity,
            min_reuse_ratio=min_reuse,
            embedder_type=embedder_type,
            model_name=self._model_name,
            backend=self,
            chunk_size=self._tokens_per_block,
        )

        logger.info(
            "SemBlend pipeline initialized for TRT-LLM: min_sim=%.2f, max_donors=%d, chunk_size=%d",
            min_similarity,
            max_donors,
            self._tokens_per_block,
        )
        return self._pipeline

    def _get_tokenizer(self):
        """Lazily load tokenizer for token-to-text decoding."""
        if self._tokenizer is not None:
            return self._tokenizer
        if self._tokenizer_load_attempted:
            return None

        self._tokenizer_load_attempted = True
        model_name = self._model_name or os.environ.get("SEMBLEND_MODEL_NAME", "")
        if not model_name:
            logger.warning(
                "No model name for tokenizer -- set model_config['model_name'] "
                "or SEMBLEND_MODEL_NAME env var"
            )
            return None

        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            logger.info("SemBlend tokenizer loaded: %s", model_name)
        except Exception as e:
            logger.error("Failed to load tokenizer for %s: %s", model_name, e)

        return self._tokenizer

    def _tokens_to_text(self, token_ids: list[int]) -> str | None:
        """Decode token IDs to text using sliding-window sampling."""
        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            return None

        # Sliding-window sampling for long prompts (same as SGLang integration)
        max_decode = 2000
        n = len(token_ids)
        if n <= max_decode:
            sampled = token_ids
        else:
            head = int(max_decode * 0.40)
            mid_w = int(max_decode * 0.30)
            tail = max_decode - head - mid_w
            mid_start = (n - mid_w) // 2
            sampled = (
                token_ids[:head] + token_ids[mid_start : mid_start + mid_w] + token_ids[n - tail :]
            )

        return tokenizer.decode(sampled, skip_special_tokens=True)


def _metadata_cache_salt(kv_metadata: Any) -> Any:
    """Read a request's cache_salt out of engine metadata, if it carries one.

    The engine-side metadata shape varies by call site (mapping or object),
    and a caller that has no salt to offer must land on None so the request
    keys into the no-cache-salt namespace rather than skipping isolation.
    """
    if isinstance(kv_metadata, Mapping):
        return kv_metadata.get("cache_salt")
    return getattr(kv_metadata, "cache_salt", None)

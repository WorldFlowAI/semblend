"""SemBlend backend abstraction — ABC for inference engine adapters.

Each inference engine (vLLM/LMCache, TRT-LLM) implements this interface
to provide engine-specific KV cache operations. The shared SemBlend
pipeline calls these methods without knowing which engine is running.

The backend is responsible for:
  1. Reporting KV block size (256 for LMCache, 128 for TRT-LLM)
  2. Injecting donor KV into the engine's paged KV cache
  3. Registering completed requests as future donors
  4. Applying RoPE position correction on injected KV tensors
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class SemBlendBackend(ABC):
    """Abstract base class for SemBlend engine adapters.

    Implementations provide the bridge between the shared SemBlend
    pipeline (embedding, donor lookup, alignment) and the engine's
    internal KV cache management.
    """

    @abstractmethod
    def get_kv_block_size(self) -> int:
        """Return the KV cache block size in tokens.

        This determines the chunk_size used by the alignment module.
        Must be a power of 2.

        Returns:
            Block size: 256 for LMCache, 128 for TRT-LLM (configurable).
        """

    @abstractmethod
    def inject_donor_kv(
        self,
        donor_kv: Any,
        block_table: Any,
        token_mapping: list[tuple[int, int]],
        layer_idx: int,
    ) -> None:
        """Write donor KV tensors into the engine's paged KV cache.

        For each (donor_pos, target_pos) pair in token_mapping, copies
        the KV vectors from donor_kv at donor_pos into the engine's
        KV cache at target_pos (translated via block_table).

        Args:
            donor_kv: Donor KV tensor (engine-specific format).
                      vLLM: [2, num_heads, seq_len, head_dim]
                      TRT-LLM: [num_blocks, 2, num_heads, tokens_per_block, head_dim]
            block_table: Block table mapping logical to physical blocks.
            token_mapping: List of (donor_pos, target_pos) pairs.
            layer_idx: Transformer layer index for layer-specific injection.
        """

    @abstractmethod
    def register_donor(
        self,
        request_id: str,
        token_ids: list[int],
        kv_metadata: dict,
    ) -> None:
        """Capture a completed request's KV tensors as donor for future reuse.

        Called after a request finishes generation. The backend should
        extract KV tensors from the engine's cache and register them
        with the donor store.

        Args:
            request_id: Unique request identifier.
            token_ids: Token IDs of the completed request.
            kv_metadata: Engine-specific metadata (block table, cache pointers, etc.).
        """

    @abstractmethod
    def apply_rope_correction(
        self,
        kv_cache: Any,
        position_deltas: Any,
        rope_config: dict,
    ) -> None:
        """Apply RoPE position correction on injected KV tensors.

        When donor KV is injected at a different position than where it
        was originally computed, the K cache needs RoPE delta correction:
            K_corrected = RoPE(target_pos - donor_pos) × K_donor

        V cache has no position encoding and needs no correction.

        Args:
            kv_cache: KV cache tensor (engine-specific format).
            position_deltas: Position delta information for correction.
                             Can be a tensor of (donor_pos, target_pos) pairs
                             or engine-specific format.
            rope_config: RoPE configuration dict with keys:
                         - rope_base: Base frequency (default 10000.0)
                         - head_dim: Dimension per attention head
                         - Additional engine-specific parameters.
        """

    def get_model_config(self) -> dict:
        """Return model configuration relevant to SemBlend.

        Optional override for backends that can provide model info
        (num_layers, num_heads, head_dim, etc.) for bathtub curve
        calibration and other pipeline decisions.

        Returns:
            Dict with optional keys: num_layers, num_heads, head_dim,
            model_name, rope_base, max_seq_len.
        """
        return {}

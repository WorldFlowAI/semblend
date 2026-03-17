"""vLLM KVConnector implementation backed by Synapse's KV cache API.

This module implements a connector that follows vLLM's KVConnectorBase
pattern, enabling cross-instance KV cache sharing through Synapse's
REST API. It combines token-level exact matching (Tier 1) with
segment-level approximate matching (Tier 3).

Integration with vLLM:

    vLLM's scheduler calls the connector methods at specific points in
    the inference pipeline:

    1. ``get_num_new_matched_tokens`` -- called during scheduling to
       determine how many prefix tokens have cached KV state, so the
       scheduler can skip prefill for those tokens.

    2. ``start_load_kv`` -- called before prefill begins to retrieve
       the cached KV tensors and inject them into the model's KV cache.

    3. ``start_save_kv`` -- called after generation to persist the
       newly computed KV state so other instances can reuse it.

Usage::

    connector = SynapseKVConnector(
        base_url="http://synapse-gateway:8080",
        tenant_id="my-org",
    )

    # Check prefix match length
    matched = connector.get_num_new_matched_tokens(token_ids)

    # Load cached KV state
    kv = connector.start_load_kv(token_ids)

    # Save new KV state after generation
    connector.start_save_kv(token_ids, kv_data, num_layers, num_heads, head_dim)
"""

from __future__ import annotations

import logging

import numpy as np

from synapse_kv_connector.client import SynapseKVClient
from synapse_kv_connector.partial_attention import (
    PartialAttentionPlan,
    build_attention_plan,
    compute_donor_kv_indices,
)
from synapse_kv_connector.segment_client import (
    KvTransferPlan,
    SynapseSegmentClient,
)

logger = logging.getLogger(__name__)


class SynapseKVConnector:
    """Stub vLLM KVConnector backed by Synapse's KV cache REST API.

    Combines token-level exact matching via ``SynapseKVClient`` (Tier 1)
    with segment-level approximate matching via ``SynapseSegmentClient``
    (Tier 3).

    The connector tries an exact token-hash match first. If no exact
    match exists, the segment client can be used for approximate prefix
    reuse (not yet wired into the scheduling path).

    Args:
        base_url: Synapse gateway base URL (e.g. "http://synapse-gateway:8080").
        tenant_id: Tenant identifier for multi-tenant isolation.
        timeout: Request timeout in seconds for API calls.
    """

    def __init__(
        self,
        base_url: str,
        tenant_id: str = "default",
        timeout: float = 30.0,
        min_reuse_ratio: float = 0.5,
        tier3_enabled: bool = True,
    ) -> None:
        self._token_client = SynapseKVClient(
            base_url=base_url,
            timeout=timeout,
            tenant_id=tenant_id,
        )
        self._segment_client = SynapseSegmentClient(
            base_url=base_url,
            timeout=timeout,
            tenant_id=tenant_id,
        )
        self._min_reuse_ratio = min_reuse_ratio
        self._tier3_enabled = tier3_enabled
        self._last_transfer_plan: KvTransferPlan | None = None
        self._last_attention_plan: PartialAttentionPlan | None = None

    @property
    def token_client(self) -> SynapseKVClient:
        """Direct access to the token-level KV client."""
        return self._token_client

    @property
    def segment_client(self) -> SynapseSegmentClient:
        """Direct access to the segment-level KV client."""
        return self._segment_client

    @property
    def last_transfer_plan(self) -> KvTransferPlan | None:
        """The most recent Tier 3 transfer plan from scheduling.

        After ``get_num_new_matched_tokens`` finds a Tier 3 approximate
        match, the transfer plan is stored here for ``start_load_kv`` to
        use when loading the donor KV state with selective recomputation.
        """
        return self._last_transfer_plan

    @property
    def last_attention_plan(self) -> PartialAttentionPlan | None:
        """The most recent PartialAttention plan from Tier 3 KV loading.

        After ``start_load_kv`` loads donor KV with non-contiguous copy
        positions, the attention plan is stored here. The vLLM worker
        uses this to determine which positions need fresh Q,K,V computation
        and which can reuse donor K,V vectors.

        For contiguous prefix copies this is None (standard vLLM prefix
        skip suffices). For non-contiguous (sparse) copies, this contains
        per-layer, per-position attention masks.
        """
        return self._last_attention_plan

    def get_num_new_matched_tokens(self, token_ids: list[int]) -> int:
        """Check how many prefix tokens have cached KV state.

        First tries an exact token-hash lookup (Tier 1). If no exact
        match and Tier 3 is enabled, queries the DELTA tree for an
        approximate donor match with a KV transfer plan.

        For Tier 3 matches, the transfer plan is stored in
        ``last_transfer_plan`` for use by ``start_load_kv``.

        This is called by vLLM's scheduler to decide how many tokens can
        skip prefill computation.

        Args:
            token_ids: The full token sequence to check.

        Returns:
            Number of matched prefix tokens. For Tier 1 this is the full
            sequence length. For Tier 3 this is the number of positions
            that can reuse donor KV (``num_copied``).
        """
        if not token_ids:
            return 0

        self._last_transfer_plan = None

        # Tier 1: exact token-hash lookup
        try:
            result = self._token_client.load_kv_state(token_ids)
            if result is not None:
                logger.debug(
                    "Tier 1 exact KV hit for %d tokens",
                    result.num_tokens,
                )
                return result.num_tokens
        except Exception:
            logger.warning(
                "Failed to check Tier 1 KV cache for %d tokens",
                len(token_ids),
                exc_info=True,
            )

        # Tier 3: approximate match via DELTA tree
        if self._tier3_enabled:
            try:
                donor_result = self._segment_client.find_donor_and_plan(
                    target_tokens=token_ids,
                    min_reuse_ratio=self._min_reuse_ratio,
                )
                if donor_result.found and donor_result.plan is not None:
                    plan = donor_result.plan
                    if plan.viable:
                        self._last_transfer_plan = plan
                        logger.debug(
                            "Tier 3 approximate match: donor=%s, "
                            "reuse_ratio=%.2f, copied=%d, placeholders=%d",
                            plan.donor_id,
                            plan.reuse_ratio,
                            plan.num_copied,
                            plan.num_placeholders,
                        )
                        return plan.num_copied
                    logger.debug(
                        "Tier 3 plan not viable: reuse_ratio=%.2f < %.2f",
                        plan.reuse_ratio,
                        self._min_reuse_ratio,
                    )
            except Exception:
                logger.warning(
                    "Failed to check Tier 3 DELTA tree for %d tokens",
                    len(token_ids),
                    exc_info=True,
                )

        return 0

    def start_load_kv(self, token_ids: list[int]) -> np.ndarray | None:
        """Load cached KV state for the given token prefix.

        If a Tier 3 transfer plan exists from the scheduling phase,
        loads the donor's KV state and returns it along with the plan's
        copy/placeholder mask. The caller (vLLM worker) uses this to
        selectively recompute only placeholder positions.

        Otherwise, attempts a Tier 1 exact-match load.

        This is called by vLLM before prefill begins. The returned
        tensor (if any) is injected into the model's KV cache, avoiding
        redundant prefill computation.

        Args:
            token_ids: Token IDs for the prefix to load.

        Returns:
            Numpy array of shape [num_layers, 2, num_heads, num_tokens, head_dim]
            with dtype float16, or None if not cached.
        """
        if not token_ids:
            return None

        self._last_attention_plan = None

        # Tier 3: use the transfer plan from scheduling phase
        if self._last_transfer_plan is not None:
            plan = self._last_transfer_plan
            self._last_transfer_plan = None  # consume to prevent stale reuse
            try:
                logger.debug(
                    "Loading donor KV for Tier 3 transfer: "
                    "donor=%s, copy %d/%d positions",
                    plan.donor_id,
                    plan.num_copied,
                    plan.target_len,
                )

                # Load donor KV tensors by hash
                donor_result = self._token_client.load_kv_state_by_hash(
                    plan.donor_id
                )
                if donor_result is None:
                    logger.warning(
                        "Donor KV not found for hash=%s, "
                        "falling back to full prefill",
                        plan.donor_id,
                    )
                    return None

                # Build output tensor with donor KV at copied positions
                donor_kv = donor_result.kv_data
                n_layers = donor_result.num_layers
                n_heads = donor_result.num_heads
                head_dim = donor_result.head_dim
                target_len = plan.target_len

                output = np.zeros(
                    [n_layers, 2, n_heads, target_len, head_dim],
                    dtype=np.float16,
                )

                # Check if copy positions form a contiguous prefix.
                # Contiguous prefix [0,1,...,N-1] uses fast 1:1 copy
                # and doesn't need PartialAttention masks.
                copy_positions = sorted(plan.copy_positions)
                is_contiguous_prefix = copy_positions == list(
                    range(len(copy_positions))
                )

                if is_contiguous_prefix:
                    # Fast path: contiguous prefix, 1:1 donor→target
                    for target_pos in copy_positions:
                        if target_pos < donor_kv.shape[3]:
                            output[:, :, :, target_pos, :] = (
                                donor_kv[:, :, :, target_pos, :]
                            )
                    logger.debug(
                        "Tier 3 KV loaded (contiguous prefix): "
                        "%d/%d positions from donor=%s",
                        len(copy_positions),
                        target_len,
                        plan.donor_id,
                    )
                else:
                    # Sparse copy path: use slot_actions for exact
                    # donor_pos → target_pos mapping, then build a
                    # PartialAttention plan so the vLLM worker knows
                    # which positions need fresh Q,K,V computation.
                    self._apply_sparse_kv_copy(
                        output, donor_kv, plan, n_layers
                    )
                    logger.debug(
                        "Tier 3 KV loaded (sparse, PartialAttention): "
                        "%d/%d positions from donor=%s, "
                        "computation_ratio=%.2f",
                        len(copy_positions),
                        target_len,
                        plan.donor_id,
                        self._last_attention_plan.computation_ratio
                        if self._last_attention_plan
                        else 1.0,
                    )

                return output

            except Exception:
                logger.warning(
                    "Failed to load donor KV for transfer plan",
                    exc_info=True,
                )
                return None

        # Tier 1: exact match
        try:
            result = self._token_client.load_kv_state(token_ids)
            if result is not None:
                logger.debug(
                    "Loaded KV state: %d tokens, %d layers",
                    result.num_tokens,
                    result.num_layers,
                )
                return result.kv_data
        except Exception:
            logger.warning(
                "Failed to load KV state for %d tokens",
                len(token_ids),
                exc_info=True,
            )

        return None

    def _apply_sparse_kv_copy(
        self,
        output: np.ndarray,
        donor_kv: np.ndarray,
        plan: KvTransferPlan,
        num_layers: int,
    ) -> None:
        """Copy donor KV at non-contiguous positions and build attention plan.

        Uses the transfer plan's slot_actions for exact donor_pos → target_pos
        mapping, then builds a PartialAttentionPlan describing which positions
        need fresh Q,K,V computation vs reusing donor K,V.

        The attention plan is stored in ``_last_attention_plan`` for the vLLM
        worker to consume.
        """
        # Build donor_pos → target_pos mapping from slot_actions
        copy_map: dict[int, int] = {}
        slot_action_dicts: list[dict] = []

        for sa in plan.slot_actions:
            sa_dict = {
                "action": sa.action,
                "target_pos": sa.target_pos,
            }
            if sa.action == "copy_from_donor" and sa.donor_pos is not None:
                copy_map[sa.target_pos] = sa.donor_pos
                sa_dict["donor_pos"] = sa.donor_pos
            slot_action_dicts.append(sa_dict)

        # If no slot_actions, fall back to 1:1 mapping from copy_positions
        if not copy_map and plan.copy_positions:
            for tp in plan.copy_positions:
                copy_map[tp] = tp
            slot_action_dicts = [
                {
                    "action": "copy_from_donor",
                    "donor_pos": tp,
                    "target_pos": tp,
                }
                for tp in plan.copy_positions
            ] + [
                {"action": "placeholder", "target_pos": tp}
                for tp in plan.placeholder_positions
            ]

        # Copy donor KV at mapped positions
        for target_pos, donor_pos in copy_map.items():
            if (
                donor_pos < donor_kv.shape[3]
                and target_pos < output.shape[3]
            ):
                output[:, :, :, target_pos, :] = (
                    donor_kv[:, :, :, donor_pos, :]
                )

        # Build PartialAttention plan for the vLLM attention kernel
        self._last_attention_plan = build_attention_plan(
            donor_id=plan.donor_id,
            target_len=plan.target_len,
            donor_len=plan.donor_len,
            copy_positions=plan.copy_positions,
            placeholder_positions=plan.placeholder_positions,
            slot_actions=slot_action_dicts,
            num_layers=num_layers,
        )

    def start_save_kv(
        self,
        token_ids: list[int],
        kv_data: np.ndarray,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        request_id: str | None = None,
    ) -> None:
        """Save KV state after generation for future reuse.

        Persists the computed KV tensors into Synapse's cache so that
        subsequent requests with the same token prefix can skip prefill.
        If Tier 3 is enabled, also registers the request in the DELTA
        tree for approximate matching by future requests.

        This is called by vLLM after generation completes.

        Args:
            token_ids: Token IDs that produced this KV state.
            kv_data: KV tensor data as float16 numpy array.
                Shape: [num_layers, 2, num_heads, num_tokens, head_dim].
            num_layers: Number of transformer layers.
            num_heads: Number of KV attention heads per layer.
            head_dim: Dimension of each attention head.
            request_id: Optional unique request ID for DELTA tree
                registration. If None, uses the token hash.
        """
        if not token_ids:
            return

        token_hash = None
        try:
            resp = self._token_client.store_kv_state(
                token_ids=token_ids,
                kv_data=kv_data,
                num_layers=num_layers,
                num_heads=num_heads,
                head_dim=head_dim,
            )
            token_hash = resp.get("tokenHash")
            logger.debug(
                "Saved KV state: hash=%s, bytes=%d, overwrote=%s",
                token_hash or "unknown",
                resp.get("bytesStored", 0),
                resp.get("overwroteExisting", False),
            )
        except Exception:
            logger.warning(
                "Failed to save KV state for %d tokens",
                len(token_ids),
                exc_info=True,
            )

        # Register in DELTA tree for Tier 3 approximate matching
        if self._tier3_enabled:
            rid = request_id or token_hash or "unknown"
            try:
                delta_resp = self._segment_client.insert_delta_node(
                    request_id=rid,
                    token_ids=token_ids,
                    kv_resident=True,
                )
                logger.debug(
                    "Registered in DELTA tree: id=%s, tree_size=%d",
                    rid,
                    delta_resp.get("treeSize", 0),
                )
            except Exception:
                logger.warning(
                    "Failed to register request in DELTA tree",
                    exc_info=True,
                )

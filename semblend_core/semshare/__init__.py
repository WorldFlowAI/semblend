"""SemShareKV: Token-level LSH matching with per-layer selective recomputation.

Based on "SemShareKV: Efficient KVCache Sharing for Semantically Similar Prompts
via Token-Level LSH Matching" (arXiv:2509.24832).

This module implements token-level semantic KV cache sharing as a mode alongside
SemBlend's chunk-level approach. Feature flag: SEMBLEND_MODE=semshare
"""
from __future__ import annotations

from semblend_core.semshare.config import SemShareConfig
from semblend_core.semshare.lsh_index import LSHIndex, LSHTable
from semblend_core.semshare.token_matcher import TokenMatchResult, match_tokens
from semblend_core.semshare.hd_detector import HDDetectionResult, detect_hd_tokens
from semblend_core.semshare.layer_schedule import LayerSchedule, compute_layer_schedule
from semblend_core.semshare.attention_recovery import (
    AttentionRecoveryResult,
    compute_attention_recovery,
)
from semblend_core.semshare.kv_rearrange import KVRearrangePlan, build_rearrange_plan

__all__ = [
    "SemShareConfig",
    "LSHIndex",
    "LSHTable",
    "TokenMatchResult",
    "match_tokens",
    "HDDetectionResult",
    "detect_hd_tokens",
    "LayerSchedule",
    "compute_layer_schedule",
    "AttentionRecoveryResult",
    "compute_attention_recovery",
    "KVRearrangePlan",
    "build_rearrange_plan",
]

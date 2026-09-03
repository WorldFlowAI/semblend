"""Pipeline-level verified paraphrase serve.

High embedding similarity with near-zero token alignment is the
paraphrase signature. When the donor store rejects every candidate on
reuse ratio, the pipeline consults the paraphrase arbiter and, on an
accepted verdict, serves the donor span whole through an identity
position map — the engine-agnostic surface every connector already
consumes (segments for vLLM, contiguous results for SGLang).
"""

from __future__ import annotations

import numpy as np
import pytest

from semblend_core.donor_store import DonorStore
from semblend_core.pipeline import SemBlendPipeline

DIM = 8

DONOR_TEXT = (
    "the migration plan moves every customer workload to the new region "
    "before the maintenance window closes and keeps replication running "
    "so reads stay consistent while the cutover completes in stages "
    "across the fleet without interrupting any active session"
)
TARGET_TEXT = (
    "before the maintenance window closes the plan migrates all customer "
    "workloads to the new region with replication kept running for "
    "consistent reads while the staged cutover finishes fleet wide "
    "without any active session being interrupted"
)

DONOR_TOKENS = list(range(1000, 1600))
TARGET_TOKENS = list(range(5000, 5400))


class _StubEmbedder:
    """Same unit vector for every text: cosine similarity 1.0."""

    dimension = DIM

    def embed(self, text: str):
        return np.ones(DIM, dtype=np.float32) / np.sqrt(DIM)


def _pipeline(monkeypatch, **kwargs) -> SemBlendPipeline:
    monkeypatch.setenv("SEMBLEND_CHUNK_FAST_PATH", "0")
    store = DonorStore(
        max_entries=16,
        embedding_dim=DIM,
        min_similarity=0.60,
        chunk_size=32,
    )
    pipeline = SemBlendPipeline(
        embedder_type="jaccard",
        donor_store=store,
        chunk_size=32,
        enable_pq_segments=False,
        **kwargs,
    )
    pipeline._embedder = _StubEmbedder()  # noqa: SLF001
    return pipeline


def test_paraphrase_serve_returns_identity_span(monkeypatch) -> None:
    monkeypatch.setenv("SEMBLEND_PARAPHRASE_SERVE", "1")
    pipeline = _pipeline(monkeypatch)
    pipeline.register_donor("d1", DONOR_TOKENS, prompt_text=DONOR_TEXT)

    result = pipeline.find_donor(TARGET_TOKENS, prompt_text=TARGET_TEXT)

    assert result.found is True
    assert result.donor_id == "d1"
    assert result.confidence_tier == "paraphrase_verified"
    serve_len = result.position_map.num_pairs
    assert serve_len == len(TARGET_TOKENS) - 8  # donor longer: target cap - reserve
    assert result.position_map.donor_positions == list(range(serve_len))
    assert result.position_map.target_positions == list(range(serve_len))
    assert result.reuse_ratio == pytest.approx(serve_len / len(TARGET_TOKENS))
    assert result.donor_tokens == DONOR_TOKENS


def test_paraphrase_serve_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("SEMBLEND_PARAPHRASE_SERVE", raising=False)
    pipeline = _pipeline(monkeypatch)
    pipeline.register_donor("d1", DONOR_TOKENS, prompt_text=DONOR_TEXT)

    result = pipeline.find_donor(TARGET_TOKENS, prompt_text=TARGET_TEXT)

    assert result.found is False


def test_fact_divergent_target_rejected(monkeypatch) -> None:
    monkeypatch.setenv("SEMBLEND_PARAPHRASE_SERVE", "1")
    monkeypatch.delenv("SEMBLEND_NLI_APPEAL", raising=False)
    pipeline = _pipeline(monkeypatch)
    pipeline.register_donor(
        "d1", DONOR_TOKENS, prompt_text=DONOR_TEXT + " the budget is 12 million"
    )

    result = pipeline.find_donor(
        TARGET_TOKENS, prompt_text=TARGET_TEXT + " the budget is 15 million"
    )

    assert result.found is False


def test_short_donor_capped_by_donor_length(monkeypatch) -> None:
    monkeypatch.setenv("SEMBLEND_PARAPHRASE_SERVE", "1")
    pipeline = _pipeline(monkeypatch)
    short_donor = DONOR_TOKENS[:64]
    pipeline.register_donor("d1", short_donor, prompt_text=DONOR_TEXT)

    result = pipeline.find_donor(TARGET_TOKENS, prompt_text=TARGET_TEXT)

    assert result.found is True
    assert result.position_map.num_pairs == 64


def test_below_min_tokens_rejected(monkeypatch) -> None:
    monkeypatch.setenv("SEMBLEND_PARAPHRASE_SERVE", "1")
    pipeline = _pipeline(monkeypatch)
    pipeline.register_donor("d1", DONOR_TOKENS[:8], prompt_text=DONOR_TEXT)

    result = pipeline.find_donor(TARGET_TOKENS, prompt_text=TARGET_TEXT)

    assert result.found is False


def test_donor_text_retained_full_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("SEMBLEND_PARAPHRASE_SERVE", "1")
    pipeline = _pipeline(monkeypatch)
    pipeline.register_donor("d1", DONOR_TOKENS, prompt_text=DONOR_TEXT)
    node = pipeline._donor_store._entries["d1"]  # noqa: SLF001
    assert node.prompt_text == DONOR_TEXT


def test_donor_text_truncated_when_disabled(monkeypatch) -> None:
    monkeypatch.delenv("SEMBLEND_PARAPHRASE_SERVE", raising=False)
    pipeline = _pipeline(monkeypatch)
    pipeline.register_donor("d1", DONOR_TOKENS, prompt_text=DONOR_TEXT)
    node = pipeline._donor_store._entries["d1"]  # noqa: SLF001
    assert node.prompt_text == DONOR_TEXT[:200]


def test_namespace_isolation(monkeypatch) -> None:
    monkeypatch.setenv("SEMBLEND_PARAPHRASE_SERVE", "1")
    pipeline = _pipeline(monkeypatch)
    pipeline.register_donor(
        "d1", DONOR_TOKENS, prompt_text=DONOR_TEXT, extra_key="tenant-a"
    )

    result = pipeline.find_donor(
        TARGET_TOKENS, prompt_text=TARGET_TEXT, extra_key="tenant-b"
    )

    assert result.found is False


def test_probe_context_compares_full_text_and_offsets_donor(monkeypatch) -> None:
    """SGLang hands the pipeline only the suffix after the radix cache's
    exact prefix match; donors carry their full text. The verdict must see
    the request's full text (a shared chat template adds entities the
    suffix lacks) and served donor positions start after the prefix."""
    from semblend_core.pipeline import ProbeContext

    monkeypatch.setenv("SEMBLEND_PARAPHRASE_SERVE", "1")
    pipeline = _pipeline(monkeypatch)
    template = "You are Qwen, created by Alibaba Cloud. "
    pipeline.register_donor("d1", DONOR_TOKENS, prompt_text=template + DONOR_TEXT)

    suffix_only = pipeline.find_donor(TARGET_TOKENS, prompt_text=TARGET_TEXT)
    assert not suffix_only.found

    result = pipeline.find_donor(
        TARGET_TOKENS,
        prompt_text=TARGET_TEXT,
        probe=ProbeContext(text=template + TARGET_TEXT, donor_offset=9),
    )
    assert result.found and result.confidence_tier == "paraphrase_verified"
    assert result.position_map.target_positions[0] == 0
    assert result.position_map.donor_positions[0] == 9
    assert len(result.position_map.donor_positions) <= len(DONOR_TOKENS) - 9

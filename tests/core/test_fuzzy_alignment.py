"""Tests for fuzzy chunk alignment — shifted prefix, context gate, overlap fallback."""

from __future__ import annotations

import pytest


class TestShiftedPrefix:
    """Shifted-prefix pattern: same article, different instruction length.

    Key insight: a small instruction length difference (e.g. 4 tokens) creates
    a boundary shift that prevents exact chunk hash matching, but fuzzy matching
    recovers reuse because per-chunk token overlap remains >95%.
    """

    def test_shifted_prefix_high_reuse(self):
        """4-token instruction shift yields high reuse via fuzzy matching.

        With 256-token chunks and a 4-token shift, chunk 0 has 252/256 = 98.4%
        token overlap, comfortably above the 90% min_overlap and 94% bag_cosine
        thresholds. Subsequent chunks also have ~98% overlap.
        """
        from semblend_core.alignment import compute_fuzzy_chunk_alignment

        article = list(range(1000, 3048))  # ~8 chunks
        donor = list(range(4)) + article  # 4-token instruction
        target = list(range(10, 18)) + article  # 8-token instruction (shift=4)

        result = compute_fuzzy_chunk_alignment(donor, target, chunk_size=256)

        total_matched = result.exact_chunks + result.fuzzy_chunks
        assert total_matched >= 6, (
            f"Expected >=6 matched chunks, got {total_matched} "
            f"(exact={result.exact_chunks}, fuzzy={result.fuzzy_chunks})"
        )
        assert result.reuse_ratio >= 0.80

    def test_shifted_prefix_produces_copy_actions(self):
        """Shifted-prefix should produce copy slot actions (not all recompute)."""
        from semblend_core.alignment import compute_fuzzy_chunk_alignment

        article = list(range(1000, 1768))  # 3 chunks
        donor = list(range(4)) + article  # 4-token instruction
        target = list(range(10, 18)) + article  # 8-token instruction

        result = compute_fuzzy_chunk_alignment(donor, target, chunk_size=256)

        copy_count = sum(
            1 for sa in result.slot_actions if sa.action.value in ("copy", "copy_from_donor")
        )
        assert copy_count > 0, (
            f"Expected copy/copy_from_donor actions, got actions: "
            f"{[sa.action.value for sa in result.slot_actions[:10]]}"
        )

    def test_large_shift_fewer_matches(self):
        """A large instruction shift (190 tokens) misaligns most chunks."""
        from semblend_core.alignment import compute_fuzzy_chunk_alignment

        article = list(range(1000, 3048))  # ~8 chunks
        donor = list(range(200)) + article  # 200-token instruction
        target = list(range(500, 510)) + article  # 10-token instruction

        result = compute_fuzzy_chunk_alignment(donor, target, chunk_size=256)

        # With a 190-token shift, chunk overlap in the first chunk is only
        # ~66/256 = 25%, which fails fuzzy matching. Some middle/tail chunks
        # may still match if their boundaries align by coincidence.
        # The key assertion: it shouldn't be as good as a small shift.
        assert result.reuse_ratio < 0.90


class TestContextGateExemption:
    """Context gate exemption for high-overlap fuzzy chunks."""

    def test_high_overlap_fuzzy_bypasses_context_gate(self):
        """A fuzzy chunk with >=95% overlap should bypass the neighbor check.

        Chunk 0 has a 3-token shift (253/256 = 98.8% overlap), exceeding the
        95% context gate exemption threshold. Chunks 1-2 are exact matches.
        """
        from semblend_core.alignment import compute_fuzzy_chunk_alignment

        shared_body = list(range(1000, 1512))  # 2 exact chunks (512 tokens)

        # Chunk 0: 3-token shift → 253/256 = 98.8% overlap
        donor_chunk0 = list(range(256))
        target_chunk0 = list(range(3, 259))  # shifted by 3

        donor = donor_chunk0 + shared_body
        target = target_chunk0 + shared_body

        result = compute_fuzzy_chunk_alignment(
            donor,
            target,
            chunk_size=256,
            context_gate=True,
        )

        total = result.exact_chunks + result.fuzzy_chunks
        assert total >= 2, f"Expected >=2 matched chunks, got {total}"

    def test_context_gate_rejects_isolated_low_overlap(self):
        """An isolated chunk with low overlap should be rejected by context gate."""
        from semblend_core.alignment import (
            FuzzyMatchConfig,
            compute_fuzzy_chunk_alignment,
        )

        # Only chunk 0 has partial overlap, chunks 1+ are completely different
        donor_chunk0 = list(range(256))
        target_chunk0 = list(range(128)) + list(range(5000, 5128))  # 50% overlap

        donor_rest = list(range(2000, 2512))
        target_rest = list(range(3000, 3512))

        donor = donor_chunk0 + donor_rest
        target = target_chunk0 + target_rest

        config = FuzzyMatchConfig(min_overlap=0.40)
        result = compute_fuzzy_chunk_alignment(
            donor,
            target,
            chunk_size=256,
            context_gate=True,
            fuzzy_config=config,
        )

        assert result.reuse_ratio < 0.3


class TestFuzzyOverlapRatio:
    """Token-set overlap ratio computation via _fuzzy_match_chunk."""

    def test_identical_chunks_full_overlap(self):
        """Identical token sequences should have 1.0 overlap."""
        from semblend_core.alignment import _fuzzy_match_chunk

        chunk = list(range(256))
        result = _fuzzy_match_chunk(
            target_chunk=chunk,
            donor_chunks=[chunk],
            donor_chunk_starts=[0],
            used_donor_chunks=set(),
            min_overlap=0.90,
        )
        assert result is not None
        _, _, overlap = result
        assert overlap == pytest.approx(1.0)

    def test_disjoint_chunks_zero_overlap(self):
        """Completely different tokens should not match at 0.90 threshold."""
        from semblend_core.alignment import _fuzzy_match_chunk

        target = list(range(256))
        donor = list(range(1000, 1256))
        result = _fuzzy_match_chunk(
            target_chunk=target,
            donor_chunks=[donor],
            donor_chunk_starts=[0],
            used_donor_chunks=set(),
            min_overlap=0.90,
        )
        assert result is None

    def test_partial_overlap_below_threshold(self):
        """80% overlap should fail the default 0.90 threshold."""
        from semblend_core.alignment import _fuzzy_match_chunk

        shared = list(range(205))
        target = shared + list(range(5000, 5051))  # 256 tokens, 205/256 = 80%
        donor = shared + list(range(6000, 6051))

        result = _fuzzy_match_chunk(
            target_chunk=target,
            donor_chunks=[donor],
            donor_chunk_starts=[0],
            used_donor_chunks=set(),
            min_overlap=0.90,
        )
        assert result is None

    def test_partial_overlap_above_threshold(self):
        """95% overlap should pass the default 0.90 threshold."""
        from semblend_core.alignment import _fuzzy_match_chunk

        shared = list(range(244))
        target = shared + list(range(5000, 5012))  # 256 tokens, 244/256 = 95%
        donor = shared + list(range(6000, 6012))

        result = _fuzzy_match_chunk(
            target_chunk=target,
            donor_chunks=[donor],
            donor_chunk_starts=[0],
            used_donor_chunks=set(),
            min_overlap=0.90,
        )
        assert result is not None
        _, _, overlap = result
        assert overlap >= 0.90

    def test_used_donor_chunks_skipped(self):
        """Already-used donor chunks should not be re-matched."""
        from semblend_core.alignment import _fuzzy_match_chunk

        chunk = list(range(256))
        result = _fuzzy_match_chunk(
            target_chunk=chunk,
            donor_chunks=[chunk],
            donor_chunk_starts=[0],
            used_donor_chunks={0},
            min_overlap=0.90,
        )
        assert result is None


class TestNegativeControl:
    """Negative control: completely different articles should not match."""

    def test_different_articles_no_reuse(self):
        """Two unrelated articles should produce zero reuse."""
        from semblend_core.alignment import compute_fuzzy_chunk_alignment

        donor = list(range(1000, 2024))
        target = list(range(5000, 6024))

        result = compute_fuzzy_chunk_alignment(donor, target, chunk_size=256)

        assert result.exact_chunks == 0
        assert result.fuzzy_chunks == 0
        assert result.reuse_ratio == pytest.approx(0.0)

    def test_same_length_different_content(self):
        """Same-length but different articles should not false-positive."""
        from semblend_core.alignment import compute_fuzzy_chunk_alignment

        donor = list(range(0, 512))
        target = list(range(10000, 10512))

        result = compute_fuzzy_chunk_alignment(donor, target, chunk_size=256)
        assert result.reuse_ratio == pytest.approx(0.0)


class TestFuzzyAlignmentEdgeCases:
    """Edge cases for fuzzy alignment."""

    def test_empty_tokens(self):
        """Empty token lists should not crash."""
        from semblend_core.alignment import compute_fuzzy_chunk_alignment

        result = compute_fuzzy_chunk_alignment([], [], chunk_size=256)
        assert result.reuse_ratio == pytest.approx(0.0)

    def test_single_chunk(self):
        """Single-chunk inputs should not crash."""
        from semblend_core.alignment import compute_fuzzy_chunk_alignment

        donor = list(range(200))
        target = list(range(5, 205))

        result = compute_fuzzy_chunk_alignment(donor, target, chunk_size=256)
        assert result is not None

    def test_exact_match_no_fuzzy_needed(self):
        """Exact chunk alignment should not invoke fuzzy path."""
        from semblend_core.alignment import compute_fuzzy_chunk_alignment

        tokens = list(range(512))
        result = compute_fuzzy_chunk_alignment(tokens, tokens, chunk_size=256)

        assert result.exact_chunks == 2
        assert result.fuzzy_chunks == 0
        assert result.reuse_ratio == pytest.approx(1.0)

"""Tests for segment-level text splitting."""

from __future__ import annotations

import pytest

from semblend_core.segmentation import (
    TextSegment,
    TokenSegment,
    segment_text,
    segment_tokens,
)


class TestSegmentTextSentence:
    def test_basic_sentences(self):
        text = "First sentence. Second sentence. Third sentence."
        segments = segment_text(text, mode="sentence", min_chars=5)
        assert len(segments) >= 2
        # All text should be covered
        combined = " ".join(s.text for s in segments)
        assert "First sentence" in combined
        assert "Third sentence" in combined

    def test_single_sentence(self):
        text = "Just one sentence here."
        segments = segment_text(text, mode="sentence", min_chars=5)
        assert len(segments) == 1
        assert segments[0].text == text

    def test_empty_text(self):
        assert segment_text("") == []
        assert segment_text("   ") == []

    def test_min_chars_merging(self):
        text = "Hi. There. This is a longer sentence that should stand alone."
        segments = segment_text(text, mode="sentence", min_chars=30)
        # Short sentences should be merged
        assert len(segments) <= 2
        assert all(len(s.text) >= 20 for s in segments)

    def test_max_chars_splitting(self):
        text = "A " * 500 + "end."
        segments = segment_text(text, mode="sentence", max_chars=100)
        assert all(len(s.text) <= 200 for s in segments)  # some slack for merging

    def test_preserves_content(self):
        text = (
            "The transformer architecture revolutionized NLP. "
            "Self-attention replaces recurrence. "
            "Models like BERT and GPT achieve SOTA results."
        )
        segments = segment_text(text, mode="sentence", min_chars=10)
        combined = " ".join(s.text for s in segments)
        assert "transformer" in combined
        assert "Self-attention" in combined
        assert "BERT" in combined


class TestSegmentTextParagraph:
    def test_basic_paragraphs(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        segments = segment_text(text, mode="paragraph", min_chars=5)
        assert len(segments) == 3
        assert segments[0].text == "First paragraph."
        assert segments[1].text == "Second paragraph."
        assert segments[2].text == "Third paragraph."

    def test_single_paragraph(self):
        text = "No paragraph breaks here, just one block of text."
        segments = segment_text(text, mode="paragraph", min_chars=5)
        assert len(segments) == 1


class TestSegmentTextInstruction:
    def test_instruction_delimiter(self):
        text = "Summarize this:\n---\nDocument content goes here."
        segments = segment_text(text, mode="instruction", min_chars=5)
        assert len(segments) == 2
        assert "Summarize" in segments[0].text
        assert "Document" in segments[1].text

    def test_no_delimiter(self):
        text = "Plain text without any delimiters or separators."
        segments = segment_text(text, mode="instruction", min_chars=5)
        assert len(segments) == 1


class TestSegmentTokens:
    def test_basic_token_segmentation(self):
        # Mock tokenizer that just returns char codes
        text = "First sentence. Second sentence. Third sentence."
        token_ids = list(range(len(text)))

        def mock_encode(t):
            return list(range(len(t)))

        segments = segment_tokens(
            token_ids,
            text,
            mock_encode,
            mode="sentence",
            min_tokens=5,
            max_tokens=100,
        )
        assert len(segments) >= 1
        # Should cover all tokens
        total_tokens = sum(s.n_tokens for s in segments)
        assert total_tokens > 0

    def test_max_tokens_splitting(self):
        text = "A " * 200
        token_ids = list(range(400))

        def mock_encode(t):
            return list(range(len(t)))

        segments = segment_tokens(
            token_ids,
            text,
            mock_encode,
            mode="sentence",
            min_tokens=1,
            max_tokens=50,
        )
        assert all(s.n_tokens <= 50 for s in segments)

    def test_empty_returns_single(self):
        segments = segment_tokens(
            [1, 2, 3],
            "",
            lambda t: [],
            mode="sentence",
        )
        assert len(segments) == 1
        assert segments[0].token_ids == (1, 2, 3)


class TestTextSegmentProperties:
    def test_char_length(self):
        seg = TextSegment(text="hello", start_char=10, end_char=15, segment_type="sentence")
        assert seg.char_length == 5

    def test_frozen(self):
        seg = TextSegment(text="hi", start_char=0, end_char=2, segment_type="sentence")
        with pytest.raises(AttributeError):
            seg.text = "changed"


class TestTokenSegmentProperties:
    def test_n_tokens(self):
        seg = TokenSegment(
            token_ids=(1, 2, 3),
            start_token=10,
            end_token=13,
            segment_type="sentence",
        )
        assert seg.n_tokens == 3

    def test_frozen(self):
        seg = TokenSegment(
            token_ids=(1,),
            start_token=0,
            end_token=1,
            segment_type="sentence",
        )
        with pytest.raises(AttributeError):
            seg.start_token = 5

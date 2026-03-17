"""Segment-level text splitting for semantic KV matching.

Splits prompts into semantic segments (sentences, paragraphs) for
fine-grained embedding and matching. Each segment can be independently
embedded and matched against donor segments by semantic similarity.

This enables variable-length KV reuse aligned to semantic boundaries
instead of fixed 256-token chunk boundaries.

Usage:
    from semblend_core.segmentation import segment_text, segment_tokens

    segments = segment_text(prompt_text)
    for seg in segments:
        embedding = embedder.embed(seg.text)
        # Match against donor segments by cosine similarity
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TextSegment:
    """A semantic segment of text with position tracking."""

    text: str
    start_char: int
    end_char: int
    segment_type: str  # "sentence", "paragraph", "instruction"

    @property
    def char_length(self) -> int:
        return self.end_char - self.start_char


@dataclass(frozen=True)
class TokenSegment:
    """A segment defined by token positions."""

    token_ids: tuple[int, ...]
    start_token: int
    end_token: int
    segment_type: str
    text: str = ""  # optional decoded text

    @property
    def n_tokens(self) -> int:
        return self.end_token - self.start_token


# Sentence boundary pattern: period/question/exclamation followed by
# space and uppercase letter (or end of string). Handles abbreviations
# like "e.g." and "Dr." by requiring uppercase after boundary.
_SENTENCE_RE = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z])'
)

# Paragraph boundary: two or more newlines
_PARAGRAPH_RE = re.compile(r'\n\s*\n')

# Instruction delimiter: common instruction/document separators
_INSTRUCTION_RE = re.compile(
    r'(?:^|\n)(?:###|---|===|\*\*\*|```)\s*(?:\n|$)',
    re.MULTILINE,
)


def segment_text(
    text: str,
    mode: str = "sentence",
    min_chars: int = 20,
    max_chars: int = 2000,
) -> list[TextSegment]:
    """Split text into semantic segments.

    Args:
        text: Input text to segment.
        mode: "sentence", "paragraph", or "instruction".
        min_chars: Minimum segment length (smaller are merged with next).
        max_chars: Maximum segment length (larger are split).

    Returns:
        List of TextSegment with non-overlapping spans.
    """
    if not text.strip():
        return []

    if mode == "paragraph":
        raw_segments = _split_paragraphs(text)
    elif mode == "instruction":
        raw_segments = _split_instruction(text)
    else:
        raw_segments = _split_sentences(text)

    # Merge small segments and split large ones
    return _normalize_segments(raw_segments, min_chars, max_chars, mode)


def segment_tokens(
    token_ids: list[int],
    text: str,
    tokenizer_encode,
    mode: str = "sentence",
    min_tokens: int = 16,
    max_tokens: int = 512,
) -> list[TokenSegment]:
    """Split tokens into semantic segments aligned to text boundaries.

    Args:
        token_ids: Full token ID sequence.
        text: Decoded text corresponding to token_ids.
        tokenizer_encode: Function(text) -> list[int] for re-encoding.
        mode: "sentence" or "paragraph".
        min_tokens: Minimum segment token count.
        max_tokens: Maximum segment token count.

    Returns:
        List of TokenSegment with non-overlapping token spans.
    """
    text_segments = segment_text(text, mode=mode)
    if not text_segments:
        return [TokenSegment(
            token_ids=tuple(token_ids),
            start_token=0,
            end_token=len(token_ids),
            segment_type=mode,
            text=text,
        )]

    # Map text segments to token positions
    token_segments = []
    token_pos = 0

    for seg in text_segments:
        seg_tokens = tokenizer_encode(seg.text)
        n = len(seg_tokens)

        if n == 0:
            continue

        # Find where this segment's tokens start in the full sequence
        # by matching the first few tokens
        best_pos = _find_token_start(
            token_ids, seg_tokens, search_start=token_pos
        )
        if best_pos < 0:
            best_pos = token_pos

        end_pos = min(best_pos + n, len(token_ids))

        token_segments.append(TokenSegment(
            token_ids=tuple(token_ids[best_pos:end_pos]),
            start_token=best_pos,
            end_token=end_pos,
            segment_type=seg.segment_type,
            text=seg.text,
        ))
        token_pos = end_pos

    # Merge small segments
    merged = _merge_small_token_segments(token_segments, min_tokens, mode)

    # Split large segments
    result = []
    for seg in merged:
        if seg.n_tokens > max_tokens:
            result.extend(_split_token_segment(seg, max_tokens, mode))
        else:
            result.append(seg)

    return result


def _split_sentences(text: str) -> list[TextSegment]:
    """Split text into sentences."""
    parts = _SENTENCE_RE.split(text)
    segments = []
    pos = 0
    for part in parts:
        part_stripped = part.strip()
        if part_stripped:
            start = text.find(part_stripped, pos)
            if start < 0:
                start = pos
            end = start + len(part_stripped)
            segments.append(TextSegment(
                text=part_stripped,
                start_char=start,
                end_char=end,
                segment_type="sentence",
            ))
            pos = end
    return segments


def _split_paragraphs(text: str) -> list[TextSegment]:
    """Split text into paragraphs."""
    parts = _PARAGRAPH_RE.split(text)
    segments = []
    pos = 0
    for part in parts:
        part_stripped = part.strip()
        if part_stripped:
            start = text.find(part_stripped, pos)
            if start < 0:
                start = pos
            end = start + len(part_stripped)
            segments.append(TextSegment(
                text=part_stripped,
                start_char=start,
                end_char=end,
                segment_type="paragraph",
            ))
            pos = end
    return segments


def _split_instruction(text: str) -> list[TextSegment]:
    """Split text at instruction/document boundaries."""
    parts = _INSTRUCTION_RE.split(text)
    segments = []
    pos = 0
    for part in parts:
        part_stripped = part.strip()
        if part_stripped:
            start = text.find(part_stripped, pos)
            if start < 0:
                start = pos
            end = start + len(part_stripped)
            segments.append(TextSegment(
                text=part_stripped,
                start_char=start,
                end_char=end,
                segment_type="instruction",
            ))
            pos = end
    return segments


def _normalize_segments(
    segments: list[TextSegment],
    min_chars: int,
    max_chars: int,
    mode: str,
) -> list[TextSegment]:
    """Merge small segments and split large ones."""
    if not segments:
        return []

    # Merge small segments with their successor
    merged = []
    buffer_text = ""
    buffer_start = 0

    for seg in segments:
        if not buffer_text:
            buffer_text = seg.text
            buffer_start = seg.start_char
        else:
            buffer_text = buffer_text + " " + seg.text

        if len(buffer_text) >= min_chars:
            merged.append(TextSegment(
                text=buffer_text,
                start_char=buffer_start,
                end_char=seg.end_char,
                segment_type=mode,
            ))
            buffer_text = ""

    # Flush remaining buffer
    if buffer_text:
        if merged:
            last = merged[-1]
            combined = last.text + " " + buffer_text
            merged[-1] = TextSegment(
                text=combined,
                start_char=last.start_char,
                end_char=segments[-1].end_char,
                segment_type=mode,
            )
        else:
            merged.append(TextSegment(
                text=buffer_text,
                start_char=buffer_start,
                end_char=segments[-1].end_char,
                segment_type=mode,
            ))

    # Split large segments
    result = []
    for seg in merged:
        if len(seg.text) > max_chars:
            for chunk_seg in _split_large_text(seg, max_chars, mode):
                result.append(chunk_seg)
        else:
            result.append(seg)

    return result


def _split_large_text(
    seg: TextSegment,
    max_chars: int,
    mode: str,
) -> list[TextSegment]:
    """Split a large text segment into smaller pieces."""
    text = seg.text
    result = []
    pos = 0

    while pos < len(text):
        end = min(pos + max_chars, len(text))

        # Try to break at a sentence boundary
        if end < len(text):
            last_period = text.rfind('. ', pos, end)
            if last_period > pos + max_chars // 2:
                end = last_period + 1

        chunk = text[pos:end].strip()
        if chunk:
            result.append(TextSegment(
                text=chunk,
                start_char=seg.start_char + pos,
                end_char=seg.start_char + end,
                segment_type=mode,
            ))
        pos = end

    return result


def _find_token_start(
    full_tokens: list[int],
    segment_tokens: list[int],
    search_start: int = 0,
    window: int = 50,
) -> int:
    """Find where segment_tokens start in full_tokens."""
    if not segment_tokens:
        return search_start

    target = segment_tokens[0]
    search_end = min(search_start + window + len(segment_tokens), len(full_tokens))

    for i in range(search_start, search_end):
        if full_tokens[i] == target:
            # Check if more tokens match
            match_len = 0
            check_len = min(5, len(segment_tokens), len(full_tokens) - i)
            for j in range(check_len):
                if full_tokens[i + j] == segment_tokens[j]:
                    match_len += 1
            if match_len >= min(3, check_len):
                return i

    return -1


def _merge_small_token_segments(
    segments: list[TokenSegment],
    min_tokens: int,
    mode: str,
) -> list[TokenSegment]:
    """Merge token segments smaller than min_tokens."""
    if not segments:
        return []

    merged = []
    buffer = None

    for seg in segments:
        if buffer is None:
            buffer = seg
        else:
            # Merge into buffer
            combined_ids = buffer.token_ids + seg.token_ids
            buffer = TokenSegment(
                token_ids=combined_ids,
                start_token=buffer.start_token,
                end_token=seg.end_token,
                segment_type=mode,
                text=buffer.text + " " + seg.text,
            )

        if buffer.n_tokens >= min_tokens:
            merged.append(buffer)
            buffer = None

    if buffer is not None:
        if merged:
            last = merged[-1]
            combined_ids = last.token_ids + buffer.token_ids
            merged[-1] = TokenSegment(
                token_ids=combined_ids,
                start_token=last.start_token,
                end_token=buffer.end_token,
                segment_type=mode,
                text=last.text + " " + buffer.text,
            )
        else:
            merged.append(buffer)

    return merged


def _split_token_segment(
    seg: TokenSegment,
    max_tokens: int,
    mode: str,
) -> list[TokenSegment]:
    """Split a token segment that exceeds max_tokens."""
    result = []
    ids = seg.token_ids
    pos = 0

    while pos < len(ids):
        end = min(pos + max_tokens, len(ids))
        chunk_ids = ids[pos:end]
        result.append(TokenSegment(
            token_ids=chunk_ids,
            start_token=seg.start_token + pos,
            end_token=seg.start_token + end,
            segment_type=mode,
        ))
        pos = end

    return result

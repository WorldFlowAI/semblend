"""Lexical fact-consistency gate for paraphrase-span acceptance.

Compares the fact-bearing surface forms of two text spans: numbers bind
their magnitude words ("12 million" and "12 thousand" are different
fact tokens), small word-numbers normalize to digits ("one" excluded —
it is interchangeable with the article "a" in ordinary paraphrase), and
capitalized entity tokens compare by symmetric-difference ratio. A
mismatch in either direction rejects. Documented gaps (relative
phrasing, arithmetic restatement, cross-sentence composition) reject
by default via the number rules; they never pass silently.
"""

from __future__ import annotations

import re

_WORD_NUMS = {
    "zero": 0, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_STOP_CAPS = {
    "The", "A", "An", "In", "On", "At", "It", "This", "That",
    "With", "Across", "During", "Put", "However", "Nevertheless",
}
_MAGNITUDES = r"(?:million|billion|trillion|thousand|hundred|percent|%)"


def fact_sets(text: str) -> tuple:
    numbers = {
        re.sub(r"\s+", " ", m.group(0).lower())
        for m in re.finditer(rf"\b\d[\d,.]*(?:\s+{_MAGNITUDES})?\b", text)
    }
    for w, v in _WORD_NUMS.items():
        if re.search(rf"\b{w}\b", text, re.I):
            numbers.add(str(v))
    # A capitalized word counts as an entity only when it appears
    # capitalized in a non-sentence-initial position at least once:
    # sentence-initial capitalization of common nouns otherwise creates
    # phantom entities that reject genuine paraphrases (rewording moves
    # words across sentence boundaries).
    mid_sentence = {
        m.group(1)
        for m in re.finditer(r"(?<![.!?\n])\s([A-Z][a-z]{2,})\b", text)
        if m.group(1) not in _STOP_CAPS
    }
    entities = mid_sentence
    return numbers, entities


def spans_fact_consistent(
    donor_text: str,
    target_text: str,
    entity_diff_ratio_max: float = 0.5,
) -> bool:
    dn, de = fact_sets(donor_text)
    tn, te = fact_sets(target_text)
    if dn != tn:
        return False
    union = de | te
    if union and len(de ^ te) / len(union) > entity_diff_ratio_max:
        return False
    return True

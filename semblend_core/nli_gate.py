"""Sentence-aligned meaning-consistency gate (the learned verifier tier).

Layered behind the lexical fact gate: sentences are paired across the
donor and target spans by cheap token-set similarity, each aligned pair
is scored with bidirectional NLI entailment in ONE batched call, and
the span is accepted only if every aligned pair clears the entailment
floor and enough of the target is covered by aligned donor content.
Fail-closed on every path: low coverage, any weak pair, or a missing
model dependency all reject.

Scale design: pairs for a span batch into a single cross-encoder
forward; verdicts are cacheable by (donor identity, target span hash)
upstream since donor text is immutable after registration; the model is
small enough to colocate with serving. Cost is bounded by sentences per
span, not span length.
"""

from __future__ import annotations

import re

_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
_ENTAIL_FLOOR = 0.5
_ALIGN_FLOOR = 0.2
_COVERAGE_FLOOR = 0.7


def _sentences(text: str) -> list:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _jaccard(a: str, b: str) -> float:
    sa = set(re.findall(r"[\w]+", a.lower()))
    sb = set(re.findall(r"[\w]+", b.lower()))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


class SentenceAlignedNliGate:
    def __init__(
        self,
        entail_floor: float = _ENTAIL_FLOOR,
        align_floor: float = _ALIGN_FLOOR,
        coverage_floor: float = _COVERAGE_FLOOR,
        model_name: str = _MODEL_NAME,
    ):
        self._entail_floor = entail_floor
        self._align_floor = align_floor
        self._coverage_floor = coverage_floor
        self._model_name = model_name
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
        return self._model

    def align(self, donor_text: str, target_text: str) -> tuple:
        """Pair each target sentence with its most-similar donor sentence.

        Returns (pairs, coverage): pairs of (target_sentence,
        donor_sentence) whose similarity clears the alignment floor, and
        the fraction of target sentences that found a partner.
        """
        d_sents = _sentences(donor_text)
        t_sents = _sentences(target_text)
        if not d_sents or not t_sents:
            return [], 0.0
        pairs = []
        matched = 0
        for t in t_sents:
            best = max(d_sents, key=lambda d: _jaccard(t, d))
            if _jaccard(t, best) >= self._align_floor:
                pairs.append((t, best))
                matched += 1
        return pairs, matched / len(t_sents)

    def spans_meaning_consistent(self, donor_text: str, target_text: str) -> bool:
        """Fail-closed sentence-aligned bidirectional entailment."""
        pairs, coverage = self.align(donor_text, target_text)
        if coverage < self._coverage_floor or not pairs:
            return False
        try:
            model = self._ensure_model()
        except Exception:
            return False
        batch = []
        for t, d in pairs:
            batch.append((d, t))
            batch.append((t, d))
        scores = model.predict(batch, apply_softmax=True)
        # label order for the nli cross-encoders: contradiction, entailment, neutral
        for i in range(0, len(scores), 2):
            fwd = float(scores[i][1])
            rev = float(scores[i + 1][1])
            if min(fwd, rev) < self._entail_floor:
                return False
        return True

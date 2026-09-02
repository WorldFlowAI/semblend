"""Engine-agnostic verified paraphrase-serve arbiter.

Decides whether a donor span may be served verbatim for a semantically
equivalent target span. Composition is fail-closed OR: the lexical fact
gate accepts outright, and only its rejections are appealed to the
sentence-aligned NLI gate — the appeal rescues surface rewordings
(number words, date formats) the lexical gate cannot represent, while
strict conjunction wrongly rejects most true paraphrases.

Verdicts are memoized by span content digests, never by donor id:
donor ids churn on every registration while the underlying spans
repeat, so content keys are the only ones that ever hit.
"""

from __future__ import annotations

import hashlib
import logging
import os

logger = logging.getLogger(__name__)

_DEFAULT_MEMO_CAPACITY = 4096


def paraphrase_serve_enabled() -> bool:
    return os.environ.get("SEMBLEND_PARAPHRASE_SERVE", "") == "1"


def nli_appeal_enabled() -> bool:
    return os.environ.get("SEMBLEND_NLI_APPEAL", "") == "1"


def _span_digest(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


class ParaphraseArbiter:
    """Fact gate with NLI appeal, memoized by span content.

    Args:
        nli_gate: Optional pre-built appeal gate exposing
            ``spans_meaning_consistent(donor_text, target_text)``. When
            None and the appeal is enabled, a ``SentenceAlignedNliGate``
            is built lazily on first appeal.
        nli_enabled: Appeal availability; defaults to the
            ``SEMBLEND_NLI_APPEAL`` environment gate.
        memo_capacity: Maximum memoized verdicts (insertion-order
            eviction).
    """

    def __init__(
        self,
        nli_gate: object | None = None,
        nli_enabled: bool | None = None,
        memo_capacity: int = _DEFAULT_MEMO_CAPACITY,
    ) -> None:
        self._nli_gate = nli_gate
        self._nli_enabled = (
            nli_appeal_enabled() if nli_enabled is None else bool(nli_enabled)
        )
        self._nli_gate_failed = False
        self._memo: dict[tuple[bytes, bytes, bool], bool] = {}
        self._memo_capacity = max(1, int(memo_capacity))

    def verdict(self, donor_text: str, target_text: str) -> bool:
        """Fail-closed serve decision for a donor/target span pair."""
        if not donor_text or not target_text:
            return False
        key = (
            _span_digest(donor_text),
            _span_digest(target_text),
            self._nli_enabled,
        )
        cached = self._memo.get(key)
        if cached is not None:
            return cached
        accepted = self._decide(donor_text, target_text)
        if len(self._memo) >= self._memo_capacity:
            self._memo.pop(next(iter(self._memo)))
        self._memo[key] = accepted
        return accepted

    def _decide(self, donor_text: str, target_text: str) -> bool:
        from semblend_core.fact_gate import spans_fact_consistent

        if spans_fact_consistent(donor_text, target_text):
            return True
        if not self._nli_enabled:
            return False
        return self._appeal(donor_text, target_text)

    def _appeal(self, donor_text: str, target_text: str) -> bool:
        gate = self._resolve_gate()
        if gate is None:
            return False
        try:
            verdict = bool(gate.spans_meaning_consistent(donor_text, target_text))
        except Exception:
            logger.warning("paraphrase nli appeal failed", exc_info=True)
            return False
        logger.info("paraphrase nli appeal verdict=%s", verdict)
        return verdict

    def _resolve_gate(self) -> object | None:
        if self._nli_gate is not None:
            return self._nli_gate
        if self._nli_gate_failed:
            return None
        try:
            from semblend_core.nli_gate import SentenceAlignedNliGate

            self._nli_gate = SentenceAlignedNliGate()
        except Exception:
            logger.warning("paraphrase nli appeal unavailable", exc_info=True)
            self._nli_gate_failed = True
            return None
        return self._nli_gate

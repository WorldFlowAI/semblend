"""The MiniLM embedder loads lazily; the SGLang adapter registers donors on a
background thread while lookups run on the caller's thread, so the first
embed calls race. A lookup that observed the half-initialized embedder
raised AttributeError on ``self._model.tokenizer`` and degraded to a miss."""

from __future__ import annotations

import threading
import time

import numpy as np

from semblend_core.embedder import MiniLMEmbedder


class _SlowFakeModel:
    class tokenizer:  # noqa: N801 - mimics SentenceTransformer's attribute
        @staticmethod
        def encode(text, add_special_tokens=False):
            return list(range(len(text) // 4))

        @staticmethod
        def decode(ids, skip_special_tokens=False):
            return "x" * (4 * len(ids))

    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def encode(self, texts, **kwargs):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.02)
        with self.lock:
            self.active -= 1
        if isinstance(texts, str):
            return np.ones(384, dtype=np.float32) / np.sqrt(384)
        return np.ones((len(texts), 384), dtype=np.float32) / np.sqrt(384)


def test_concurrent_first_embed_waits_for_model_load(monkeypatch):
    embedder = MiniLMEmbedder()

    def slow_init(self):
        time.sleep(0.3)
        self._model = _SlowFakeModel()
        self._available = True

    monkeypatch.setattr(MiniLMEmbedder, "_init", slow_init)

    long_text = "word " * 3000  # forces the segmented path (> 512 est. tokens)
    results: list = []
    errors: list = []

    def worker():
        try:
            results.append(embedder.embed(long_text))
        except Exception as exc:  # pragma: no cover - the regression itself
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(results) == 4 and all(r is not None for r in results)


def test_concurrent_embeds_are_serialized(monkeypatch):
    """SentenceTransformer's fast tokenizer is reconfigured on every encode
    call; overlapping calls from the registration thread and a lookup raised
    RuntimeError("Already borrowed") and the lookup degraded to a miss."""
    embedder = MiniLMEmbedder()
    model = _SlowFakeModel()

    def quick_init(self):
        self._model = model
        self._available = True

    monkeypatch.setattr(MiniLMEmbedder, "_init", quick_init)
    texts = ["short text", "word " * 3000, "another short one", "token " * 2500]

    def worker(text):
        if len(text) > 100:
            embedder.embed_with_segments(text, chunk_size=256)
        else:
            embedder.embed(text)

    threads = [threading.Thread(target=worker, args=(t,)) for t in texts * 2]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert model.max_active == 1


def test_failed_load_marks_unavailable_once(monkeypatch):
    embedder = MiniLMEmbedder()
    calls = {"n": 0}

    def failing_init(self):
        calls["n"] += 1
        self._available = False

    monkeypatch.setattr(MiniLMEmbedder, "_init", failing_init)
    assert embedder.embed("hello") is None
    assert embedder.embed("hello") is None
    assert calls["n"] == 1

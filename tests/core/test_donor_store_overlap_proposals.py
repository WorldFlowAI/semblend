"""Candidate proposal must not depend on cosine rank alone: with many
near-duplicate donors (same template, different numbers) MiniLM ranks are
near-ties and the donor that actually shares the tokens can fall outside
the top-k, so the wrapper-shift lane missed 21 of 24 items on GPU. Exact chunk
overlap proposes the donor that shares the tokens."""

from __future__ import annotations

import random
import time

import numpy as np

from semblend_core.donor_store import DonorNode, DonorStore


def _unit(seed: int, base: np.ndarray, noise: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = base + noise * rng.standard_normal(base.shape)
    return v / np.linalg.norm(v)


def test_exact_overlap_donor_is_proposed_outside_cosine_top_k():
    dim = 384
    store = DonorStore(max_entries=100, embedding_dim=dim, min_similarity=0.60, chunk_size=16)
    base = _unit(0, np.ones(dim), 0.0)
    rng = random.Random(7)
    # Twenty decoys: embeddings very close to the query, tokens unrelated.
    for i in range(20):
        toks = [rng.randint(5, 30000) for _ in range(640)]
        store.add_donor(DonorNode(request_id=f"decoy-{i}", token_ids=toks, embedding=_unit(i + 1, base, 0.01), timestamp=time.time()))
    # The real donor: identical evidence block, embedding slightly further away.
    evidence = [rng.randint(5, 30000) for _ in range(640)]
    store.add_donor(DonorNode(request_id="real", token_ids=list(range(100, 116)) + evidence, embedding=_unit(99, base, 0.03), timestamp=time.time()))

    query = list(range(200, 232)) + evidence  # wrapper differs by one whole chunk
    match = store.find_donor(query_embedding=base, query_tokens=query, top_k=5, min_reuse_ratio=0.5)
    assert match is not None
    assert match.donor.request_id == "real"


def test_best_alignment_wins_when_fast_estimate_ranks_a_near_duplicate_first(monkeypatch):
    # The fast estimate is grid-dependent: a wrapper shift that is not a
    # multiple of the chunk size can score the true donor below a decoy
    # that shares grid-aligned chunks. Aligning only the top estimate then
    # serves the decoy (reuse near zero after the gate) and the true donor
    # is never tried. On GPU this turned the same-document lane into a
    # paraphrase serve for 22 of 24 items.
    import semblend_core.donor_store as ds

    dim = 384
    store = DonorStore(max_entries=100, embedding_dim=dim, min_similarity=0.60, chunk_size=16)
    base = _unit(0, np.ones(dim), 0.0)
    rng = random.Random(11)
    evidence = [rng.randint(5, 30000) for _ in range(640)]
    query = list(range(200, 232)) + evidence
    decoy = query[:256] + [rng.randint(5, 30000) for _ in range(len(query) - 256)]
    store.add_donor(DonorNode(request_id="decoy", token_ids=decoy, embedding=_unit(1, base, 0.01), timestamp=time.time()))
    store.add_donor(DonorNode(request_id="real", token_ids=list(range(100, 115)) + evidence, embedding=_unit(2, base, 0.01), timestamp=time.time()))

    monkeypatch.setattr(ds, "estimate_reuse_ratio", lambda d, t, chunk_size=16: 0.4 if d == decoy else 0.1)
    match = store.find_donor(query_embedding=base, query_tokens=query, top_k=5, min_reuse_ratio=0.0)
    assert match is not None
    assert match.donor.request_id == "real"
    assert match.alignment.reuse_ratio > 0.9

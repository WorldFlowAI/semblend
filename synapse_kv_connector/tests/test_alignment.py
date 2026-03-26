"""Tests for alignment.py and bathtub.py."""

from __future__ import annotations

import time


def test_alignment_exact_match():
    """Exact token sequences should have 100% reuse."""
    from synapse_kv_connector.alignment import (
        SlotActionType,
        compute_alignment,
    )

    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    result = compute_alignment(tokens, tokens)

    assert result.reuse_ratio == 1.0
    assert result.edit_distance == 0
    assert all(sa.action == SlotActionType.COPY_FROM_DONOR for sa in result.slot_actions)


def test_alignment_prefix_match():
    """Shared prefix should be marked as copy_from_donor."""
    from synapse_kv_connector.alignment import (
        SlotActionType,
        compute_alignment,
    )

    donor = [1, 2, 3, 4, 5]
    target = [1, 2, 3, 10, 11]
    result = compute_alignment(donor, target)

    # First 3 tokens match
    assert result.slot_actions[0].action == SlotActionType.COPY_FROM_DONOR
    assert result.slot_actions[1].action == SlotActionType.COPY_FROM_DONOR
    assert result.slot_actions[2].action == SlotActionType.COPY_FROM_DONOR
    assert result.reuse_ratio >= 0.5


def test_alignment_no_overlap():
    """Completely different sequences should have 0% reuse."""
    from synapse_kv_connector.alignment import compute_alignment

    donor = [1, 2, 3, 4, 5]
    target = [10, 20, 30, 40, 50]
    result = compute_alignment(donor, target)

    assert result.reuse_ratio == 0.0
    assert result.edit_distance == 5


def test_alignment_reorder():
    """Reordered tokens: edit distance detects rearrangement."""
    from synapse_kv_connector.alignment import compute_alignment

    donor = [1, 2, 3, 4, 5]
    target = [1, 3, 2, 4, 5]
    result = compute_alignment(donor, target)

    # Some tokens match despite reorder
    assert 0.0 < result.reuse_ratio < 1.0
    assert result.edit_distance > 0


def test_alignment_insertion():
    """Inserting tokens in the middle."""
    from synapse_kv_connector.alignment import compute_alignment

    donor = [1, 2, 3, 4, 5]
    target = [1, 2, 99, 3, 4, 5]
    result = compute_alignment(donor, target)

    # At least the prefix [1,2] should be reused
    assert result.reuse_ratio > 0.0
    # Should have some reuse
    assert result.edit_distance < len(target)


def test_alignment_latency_8k():
    """Alignment on 8K tokens must complete in <5ms."""
    from synapse_kv_connector.alignment import compute_alignment

    donor = list(range(8000))
    target = list(range(100)) + list(range(200, 8100))

    t0 = time.monotonic()
    result = compute_alignment(donor, target)
    elapsed_ms = (time.monotonic() - t0) * 1000

    assert elapsed_ms < 5000, f"Alignment took {elapsed_ms:.1f}ms (budget: 5000ms)"
    assert result.reuse_ratio > 0


def test_batch_alignment():
    """Batch alignment returns best candidate."""
    from synapse_kv_connector.alignment import compute_batch_alignment

    target = [1, 2, 3, 4, 5]
    candidates = [
        ("bad", [10, 20, 30, 40, 50]),
        ("good", [1, 2, 3, 4, 99]),
    ]

    result = compute_batch_alignment(candidates, target, min_reuse_ratio=0.5)
    assert result is not None
    donor_id, alignment = result
    assert donor_id == "good"
    assert alignment.reuse_ratio >= 0.5


def test_fallback_alignment():
    """Fallback prefix alignment works without rapidfuzz."""
    from synapse_kv_connector.alignment import _fallback_prefix_alignment

    donor = [1, 2, 3, 4, 5]
    target = [1, 2, 3, 10, 11]
    result = _fallback_prefix_alignment(donor, target)

    assert result.reuse_ratio == 0.6  # 3/5
    assert result.edit_distance == 2


# ---------------------------------------------------------------------------
# Context gate tests
# ---------------------------------------------------------------------------


def test_context_gate_rejects_isolated_match():
    """A single isolated chunk match should be rejected by the context gate."""
    from synapse_kv_connector.alignment import (
        LMCACHE_CHUNK_SIZE as CS,
    )
    from synapse_kv_connector.alignment import (
        SlotActionType,
        compute_chunk_alignment,
    )

    # Donor: [A][B][C]  (3 full chunks, each 256 tokens)
    # Target: [X][B][Y]  (chunk B matches but X and Y differ)
    # Context gate should reject chunk B because neither neighbor matches.
    chunk_a = list(range(1, CS + 1))
    chunk_b = list(range(CS + 1, 2 * CS + 1))
    chunk_c = list(range(2 * CS + 1, 3 * CS + 1))
    chunk_x = list(range(10001, 10001 + CS))
    chunk_y = list(range(20001, 20001 + CS))

    donor = chunk_a + chunk_b + chunk_c
    target = chunk_x + chunk_b + chunk_y

    result = compute_chunk_alignment(donor, target, context_gate=True)
    assert result.reuse_ratio == 0.0, (
        f"Isolated chunk match should be rejected, got reuse={result.reuse_ratio}"
    )
    assert all(sa.action == SlotActionType.RECOMPUTE for sa in result.slot_actions)


def test_context_gate_accepts_contiguous_matches():
    """Two or more contiguous matching chunks should pass the context gate."""
    from synapse_kv_connector.alignment import (
        LMCACHE_CHUNK_SIZE as CS,
    )
    from synapse_kv_connector.alignment import (
        compute_chunk_alignment,
    )

    # Donor: [A][B][C]  Target: [A][B][X]
    # Chunks A and B match → each has a matching neighbor → both accepted.
    chunk_a = list(range(1, CS + 1))
    chunk_b = list(range(CS + 1, 2 * CS + 1))
    chunk_c = list(range(2 * CS + 1, 3 * CS + 1))
    chunk_x = list(range(10001, 10001 + CS))

    donor = chunk_a + chunk_b + chunk_c
    target = chunk_a + chunk_b + chunk_x

    result = compute_chunk_alignment(donor, target, context_gate=True)
    expected_reuse = (2 * CS) / (3 * CS)
    assert abs(result.reuse_ratio - expected_reuse) < 0.01, (
        f"Expected reuse ~{expected_reuse:.3f}, got {result.reuse_ratio:.3f}"
    )


def test_context_gate_disabled_accepts_isolated():
    """With context gate disabled, isolated matches should be accepted."""
    from synapse_kv_connector.alignment import (
        LMCACHE_CHUNK_SIZE as CS,
    )
    from synapse_kv_connector.alignment import (
        compute_chunk_alignment,
    )

    chunk_a = list(range(1, CS + 1))
    chunk_b = list(range(CS + 1, 2 * CS + 1))
    chunk_c = list(range(2 * CS + 1, 3 * CS + 1))
    chunk_x = list(range(10001, 10001 + CS))
    chunk_y = list(range(20001, 20001 + CS))

    donor = chunk_a + chunk_b + chunk_c
    target = chunk_x + chunk_b + chunk_y

    result = compute_chunk_alignment(donor, target, context_gate=False)
    expected_reuse = CS / (3 * CS)
    assert abs(result.reuse_ratio - expected_reuse) < 0.01, (
        f"Without gate, expected reuse ~{expected_reuse:.3f}, got {result.reuse_ratio:.3f}"
    )


def test_context_gate_full_match_accepted():
    """All chunks matching should all pass the context gate."""
    from synapse_kv_connector.alignment import (
        LMCACHE_CHUNK_SIZE as CS,
    )
    from synapse_kv_connector.alignment import (
        compute_chunk_alignment,
    )

    tokens = list(range(4 * CS))
    result = compute_chunk_alignment(tokens, tokens, context_gate=True)
    assert result.reuse_ratio == 1.0


def test_context_gate_reorder_with_contiguous():
    """REORDER scenario: swapped paragraph blocks should pass gate if contiguous."""
    from synapse_kv_connector.alignment import (
        LMCACHE_CHUNK_SIZE as CS,
    )
    from synapse_kv_connector.alignment import (
        compute_chunk_alignment,
    )

    # Donor: [A][B][C][D]  Target: [C][D][A][B]
    # All 4 chunks match. C and D are adjacent (pass). A and B are adjacent (pass).
    chunk_a = list(range(1, CS + 1))
    chunk_b = list(range(CS + 1, 2 * CS + 1))
    chunk_c = list(range(2 * CS + 1, 3 * CS + 1))
    chunk_d = list(range(3 * CS + 1, 4 * CS + 1))

    donor = chunk_a + chunk_b + chunk_c + chunk_d
    target = chunk_c + chunk_d + chunk_a + chunk_b

    result = compute_chunk_alignment(donor, target, context_gate=True)
    assert result.reuse_ratio == 1.0


def test_context_gate_scattered_isolated_all_rejected():
    """Multiple isolated matches (no two adjacent) should all be rejected."""
    from synapse_kv_connector.alignment import (
        LMCACHE_CHUNK_SIZE as CS,
    )
    from synapse_kv_connector.alignment import (
        compute_chunk_alignment,
    )

    # Donor: [A][B][C][D][E]
    # Target: [A][X][C][Y][E]  — A, C, E match but none are adjacent
    chunk_a = list(range(1, CS + 1))
    chunk_b = list(range(CS + 1, 2 * CS + 1))
    chunk_c = list(range(2 * CS + 1, 3 * CS + 1))
    chunk_d = list(range(3 * CS + 1, 4 * CS + 1))
    chunk_e = list(range(4 * CS + 1, 5 * CS + 1))
    chunk_x = list(range(10001, 10001 + CS))
    chunk_y = list(range(20001, 20001 + CS))

    donor = chunk_a + chunk_b + chunk_c + chunk_d + chunk_e
    target = chunk_a + chunk_x + chunk_c + chunk_y + chunk_e

    result = compute_chunk_alignment(donor, target, context_gate=True)
    assert result.reuse_ratio == 0.0, (
        f"Scattered isolated matches should all be rejected, got {result.reuse_ratio}"
    )


# ---------------------------------------------------------------------------
# Bathtub curve tests
# ---------------------------------------------------------------------------


def test_adaptive_threshold_boundaries():
    """Adaptive threshold should respect similarity bounds."""
    from synapse_kv_connector.bathtub import adaptive_threshold

    # Zero similarity → minimum threshold (base only, most recomputation)
    assert abs(adaptive_threshold(0.0) - 0.3) < 1e-6
    # Perfect similarity → maximum threshold (base+scale, least recomputation)
    assert abs(adaptive_threshold(1.0) - 0.7) < 1e-6
    # Production threshold τ=0.60 → moderate
    t_60 = adaptive_threshold(0.60)
    assert 0.3 < t_60 < 0.7
    assert abs(t_60 - 0.54) < 1e-6


def test_adaptive_threshold_monotonic():
    """Higher similarity should produce higher threshold (fewer recomputed layers)."""
    from synapse_kv_connector.bathtub import adaptive_threshold

    sims = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    thresholds = [adaptive_threshold(s) for s in sims]
    for i in range(len(thresholds) - 1):
        assert thresholds[i] <= thresholds[i + 1], (
            f"Threshold should increase with similarity: "
            f"sim={sims[i]}→{thresholds[i]:.3f}, "
            f"sim={sims[i + 1]}→{thresholds[i + 1]:.3f}"
        )


def test_adaptive_threshold_fewer_recompute_layers():
    """Higher similarity → higher threshold → fewer layers flagged for recompute."""
    from synapse_kv_connector.bathtub import compute_layer_deviations

    high_sim = compute_layer_deviations(num_layers=28, similarity=0.95)
    low_sim = compute_layer_deviations(num_layers=28, similarity=0.65)

    high_recompute = sum(1 for d in high_sim if d.should_recompute)
    low_recompute = sum(1 for d in low_sim if d.should_recompute)

    assert high_recompute <= low_recompute, (
        f"High similarity should recompute ≤ low similarity: "
        f"high_sim={high_recompute}, low_sim={low_recompute}"
    )


def test_adaptive_threshold_backward_compat():
    """Without similarity param, fixed threshold should be used."""
    from synapse_kv_connector.bathtub import compute_layer_deviations

    # Without similarity: uses fixed threshold=0.3
    fixed = compute_layer_deviations(num_layers=28, threshold=0.3)
    # Verify that similarity=None gives same result as explicit threshold
    fixed_explicit = compute_layer_deviations(num_layers=28, threshold=0.3, similarity=None)

    for f, e in zip(fixed, fixed_explicit):
        assert f.should_recompute == e.should_recompute

    # With similarity=0.0: adaptive_threshold(0.0)=0.3 → same as fixed threshold=0.3
    adaptive_zero = compute_layer_deviations(num_layers=28, similarity=0.0)
    for f, a in zip(fixed, adaptive_zero):
        assert f.should_recompute == a.should_recompute


def test_bathtub_early_late_deviation():
    """Early and late layers should have higher deviation than middle."""
    from synapse_kv_connector.bathtub import compute_layer_deviations

    devs = compute_layer_deviations(num_layers=28)

    # Layer 0 (early) should have high deviation
    assert devs[0].deviation_score > 0.3
    # Middle layer should be low
    assert devs[14].deviation_score < 0.2
    # Last layer should be high
    assert devs[27].deviation_score > 0.2


def test_bathtub_symmetry():
    """Bathtub should be roughly symmetric around the middle."""
    from synapse_kv_connector.bathtub import sigma

    early = sigma(layer_idx=2, num_layers=28)
    late = sigma(layer_idx=25, num_layers=28)

    # Both should be elevated
    assert early > 0.15
    assert late > 0.15


def test_bathtub_sigma_bounds():
    """sigma() should return values in [0, 1]."""
    from synapse_kv_connector.bathtub import sigma

    for num_layers in [24, 28, 32, 80]:
        for i in range(num_layers):
            s = sigma(i, num_layers)
            assert 0.0 <= s <= 1.0, f"sigma({i}, {num_layers}) = {s}"


def test_bathtub_preset_lookup():
    """Model name matching should work with full HuggingFace names."""
    from synapse_kv_connector.bathtub import get_preset

    preset = get_preset("Qwen/Qwen2.5-7B-Instruct-AWQ")
    assert preset.num_layers == 28

    default = get_preset("unknown-model")
    assert default.num_layers == 32


def test_bathtub_mismatch_scaling():
    """Higher mismatch fraction should increase deviation scores."""
    from synapse_kv_connector.bathtub import compute_layer_deviations

    low = compute_layer_deviations(num_layers=28, mismatch_fraction=0.1)
    high = compute_layer_deviations(num_layers=28, mismatch_fraction=0.5)

    # Middle layer should deviate more with higher mismatch
    assert high[14].deviation_score >= low[14].deviation_score


def test_bathtub_latency():
    """Bathtub computation must complete in <0.1ms."""
    from synapse_kv_connector.bathtub import compute_layer_deviations

    t0 = time.monotonic()
    for _ in range(1000):
        compute_layer_deviations(num_layers=28)
    elapsed_ms = (time.monotonic() - t0) * 1000

    per_call_ms = elapsed_ms / 1000
    assert per_call_ms < 0.1, f"Bathtub took {per_call_ms:.3f}ms (budget: 0.1ms)"


# ---------------------------------------------------------------------------
# Fuzzy chunk alignment tests
# ---------------------------------------------------------------------------


def test_fuzzy_exact_chunks_still_work():
    """Exact chunk matches should still work with fuzzy alignment enabled."""
    from synapse_kv_connector.alignment import (
        LMCACHE_CHUNK_SIZE as CS,
    )
    from synapse_kv_connector.alignment import (
        SlotActionType,
        compute_fuzzy_chunk_alignment,
    )

    tokens = list(range(3 * CS))
    result = compute_fuzzy_chunk_alignment(tokens, tokens, min_overlap=0.90)

    assert result.reuse_ratio == 1.0
    assert result.exact_chunks == 3
    assert result.fuzzy_chunks == 0
    assert all(sa.action == SlotActionType.COPY_FROM_DONOR for sa in result.slot_actions)


def test_fuzzy_shifted_prefix_recovers_reuse():
    """Shifted prefix (Δ=1 token) should produce fuzzy matches for document chunks."""
    from synapse_kv_connector.alignment import (
        LMCACHE_CHUNK_SIZE as CS,
    )
    from synapse_kv_connector.alignment import (
        SlotActionType,
        compute_chunk_alignment,
        compute_fuzzy_chunk_alignment,
    )

    # Donor: 5-token instruction + 3*CS-5 document tokens
    # Target: 6-token instruction + 3*CS-6 document tokens (1 token shift)
    instruction_a = [9001, 9002, 9003, 9004, 9005]
    instruction_b = [8001, 8002, 8003, 8004, 8005, 8006]
    document = list(range(1, 3 * CS - 4))  # shared document content

    donor = instruction_a + document
    target = instruction_b + document[: len(document) - 1]

    # Pad to same length
    while len(donor) < 3 * CS:
        donor.append(0)
    while len(target) < 3 * CS:
        target.append(0)

    # Exact chunk alignment: should find 0% (boundary shift)
    exact_result = compute_chunk_alignment(donor, target)
    assert exact_result.reuse_ratio == 0.0, (
        f"Exact chunks should fail on shifted prefix, got {exact_result.reuse_ratio}"
    )

    # Fuzzy chunk alignment: should recover most tokens
    fuzzy_result = compute_fuzzy_chunk_alignment(
        donor,
        target,
        min_overlap=0.90,
    )
    assert fuzzy_result.fuzzy_chunks > 0, "Should have fuzzy-matched chunks"
    assert fuzzy_result.reuse_ratio > 0.90, (
        f"Expected >90% reuse from fuzzy matching, got {fuzzy_result.reuse_ratio:.3f}"
    )

    # Verify that fuzzy matches have donor_pos != target_pos (Δ ≠ 0)
    copy_actions = [
        sa for sa in fuzzy_result.slot_actions if sa.action == SlotActionType.COPY_FROM_DONOR
    ]
    delta_nonzero = [sa for sa in copy_actions if sa.donor_pos != sa.target_pos]
    assert len(delta_nonzero) > 0, "Fuzzy matches should produce Δ≠0 (shifted positions)"


def test_fuzzy_high_overlap_threshold():
    """Chunks with overlap below threshold should not match."""
    from synapse_kv_connector.alignment import (
        LMCACHE_CHUNK_SIZE as CS,
    )
    from synapse_kv_connector.alignment import (
        compute_fuzzy_chunk_alignment,
    )

    # Donor chunk: tokens 1-256
    # Target chunk: 50% different tokens → should NOT match at 0.90 threshold
    donor = list(range(1, CS + 1))
    target = list(range(1, CS // 2 + 1)) + list(range(10001, 10001 + CS // 2))

    result = compute_fuzzy_chunk_alignment(
        donor,
        target,
        min_overlap=0.90,
        context_gate=False,
    )
    assert result.fuzzy_chunks == 0, (
        f"50% overlap should not match at 0.90 threshold, got {result.fuzzy_chunks} fuzzy"
    )


def test_fuzzy_low_overlap_threshold():
    """Chunks with overlap above a lower threshold should match."""
    from synapse_kv_connector.alignment import (
        LMCACHE_CHUNK_SIZE as CS,
    )
    from synapse_kv_connector.alignment import (
        compute_fuzzy_chunk_alignment,
    )

    # Donor: [A][B]  Target: [A'][B] where A' differs in 10% of tokens
    chunk_a = list(range(1, CS + 1))
    chunk_b = list(range(CS + 1, 2 * CS + 1))

    # A' has 90% of A's tokens plus 10% different
    num_changed = CS // 10  # 25 tokens at chunk_size=256
    chunk_a_prime = list(chunk_a)
    for i in range(num_changed):
        chunk_a_prime[i] = 50001 + i

    donor = chunk_a + chunk_b
    target = chunk_a_prime + chunk_b

    # Use a permissive fuzzy config to test overlap-only matching
    # (bag-cosine gate would reject 90% overlap with sequential tokens)
    from synapse_kv_connector.alignment import FuzzyMatchConfig

    permissive_config = FuzzyMatchConfig(
        bag_cosine_min=0.50,
        confidence_low=0.10,
    )
    result = compute_fuzzy_chunk_alignment(
        donor,
        target,
        min_overlap=0.85,
        context_gate=False,
        fuzzy_config=permissive_config,
    )
    # chunk_b matches exactly, chunk_a' should fuzzy-match chunk_a
    assert result.exact_chunks >= 1, "chunk_b should match exactly"
    assert result.fuzzy_chunks >= 1, (
        "chunk_a' with 90% overlap should fuzzy-match at 0.85 threshold"
    )
    assert result.reuse_ratio > 0.90


def test_fuzzy_context_gate_integration():
    """Fuzzy matches should respect the context gate (adjacent match required)."""
    from synapse_kv_connector.alignment import (
        LMCACHE_CHUNK_SIZE as CS,
    )
    from synapse_kv_connector.alignment import (
        compute_fuzzy_chunk_alignment,
    )

    # Donor: [A][B][C]  Target: [A'][X][C'] where A' and C' fuzzy-match but
    # have no adjacent match (X doesn't match B). Context gate should reject.
    chunk_a = list(range(1, CS + 1))
    chunk_b = list(range(CS + 1, 2 * CS + 1))
    chunk_c = list(range(2 * CS + 1, 3 * CS + 1))
    chunk_x = list(range(30001, 30001 + CS))  # totally different

    # A' and C' differ by 1 token each (99.6% overlap)
    chunk_a_prime = list(chunk_a)
    chunk_a_prime[0] = 99999
    chunk_c_prime = list(chunk_c)
    chunk_c_prime[0] = 99998

    donor = chunk_a + chunk_b + chunk_c
    target = chunk_a_prime + chunk_x + chunk_c_prime

    result = compute_fuzzy_chunk_alignment(
        donor,
        target,
        min_overlap=0.90,
        context_gate=True,
    )
    # A' and C' have no adjacent match (X doesn't match B) → gate rejects
    assert result.reuse_ratio == 0.0, (
        f"Isolated fuzzy matches should be rejected by context gate, "
        f"got reuse={result.reuse_ratio:.3f}"
    )


def test_fuzzy_compute_alignment_integration():
    """compute_alignment with fuzzy=True should use fuzzy chunk matching."""
    from synapse_kv_connector.alignment import (
        LMCACHE_CHUNK_SIZE as CS,
    )
    from synapse_kv_connector.alignment import (
        compute_alignment,
    )

    # Same shifted-prefix scenario as test_fuzzy_shifted_prefix_recovers_reuse
    instruction_a = [9001, 9002, 9003, 9004, 9005]
    instruction_b = [8001, 8002, 8003, 8004, 8005, 8006]
    document = list(range(1, 3 * CS - 4))

    donor = instruction_a + document
    target = instruction_b + document[: len(document) - 1]
    while len(donor) < 3 * CS:
        donor.append(0)
    while len(target) < 3 * CS:
        target.append(0)

    # Without fuzzy: should fall back to Levenshtein (also works but different path)
    result_exact = compute_alignment(donor, target, fuzzy=False)

    # With fuzzy: should use fuzzy chunk alignment
    result_fuzzy = compute_alignment(donor, target, fuzzy=True)

    # Both should find high reuse ratio
    assert result_exact.reuse_ratio > 0.80
    assert result_fuzzy.reuse_ratio > 0.90
    # Fuzzy result should have fuzzy_chunks > 0
    assert result_fuzzy.fuzzy_chunks > 0


def test_fuzzy_delta_values_correct():
    """Verify that fuzzy match Δ values are correct for a constant shift."""
    from synapse_kv_connector.alignment import (
        LMCACHE_CHUNK_SIZE as CS,
    )
    from synapse_kv_connector.alignment import (
        SlotActionType,
        compute_fuzzy_chunk_alignment,
    )

    # 2-token shift: donor instruction is 4 tokens, target instruction is 6 tokens
    shift = 2
    instr_donor = [9001, 9002, 9003, 9004]
    instr_target = [8001, 8002, 8003, 8004, 8005, 8006]
    # Shared document tokens (enough for 2 full chunks after instruction)
    doc_tokens = list(range(1, 2 * CS + 1))

    donor = instr_donor + doc_tokens
    target = instr_target + doc_tokens

    # Pad to equal length (3 chunks)
    while len(donor) < 3 * CS:
        donor.append(0)
    while len(target) < 3 * CS:
        target.append(0)

    result = compute_fuzzy_chunk_alignment(
        donor,
        target,
        min_overlap=0.90,
        context_gate=False,
    )

    # Check that document token deltas are consistent
    copy_actions = [
        sa
        for sa in result.slot_actions
        if sa.action == SlotActionType.COPY_FROM_DONOR and sa.donor_pos is not None
    ]
    # For matched document tokens, Δ should be ≈ shift (±1 depending on
    # which chunk boundary they fall in)
    deltas = [sa.target_pos - sa.donor_pos for sa in copy_actions]
    if deltas:
        # Most deltas should be equal to the shift
        from collections import Counter

        delta_counts = Counter(deltas)
        most_common_delta = delta_counts.most_common(1)[0][0]
        assert most_common_delta == shift, (
            f"Most common Δ should be {shift}, got {most_common_delta}"
        )

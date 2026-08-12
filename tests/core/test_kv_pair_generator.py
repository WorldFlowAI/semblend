"""Paired-KV training-data extraction from near-duplicate pairs.

Given donor/target token sequences, the generator aligns them (CDC +
diagonal, token-verified) and extracts 1:1 divergence records: aligned
positions where the two sequences substitute exactly one run of equal
length (the imputation training positions), each with its flanking matched
context. Insertions/deletions (unequal gap lengths) are excluded in v0.
"""

from semblend_core.kv_pair_generator import extract_divergence_records


def _seqs(sub_at, donor_tok, target_tok, n=400, shift=7):
    """Identical content with a positional shift and one substitution."""
    base = [1000 + (i * 37) % 800 for i in range(n)]
    donor = [77] * shift + list(base)
    target = list(base)
    donor[shift + sub_at] = donor_tok
    target[sub_at] = target_tok
    return donor, target


class TestExtractDivergenceRecords:
    def test_finds_single_substitution_with_offset(self):
        donor, target = _seqs(sub_at=200, donor_tok=5, target_tok=9)
        recs = extract_divergence_records(
            donor_tokens=donor, target_tokens=target, donor_id="d1"
        )
        subs = [r for r in recs if r["kind"] == "substitution"]
        assert len(subs) == 1
        r = subs[0]
        assert r["target_pos"] == 200
        assert r["donor_pos"] == 207  # shifted by 7
        assert r["donor_tok"] == 5 and r["target_tok"] == 9
        assert r["run_before"] >= 32 and r["run_after"] >= 32

    def test_equal_length_multi_token_substitution(self):
        donor, target = _seqs(sub_at=150, donor_tok=5, target_tok=9)
        # widen to a 3-token substitution
        for k, (dt, tt) in enumerate([(5, 9), (6, 10), (7, 11)]):
            donor[157 + k] = dt
            target[150 + k] = tt
        recs = extract_divergence_records(
            donor_tokens=donor, target_tokens=target, donor_id="d1"
        )
        subs = [r for r in recs if r["kind"] == "substitution"]
        assert len(subs) == 3
        assert [r["target_pos"] for r in subs] == [150, 151, 152]

    def test_unequal_gap_excluded(self):
        donor, target = _seqs(sub_at=200, donor_tok=5, target_tok=9)
        target.insert(350, 4242)  # insertion -> unequal gap downstream
        recs = extract_divergence_records(
            donor_tokens=donor, target_tokens=target, donor_id="d1"
        )
        subs = [r for r in recs if r["kind"] == "substitution"]
        gaps = [r for r in recs if r["kind"] == "excluded_gap"]
        # the interior substitution survives; the insertion region is
        # excluded (unequal donor/target gap), never a substitution
        assert len(subs) == 1 and subs[0]["target_pos"] == 200
        assert gaps and all(g["donor_gap"] != g["target_gap"] for g in gaps)

    def test_identical_sequences_no_records(self):
        base = [1000 + i for i in range(300)]
        recs = extract_divergence_records(
            donor_tokens=list(base), target_tokens=list(base), donor_id="d1"
        )
        assert [r for r in recs if r["kind"] == "substitution"] == []

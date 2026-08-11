"""Tokenization-invariant alignment via canonical text space.

Reformatting (whitespace collapse, line rewrapping) changes token
boundaries nearly everywhere, so token-level alignment captures ~0% of a
class where the CONTENT is near-identical. Canonical-space alignment
matches whitespace-collapsed text, projects matched spans onto each side's
token sequence, and keeps only resynced subruns where both tokenizations
agree byte-for-byte — reusable through the existing token-identical join
machinery, with bubbles recomputed in-prefill.
"""

from semblend_core.canonical_alignment import (
    canonicalize,
    resynced_token_runs,
)


class TestCanonicalize:
    def test_collapses_whitespace_with_offset_map(self):
        text = "alpha  beta\n\tgamma"
        canon, cmap = canonicalize(text)
        assert canon == "alpha beta gamma"
        # offset map points every canonical char at its original index
        assert text[cmap[0]] == "a"
        assert text[cmap[5]] in " \n\t"  # the collapsed separator
        assert text[cmap[11]] == "g"
        assert len(cmap) == len(canon)

    def test_identity_on_already_canonical_text(self):
        text = "one two three"
        canon, cmap = canonicalize(text)
        assert canon == text
        assert cmap == list(range(len(text)))


class _WordTokenizer:
    """BPE-like test tokenizer: leading whitespace attaches to the word
    (so a separator change perturbs exactly one token, like real BPE)."""

    def encode_with_offsets(self, text):
        import re

        ids, offsets = [], []
        for m in re.finditer(r"\s*\S+", text):
            tok = m.group(0)
            ids.append(hash(tok) & 0x7FFFFFFF)
            offsets.append((m.start(), m.end()))
        return ids, offsets


class TestResyncedTokenRuns:
    def test_reformatted_content_recovers_word_tokens(self):
        donor = "HEADER\n" + " ".join(f"word{i}" for i in range(50))
        target = "T[9] session\n" + "\n".join(
            " ".join(f"word{i}" for i in range(k, min(k + 10, 50)))
            for k in range(0, 50, 10)
        )
        tok = _WordTokenizer()
        runs = resynced_token_runs(
            donor_text=donor,
            target_text=target,
            donor_ids_offsets=tok.encode_with_offsets(donor),
            target_ids_offsets=tok.encode_with_offsets(target),
            min_run_tokens=4,
        )
        assert runs, "expected at least one resynced run"
        covered = sum(r["length"] for r in runs)
        # All 50 words are identical content; only whitespace differs, so
        # word tokens resync (whitespace tokens differ and become bubbles).
        assert covered >= 40
        # Every run must be token-verified: donor ids == target ids
        for r in runs:
            assert r["donor_token_start"] >= 0 and r["target_token_start"] >= 0
            assert r["verified"] is True

    def test_disjoint_content_yields_no_runs(self):
        tok = _WordTokenizer()
        runs = resynced_token_runs(
            donor_text=" ".join(f"alpha{i}" for i in range(40)),
            target_text=" ".join(f"beta{i}" for i in range(40)),
            donor_ids_offsets=tok.encode_with_offsets(
                " ".join(f"alpha{i}" for i in range(40))
            ),
            target_ids_offsets=tok.encode_with_offsets(
                " ".join(f"beta{i}" for i in range(40))
            ),
            min_run_tokens=4,
        )
        assert runs == []

    def test_edited_word_splits_runs(self):
        base = [f"word{i}" for i in range(60)]
        donor = " ".join(base)
        edited = list(base)
        edited[30] = "EDITED"
        # rewrap every 10 words (realistic reformat) + one content edit
        target = "\n".join(
            " ".join(edited[k : k + 10]) for k in range(0, 60, 10)
        )
        tok = _WordTokenizer()
        runs = resynced_token_runs(
            donor_text=donor,
            target_text=target,
            donor_ids_offsets=tok.encode_with_offsets(donor),
            target_ids_offsets=tok.encode_with_offsets(target),
            min_run_tokens=4,
        )
        assert len(runs) >= 2  # split around the edit
        covered_words = sum(r["length"] for r in runs)
        assert covered_words >= 40

"""Fact gate: divergence detection and the adversarial regressions that
shaped the rules (magnitude binding, one/a interchange)."""

from semblend_core.fact_gate import spans_fact_consistent


def test_identical_facts_paraphrased_accept():
    assert spans_fact_consistent(
        "The tally recorded exactly 47 approvals in Geneva.",
        "Precisely 47 approvals appeared in the Geneva tally.",
    )


def test_divergent_number_rejects():
    assert not spans_fact_consistent(
        "The tally recorded exactly 47 approvals.",
        "The tally recorded exactly 52 approvals.",
    )


def test_unit_swap_rejects():
    # regression: bare-digit comparison accepted a thousandfold divergence
    assert not spans_fact_consistent(
        "Costs fell to 12 million this quarter.",
        "Costs fell to 12 thousand this quarter.",
    )


def test_one_article_interchange_accepts():
    # regression: "a"/"one" interchange is ordinary paraphrase
    assert spans_fact_consistent(
        "The board held a meeting on the matter.",
        "The board held one meeting on the matter.",
    )


def test_entity_swap_rejects():
    assert not spans_fact_consistent(
        "The committee met in Geneva with 47 delegates.",
        "The committee met in Zurich with 47 delegates.",
    )


def test_word_number_divergence_rejects():
    assert not spans_fact_consistent(
        "They filed seven complaints.",
        "They filed nine complaints.",
    )


def test_sentence_initial_common_noun_is_not_an_entity():
    # rewording moves words across sentence boundaries; capitalization
    # from sentence position must not create phantom entities
    assert spans_fact_consistent(
        "Cluster region A reported nominal utilization in cycle 3.",
        "Region A of the cluster reported nominal utilization in cycle 3.",
    )


def test_mid_sentence_entity_still_detected():
    assert not spans_fact_consistent(
        "The delegates met in Geneva for cycle 3.",
        "The delegates met in Zurich for cycle 3.",
    )

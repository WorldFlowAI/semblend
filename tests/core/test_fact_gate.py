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


def test_lowercase_identifier_swap_rejects():
    # regression: two reports that share every number and differ only in
    # a lowercase service identifier passed the gate and served the wrong
    # document's KV (cart-sync answered as billing-eu)
    assert not spans_fact_consistent(
        "during window 0 the billing-eu service handled 1200 requests with p99 at 340 ms.",
        "in window 0, service cart-sync processed 1200 requests, p99 340 ms.",
    )


def test_same_identifier_reworded_accepts():
    assert spans_fact_consistent(
        "during window 0 the billing-eu service handled 1200 requests with p99 at 340 ms.",
        "in window 0, service billing-eu processed 1200 requests, p99 340 ms.",
    )


def test_common_hyphenated_word_dropped_still_accepts():
    # ordinary rewording may un-hyphenate a compound word; one such change
    # among several stable identifiers stays within the tolerance
    assert spans_fact_consistent(
        "the long-term plan for us-east-1 and eu-west-2 keeps 3 replicas.",
        "the long term plan for us-east-1 and eu-west-2 keeps 3 replicas.",
    )


def test_code_identifier_swap_rejects():
    assert not spans_fact_consistent(
        "incident INC-4471 was opened at 09:15 and closed after 40 minutes.",
        "incident INC-4472 was opened at 09:15 and closed after 40 minutes.",
    )

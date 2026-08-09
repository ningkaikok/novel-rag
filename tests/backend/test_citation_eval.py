from citation_eval import evaluate_citations


def test_valid_and_invalid_citation_numbers():
    metrics = evaluate_citations(
        "顾长风中了蚀骨散[2]，后来痊愈[1][9]。",
        [{"text": "沈砚之治好了他。"}, {"text": "奇毒名叫蚀骨散。"}],
        ["蚀骨散"],
    )

    assert metrics["citation_numbers"] == [2, 1, 9]
    assert metrics["valid_citation_numbers"] == [2, 1]
    assert metrics["invalid_citation_numbers"] == [9]
    assert metrics["valid_number_ratio"] == 2 / 3
    assert metrics["expected_evidence_coverage"] == 1.0


def test_missing_citations_do_not_fake_support():
    metrics = evaluate_citations(
        "顾长风中了蚀骨散。",
        [{"text": "奇毒名叫蚀骨散。"}],
        ["蚀骨散"],
    )

    assert metrics["citation_numbers"] == []
    assert metrics["valid_number_ratio"] == 0.0
    assert metrics["expected_evidence_coverage"] == 0.0

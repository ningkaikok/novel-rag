"""引用的离线可验证指标。

纯规则只能验证两件事：编号是否指向真实来源，以及被引用来源是否覆盖评测集声明的
期望证据关键词。它不能完全替代人工语义判断，所以返回值明确保留
``manual_support_review_required``，避免把“编号合法”误写成“事实一定被支持”。
"""
import re
from collections.abc import Mapping, Sequence

_CITATION_RE = re.compile(r"\[(\d+)]")


def evaluate_citations(
    answer: str,
    sources: Sequence[Mapping],
    expected_keywords: Sequence[str] = (),
) -> dict:
    """计算一条回答的引用指标，来源编号按界面约定从 1 开始。"""
    mentioned = [int(value) for value in _CITATION_RE.findall(answer)]
    unique_mentioned = list(dict.fromkeys(mentioned))
    valid = [number for number in unique_mentioned if 1 <= number <= len(sources)]
    invalid = [number for number in unique_mentioned if number not in valid]
    cited_sources = [sources[number - 1] for number in valid]

    covered_keywords = [
        keyword
        for keyword in expected_keywords
        if any(keyword in str(source.get("text") or source.get("excerpt") or "") for source in cited_sources)
    ]
    keyword_coverage = (
        len(covered_keywords) / len(expected_keywords) if expected_keywords else None
    )

    return {
        "citation_numbers": unique_mentioned,
        "valid_citation_numbers": valid,
        "invalid_citation_numbers": invalid,
        "valid_number_ratio": (
            len(valid) / len(unique_mentioned) if unique_mentioned else 0.0
        ),
        "cited_source_count": len(valid),
        "expected_keywords": list(expected_keywords),
        "covered_expected_keywords": covered_keywords,
        "expected_evidence_coverage": keyword_coverage,
        "manual_support_review_required": True,
    }

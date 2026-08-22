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
    """计算一条回答的引用指标，来源编号按界面约定从 1 开始。

    各返回字段含义：
    - ``valid_number_ratio``：合法编号占全部引用编号的比例（1.0 = 没有越界引用）
    - ``cited_source_count``：实际被引用的不同来源个数
    - ``expected_evidence_coverage``：期望证据关键词出现在**被引用片段**中的比例；
      没有声明期望关键词时为 None，与"覆盖率为 0"区分开
    - ``manual_support_review_required``：恒为 True，见模块 docstring
    """
    # 回答里出现的全部引用编号。dict.fromkeys 去重但保留首次出现顺序——
    # 同一 [1] 引用三次只算一个来源，重复出现不应虚增 cited_source_count。
    mentioned = [int(value) for value in _CITATION_RE.findall(answer)]
    unique_mentioned = list(dict.fromkeys(mentioned))
    # 编号是否落在 [1, len(sources)] 内。界面只把合法编号变成可点击按钮，
    # 越界编号（模型幻觉出的 [99]）在这里单独统计而不是悄悄丢弃。
    valid = [number for number in unique_mentioned if 1 <= number <= len(sources)]
    invalid = [number for number in unique_mentioned if number not in valid]
    cited_sources = [sources[number - 1] for number in valid]

    # 关键词只检查**被引用的**片段，而不是全部来源：如果证据在来源里、
    # 却没出现在被引用的片段中，说明模型找对了材料但没引用对，同样算未覆盖。
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

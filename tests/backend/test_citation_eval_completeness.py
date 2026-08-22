"""M3.5 引用完整性指标：分句规则与豁免词表。

全部纯函数测试，不碰数据库或模型。
"""

import json

from citation_eval import (
    DEFAULT_EXEMPT_PHRASES,
    evaluate_citations,
    evaluate_completeness,
    split_statements,
)


def test_split_statements_keeps_delimiters_and_drops_empty():
    answer = "顾长风中了蚀骨散[1]。后来痊愈了吗？痊愈了！"
    assert split_statements(answer) == [
        "顾长风中了蚀骨散[1]。",
        "后来痊愈了吗？",
        "痊愈了！",
    ]
    # 空串和纯空白不应产生幽灵句子
    assert split_statements("   ") == []


def test_meta_and_greeting_sentences_are_exempted():
    """元描述/拒答/寒暄句没有引用不算完整性缺陷——这是词表存在的意义。

    注意启发式的粒度：豁免按整句匹配（比如以"根据检索结果"开头的句子即使
    后半句带事实和引用也会被豁免），这是可接受的近似——评测统计宁可少算
    一句事实，也不把客套话当成缺引用的事实陈述。
    """
    answer = "顾长风中了蚀骨散[1]。根据提供的片段无法确定具体年份。希望对你有帮助！"
    metrics = evaluate_completeness(answer)
    # 三句里只有第一句是事实陈述（且带引用）
    assert metrics["statement_count"] == 3
    assert metrics["exempted_count"] == 2
    assert metrics["factual_statement_count"] == 1
    assert metrics["uncited_statement_count"] == 0
    assert metrics["uncited_ratio"] == 0.0


def test_uncited_ratio_counts_only_factual_statements():
    answer = "顾长风中了蚀骨散[1]。沈砚之是游方郎中。希望这个回答对你有帮助！"
    metrics = evaluate_completeness(answer)
    assert metrics["factual_statement_count"] == 2
    assert metrics["uncited_statement_count"] == 1
    assert metrics["uncited_ratio"] == 0.5
    assert metrics["uncited_statements"] == ["沈砚之是游方郎中。"]


def test_custom_exempt_phrases_replace_default_list():
    answer = "顾长风中了蚀骨散。根据检索结果整理如下。"
    # 自定义词表整体替换默认词表：默认词表里的"根据检索结果"不再生效，
    # 第二句只因命中自定义的"整理如下"才被豁免
    metrics = evaluate_completeness(answer, exempt_phrases=("整理如下",))
    assert metrics["factual_statement_count"] == 1
    assert DEFAULT_EXEMPT_PHRASES  # 默认词表本身没被改动


def test_no_factual_statement_gives_none_not_zero():
    """纯拒答回答没有任何可评估陈述——ratio 必须是 None 而不是 0/100%。"""
    answer = "根据提供的片段无法确定。"
    metrics = evaluate_completeness(answer)
    assert metrics["factual_statement_count"] == 0
    assert metrics["uncited_ratio"] is None


def test_evaluate_citations_keeps_existing_fields_and_adds_group():
    """向后兼容：旧字段一个不少、值不变；新增 completeness 分组。"""
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
    assert metrics["manual_support_review_required"] is True
    # 新增的三分类分组（整条回答只有一句话，且带引用）
    assert metrics["completeness"]["factual_statement_count"] == 1
    assert metrics["completeness"]["cited_statement_count"] == 1
    assert metrics["faithfulness_method"] == "shadow_only_judge_support"
    # 快照序列化安全（落库走 JSON）：新分组必须可序列化
    json.dumps(metrics)

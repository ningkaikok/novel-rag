"""低置信度信号纯函数的单元测试。

compute_confidence 不碰数据库、不加载模型，直接喂 (SourceChunk, 分数) 对
即可验证全部信号逻辑——这也是它被刻意设计成纯函数的原因之一：离线校准
脚本和在线挂钩点跑的是同一份代码，测试结论对两边同时有效。
"""

import math

from chunk_model import SourceChunk
from confidence import (
    SCORE_GAP_LOW,
    TERM_COVERAGE_MIN,
    compute_confidence,
    normalized_score,
)


def _chunk(novel: str, text: str) -> SourceChunk:
    return SourceChunk(novel=novel, chunk_id=1, text=text, distance=0.0)


def test_normalized_score_is_sigmoid():
    # sigmoid(0)=0.5；大 logit 饱和到接近 1，且不会因 exp 溢出抛异常
    assert normalized_score(0.0) == 0.5
    assert normalized_score(20.0) > 0.999999
    assert normalized_score(-20.0) < 0.000001
    assert abs(normalized_score(2.3026) - 0.909) < 0.001


def test_score_gap_uses_only_reranker_scores_in_order():
    # 第 1 名 sigmoid ≈ 0.953、第 2 名 ≈ 0.881，差距应只由这两个重排分数决定，
    # 与候选的 distance 字段（向量/BM25 原始分的遗留出口）完全无关
    a = SourceChunk(novel="书", chunk_id=1, text="甲", distance=0.99)
    b = SourceChunk(novel="书", chunk_id=2, text="乙", distance=-50.0)
    signals = compute_confidence("问题", [(a, 3.0), (b, 2.0)])
    expected = 1 / (1 + math.exp(-3.0)) - 1 / (1 + math.exp(-2.0))
    assert signals["score_gap"] == round(expected, 6)


def test_single_candidate_means_no_gap_signal():
    # 只有一条候选时无从比较，记 1.0 且不触发——「没法比」≠「有歧义」。
    # 文本刻意覆盖全部问题词，排除 term_coverage 触发路径的干扰。
    signals = compute_confidence("韩立的师父是谁", [(_chunk("书", "韩立的师父是墨大夫"), 5.0)])
    assert signals["score_gap"] == 1.0
    assert not signals["is_low_confidence"]


def test_empty_candidates_is_reported_as_not_low():
    # 空结果属于「完全无证据」，走现有拒答逻辑；扩展救不了空索引，
    # 绝不能触发补救。这里验证它不会被误判成低置信。
    signals = compute_confidence("韩立的师父是谁", [])
    assert signals["is_low_confidence"] is False


def test_term_coverage_counts_query_terms_found_in_candidates():
    question = "韩立的师父是谁"  # 分词后含「韩立」「师父」「大夫」等词
    covered = _chunk("雾隐山庄", "韩立的师父墨大夫教他长春功")
    missing = _chunk("雾隐山庄", "庄主正在练剑")
    full = compute_confidence(question, [(covered, 5.0)])["term_coverage"]
    zero = compute_confidence(question, [(missing, 5.0)])["term_coverage"]
    assert full == 1.0
    assert zero == 0.0


def test_term_coverage_full_coverage_when_question_has_no_content_words():
    # 全是停用词的问题没有可检查的关键词，视为全覆盖，不该误触发补救
    signals = compute_confidence("讲讲呗", [(_chunk("书", "随便什么"), 5.0)])
    assert signals["term_coverage"] == 1.0


def test_cross_book_dispersion_counts_distinct_novels():
    scored = [
        (_chunk("凡人修仙传", "甲"), 5.0),
        (_chunk("诡秘之主", "乙"), 4.9),
        (_chunk("凡人修仙传", "丙"), 4.8),
    ]
    assert compute_confidence("问题", scored)["cross_book_dispersion"] == 2


def test_low_confidence_triggers_on_low_term_coverage_alone():
    # 覆盖率 0 → 单独触发；阈值常量本身也应小于 1，否则规则形同虚设
    assert TERM_COVERAGE_MIN < 1.0
    signals = compute_confidence(
        "韩立的师父是谁",
        [
            (_chunk("雾隐山庄", "完全无关的文本内容"), 8.0),
            (_chunk("雾隐山庄", "另一段无关"), 7.9),
        ],
    )
    assert "term_coverage" in signals["low_signals"]
    assert signals["is_low_confidence"] is True


def test_close_scores_within_one_book_flag_entity_ambiguity():
    assert SCORE_GAP_LOW > 0
    # 分差极小 + 候选全在同一本书 → 实体歧义信号，两个 low_signals 都要出现
    same_book = [
        (_chunk("凡人修仙传", "韩立相关原文片段"), 3.0),
        (_chunk("凡人修仙传", "韩立相关的另一段"), 2.99),
    ]
    signals = compute_confidence("韩立的师父是谁", same_book)
    assert signals["score_gap"] < SCORE_GAP_LOW
    assert "score_gap" in signals["low_signals"]
    assert "cross_book_dispersion" in signals["low_signals"]
    assert signals["is_low_confidence"] is True


def test_close_scores_across_books_do_not_trigger():
    # 多书分散且分差小更像「问题问得泛」，换措辞帮助不大——保守起见不触发。
    # 注意覆盖了关键词，避免 term_coverage 单独触发干扰本用例的意图。
    text = "韩立的师父是墨大夫"
    cross_books = [
        (_chunk("凡人修仙传", text), 3.0),
        (_chunk("诡秘之主", text), 2.99),
    ]
    signals = compute_confidence("韩立的师父是谁", cross_books)
    assert signals["is_low_confidence"] is False


def test_high_confidence_case_fires_nothing():
    scored = [
        (_chunk("凡人修仙传", "韩立的师父是墨大夫，他暗中传授长春功"), 9.0),
        (_chunk("凡人修仙传", "墨大夫另有图谋的段落"), -2.0),
    ]
    signals = compute_confidence("韩立的师父是谁", scored)
    assert signals["score_gap"] >= SCORE_GAP_LOW
    assert signals["term_coverage"] == 1.0
    assert signals["low_signals"] == []
    assert signals["is_low_confidence"] is False

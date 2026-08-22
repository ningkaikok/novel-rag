"""M4 关系抽取评测脚本：混淆计算与标注集完整性。

不跑任何模型：混淆四格和 P/R/F1 用 mock 数据验证；
标注集本身做结构校验（关系类型合法、人名确实出现在原创语料里）。
"""

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "eval_graph_under_test",
        ROOT / "scripts" / "eval_graph.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eval_graph = _load_script()


@dataclass
class FakeChunk:
    novel: str
    text: str
    chunk_id: int | None = None


# ---------------------------------------------------------------- 混淆计算


def _mock_cases_and_predictions():
    cases = [
        {"id": "c1", "label": "positive"},
        {"id": "c2", "label": "positive"},  # c2 漏判
        {"id": "c3", "label": "negative"},  # c3 误报
        {"id": "c4", "label": "negative"},
        {"id": "c5", "label": "negative"},
    ]
    predictions = {"c1": True, "c2": False, "c3": True, "c4": False, "c5": False}
    return cases, predictions


def test_confusion_counts_all_four_cells():
    cases, predictions = _mock_cases_and_predictions()
    counts = eval_graph.confusion_counts(cases, predictions)
    assert counts["tp"] == 1 and counts["fn"] == 1
    assert counts["fp"] == 1 and counts["tn"] == 2


def test_prf_matches_hand_computation():
    # TP=1, FP=1, FN=1 → P=0.5, R=0.5, F1=0.5
    precision, recall, f1 = eval_graph.prf(1, 1, 1)
    assert (precision, recall, f1) == (0.5, 0.5, 0.5)


def test_prf_zero_denominators_return_none_not_crash():
    """一个正类都没预测出来时精确率无定义，显示为 — 而不是除零崩溃。"""
    precision, recall, f1 = eval_graph.prf(0, 0, 3)
    assert precision is None and f1 is None
    assert recall == 0.0


def test_normalize_pair_is_order_insensitive():
    assert eval_graph.normalize_pair("小顺", "沈砚秋") == ("小顺", "沈砚秋")
    assert eval_graph.normalize_pair("沈砚秋", "小顺") == ("小顺", "沈砚秋")


# ---------------------------------------------------------------- 共现基线预测


def test_cooccurrence_baseline_predicts_only_keyword_chunk_pairs():
    """两个人物只在含关系触发词的片段里同框才预测 positive——线上建图的原始逻辑。"""
    corpus = {
        "书": [
            FakeChunk("书", "甲是乙的师父", chunk_id=0),  # 师徒关键词 + 同框
            FakeChunk("书", "甲和乙一起吃饭", chunk_id=1),  # 同框但无关键词
            FakeChunk("书", "甲独自赶路", chunk_id=2),
        ]
    }
    cases = [
        {"id": "s1", "novel": "书", "a": "甲", "b": "乙", "relation": "师徒"},
        {"id": "s2", "novel": "书", "a": "乙", "b": "丙", "relation": "师徒"},
    ]
    predictions = eval_graph.predict_cooccurrence(cases, corpus, {"书": ["甲", "乙", "丙"]})

    assert predictions == {"s1": True, "s2": False}


def test_llm_prediction_requires_explicit_and_confidence_gate():
    """co_occurrence 边和高置信度以下的 explicit 边都不算正类（门槛语义与线上一致）。"""
    corpus = {
        "书": [FakeChunk("书", "甲是乙的师父。乙遇到丙", chunk_id=0)],
    }

    def fake_generate(prompt):
        return (
            '[{"a": "甲", "b": "乙", "direction": "甲→乙", '
            '"kind": "explicit", "confidence": 0.9, "chunks": [1]}, '
            '{"a": "乙", "b": "丙", "direction": "none", '
            '"kind": "co_occurrence", "confidence": 0.9, "chunks": [1]}]'
        )

    cases = [
        {"id": "l1", "novel": "书", "a": "甲", "b": "乙", "relation": "师徒"},
        {"id": "l2", "novel": "书", "a": "乙", "b": "丙", "relation": "师徒"},
    ]
    errors: list[str] = []
    predictions = eval_graph.predict_llm(
        cases,
        corpus,
        {"书": ["甲", "乙", "丙"]},
        fake_generate,
        min_confidence=0.7,
        errors=errors,
    )

    assert predictions == {"l1": True, "l2": False}
    assert errors == []


# ---------------------------------------------------------------- 标注集完整性


def test_eval_set_structure_and_corpus_membership():
    """标注集必须自洽：标签合法、关系类型受支持、人名逐字出现在语料原文里。

    这条测试保证评测脚本随时可跑：谁改了语料或词表导致标注失配，CI 立刻变红。
    """
    from graph import RELATION_KEYWORDS

    payload = json.loads((ROOT / "tests" / "graph_eval_set.json").read_text(encoding="utf-8"))
    cases, cast = payload["cases"], payload["cast"]

    assert len(cases) >= 15, "标注集太小没有统计意义"
    corpus_texts = {
        novel: (ROOT / "tests" / "ci_corpus" / f"{novel}.txt").read_text(encoding="utf-8")
        for novel in cast
    }
    seen_ids = set()
    positives = 0
    for case in cases:
        assert case["id"] not in seen_ids, f"id 重复：{case['id']}"
        seen_ids.add(case["id"])
        assert case["label"] in {"positive", "negative"}
        positives += case["label"] == "positive"
        assert case["relation"] in RELATION_KEYWORDS, f"{case['id']} 关系类型不支持"
        assert case["a"] != case["b"]
        for name in (case["a"], case["b"]):
            assert name in cast.get(case["novel"], []), (
                f"{case['id']} 人名 {name} 不在 cast 名单里"
            )
            assert name in corpus_texts[case["novel"]], (
                f"{case['id']} 人名 {name} 没出现在《{case['novel']}》原文里"
            )
    assert 0 < positives < len(cases), "全是正类或全是负类的标注集没有区分度"

"""M3.5 影子评测脚本：混淆矩阵计算与标注集完整性。

不跑任何模型：矩阵计算用 mock 的 (predicted, human) 对；
标注集本身只做结构校验（标签合法、证据确实出自仓库原创语料）。
"""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "eval_faithfulness_shadow_under_test",
        ROOT / "scripts" / "eval_faithfulness_shadow.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shadow = _load_script()


def test_confusion_matrix_counts_mock_rows():
    rows = [
        ("supported", "supported"),
        ("supported", "partial"),  # 预测过宽
        ("unsupported", "unsupported"),
        ("uncertain", "supported"),  # Judge 拿不准但人工有把握
        ("uncertain", "partial"),
        ("supported", "supported"),
    ]
    matrix = shadow.build_confusion_matrix(rows)
    assert matrix["supported"]["supported"] == 2
    assert matrix["supported"]["partial"] == 1
    assert matrix["unsupported"]["unsupported"] == 1
    assert matrix["uncertain"]["supported"] == 1
    assert matrix["uncertain"]["partial"] == 1
    # 没出现过的格子是 0 而不是 KeyError
    assert matrix["unsupported"].get("supported", 0) == 0


def test_matrix_tolerates_unseen_prediction_label():
    """未来 Judge 输出新增标签时，矩阵不应崩溃。"""
    matrix = shadow.build_confusion_matrix([("meh", "supported")])
    assert matrix["meh"]["supported"] == 1


def test_format_matrix_renders_all_columns():
    text = shadow._format_matrix(shadow.build_confusion_matrix([]))
    for label in shadow.HUMAN_LABELS:
        assert label in text


# ---------------------------------------------------------------- 多 Judge 对比辅助


def test_binary_pr_one_vs_rest_counts_partial_as_negative():
    """supported 的二元 P/R：预测端不会产出 partial，人工 partial 记假阳性。"""
    rows = [
        ("supported", "supported"),  # TP
        ("supported", "partial"),  # FP（刻意不算召回）
        ("uncertain", "supported"),  # FN
        ("unsupported", "unsupported"),
    ]
    precision, recall = shadow.binary_pr(rows, "supported")
    assert (precision, recall) == (0.5, 0.5)
    uns_p, uns_r = shadow.binary_pr(rows, "unsupported")
    assert (uns_p, uns_r) == (1.0, 1.0)


def test_binary_pr_zero_denominator_returns_none():
    precision, recall = shadow.binary_pr([("uncertain", "partial")], "supported")
    assert precision is None and recall is None


class _FlakyJudge:
    """前 n 次调用抛异常，之后正常输出 JSON 的 mock 生成函数。"""

    def __init__(self, fail_times: int):
        self.fail_times = fail_times
        self.calls = 0

    def __call__(self, prompt):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("模拟网络抖动")
        return iter(['{"label": "supported", "reason": "ok"}'])


def _retry_case():

    return {"statement": "测试陈述。", "evidence": ["测试证据，包含测试陈述的内容。"]}


def test_judge_with_retry_recovers_after_transient_failure(monkeypatch):
    monkeypatch.setattr(shadow, "_MIN_CALL_INTERVAL_S", 0)
    judge = _FlakyJudge(fail_times=1)
    label, reason, failed = shadow.judge_with_retry(_retry_case(), judge)
    assert (label, failed) == ("supported", False)
    assert judge.calls == 2


def test_judge_with_retry_degrades_to_uncertain_after_exhaustion(monkeypatch):
    monkeypatch.setattr(shadow, "_MIN_CALL_INTERVAL_S", 0)
    judge = _FlakyJudge(fail_times=99)  # 永远失败
    label, _, failed = shadow.judge_with_retry(_retry_case(), judge)
    assert (label, failed) == ("uncertain", True)
    assert judge.calls == 1 + shadow._MAX_RETRIES


def test_judge_with_retry_does_not_retry_model_uncertainty(monkeypatch):
    """模型正常回答的 uncertain 是真实判断，不该被重试放大调用量。"""
    monkeypatch.setattr(shadow, "_MIN_CALL_INTERVAL_S", 0)

    def stable_uncertain(prompt):
        return iter('{"label": "uncertain", "reason": "拿不准"}')

    label, _, failed = shadow.judge_with_retry(_retry_case(), stable_uncertain)
    assert (label, failed) == ("uncertain", False)


# ---------------------------------------------------------------- 标注集本身

CASES = shadow.load_cases()
_ORIGINAL_CORPORA = [
    ROOT / "tests" / "ci_corpus" / "沙海航灯.txt",
    ROOT / "tests" / "ci_corpus" / "青梧镇异闻.txt",
    ROOT / "data" / "novels" / "雾隐山庄.txt",
]
_ORIGINAL_TEXT = "".join(p.read_text(encoding="utf-8") for p in _ORIGINAL_CORPORA)


def test_shadow_set_has_enough_cases_with_valid_labels():
    assert len(CASES) >= 15, "标注集太少，混淆矩阵没有统计意义"
    for case in CASES:
        assert case["human_label"] in shadow.HUMAN_LABELS, case["id"]
        assert case["statement"] and case["evidence"], case["id"]
    # 刻意包含的易混淆案例必须真的在集子里
    categories = {case["category"] for case in CASES}
    assert any("隐喻" in c for c in categories)
    assert any("换名" in c for c in categories)
    assert any("顺序颠倒" in c for c in categories)
    assert any("数字" in c or "挪用" in c for c in categories)


def test_all_evidence_comes_from_original_corpus_only():
    """隐私/版权红线：每条证据必须是仓库原创语料的子串，不得引入版权原文。

    比较前把中英文引号和空白归一化：JSON 里写弯引号更易读，语料文件里
    用的是直引号，两者应视为同一字符。
    """

    def normalize(text: str) -> str:
        for left, right in (
            ("\u201c", '"'),
            ("\u201d", '"'),
            ("\u2018", "'"),
            ("\u2019", "'"),
        ):
            text = text.replace(left, right)
        return "".join(text.split())

    corpus = normalize(_ORIGINAL_TEXT)
    for case in CASES:
        for evidence in case["evidence"]:
            assert normalize(evidence) in corpus, f"{case['id']} 的证据不是原创语料的逐字子串"


def test_shadow_set_is_loadable_json_with_expected_shape():
    payload = json.loads(
        (ROOT / "tests" / "citation_shadow_set.json").read_text(encoding="utf-8")
    )
    assert isinstance(payload["cases"], list)
    assert set(CASES[0]) >= {"id", "statement", "evidence", "human_label"}

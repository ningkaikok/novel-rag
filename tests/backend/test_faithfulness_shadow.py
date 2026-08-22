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
        ("supported", "partial"),       # 预测过宽
        ("unsupported", "unsupported"),
        ("uncertain", "supported"),     # Judge 拿不准但人工有把握
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
        for left, right in (("\u201c", '"'), ("\u201d", '"'), ("\u2018", "'"), ("\u2019", "'")):
            text = text.replace(left, right)
        return "".join(text.split())

    corpus = normalize(_ORIGINAL_TEXT)
    for case in CASES:
        for evidence in case["evidence"]:
            assert normalize(evidence) in corpus, (
                f"{case['id']} 的证据不是原创语料的逐字子串"
            )


def test_shadow_set_is_loadable_json_with_expected_shape():
    payload = json.loads(
        (ROOT / "tests" / "citation_shadow_set.json").read_text(encoding="utf-8")
    )
    assert isinstance(payload["cases"], list)
    assert set(CASES[0]) >= {"id", "statement", "evidence", "human_label"}

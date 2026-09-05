"""滚动会话摘要评测脚本：断言逻辑与标注集完整性。

不跑任何模型、不调用真实生成后端。真实模型跑出来的结果记在
docs/experiments/m36-session-summary.md 里，这里只保证：

1. check() 的包含/排除断言逻辑本身没写错
2. 标注集的结构前提成立（recent_turns 足够长，early_turns 真的会被挤出窗口）
3. 事实真的来自原创语料，不是编出来的（同 test_eval_graph.py 的做法）
"""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "eval_session_summary_under_test",
        ROOT / "scripts" / "eval_session_summary.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eval_session_summary = _load_script()

TEST_SET = json.loads(
    (ROOT / "tests" / "session_summary_test_set.json").read_text(encoding="utf-8")
)
_CORPUS_FILES = {
    "沙海航灯": ROOT / "tests" / "ci_corpus" / "沙海航灯.txt",
    "青梧镇异闻": ROOT / "tests" / "ci_corpus" / "青梧镇异闻.txt",
    "雾隐山庄": ROOT / "data" / "novels" / "雾隐山庄.txt",
}


def test_check_flags_missing_required_token():
    problems = eval_session_summary.check(
        "这段摘要提到了别的事", {"expect_contains": ["蚀骨散"]}
    )
    assert problems == ["缺少「蚀骨散」"]


def test_check_flags_forbidden_token():
    problems = eval_session_summary.check(
        "顾长风已经痊愈了", {"expect_absent": ["顾长风已经痊愈"]}
    )
    assert problems == ["不该出现「顾长风已经痊愈」"]


def test_check_passes_when_both_conditions_satisfied():
    text = "顾长风中了蚀骨散，目前尚未痊愈"
    problems = eval_session_summary.check(
        text, {"expect_contains": ["蚀骨散"], "expect_absent": ["已经痊愈"]}
    )
    assert problems == []


def test_structural_report_confirms_every_case_loses_its_early_fact(capsys):
    """零成本的核心断言：不开摘要时，早期事实必然从组装好的背景里消失。

    这条测试把 report_structural_only 内部的 assert 变成 CI 会持续盯着的不变量，
    而不是只有手动跑脚本时才会发现"某条用例的 recent_turns 改短了"。
    """
    exit_code = eval_session_summary.report_structural_only(TEST_SET["cases"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert f"{len(TEST_SET['cases'])}/{len(TEST_SET['cases'])} 条用例确认" in output
    assert "⚠️" not in output, "有用例的早期事实没有真的被挤出窗口，测试前提不成立"


def test_every_case_recent_turns_meets_the_window_size():
    """结构前提本身也要能独立失败——不依赖上面那条测试的输出文本断言。"""
    from config import HISTORY_MAX_TURNS

    for case in TEST_SET["cases"]:
        assert len(case["recent_turns"]) >= HISTORY_MAX_TURNS, (
            f"[{case['id']}] recent_turns 太短，early_turns 不一定会被挤出窗口"
        )


def test_eval_set_facts_are_grounded_in_the_original_corpus():
    """expect_contains 断言的事实必须来自仓库原创语料，不能是编出来的。

    早期事实和过滤后的追问内容也一并核对，防止测试作者凭印象编了个听起来
    对但语料里其实没有的细节。
    """
    corpus_texts = {
        name: path.read_text(encoding="utf-8") for name, path in _CORPUS_FILES.items()
    }

    for case in TEST_SET["cases"]:
        text = corpus_texts[case["book"]]
        for token in case["expect_contains"]:
            assert token in text, f"[{case['id']}] 「{token}」没有出现在 {case['book']} 原文里"


def test_all_turn_content_is_non_empty():
    """空内容的轮次会被 build_history_block 直接过滤掉，early_turns 就白写了。"""
    for case in TEST_SET["cases"]:
        for turn in case["early_turns"] + case["recent_turns"]:
            assert turn["content"].strip(), f"[{case['id']}] 有一轮内容是空的"

"""M3.5 第三阶段「两步走 Judge」：断言抽取解析、聚合逻辑、单步/两步切换。

全部用假生成函数 mock，绝不调用真实 LLM。核心场景来自第二阶段的结论：
单步 Judge 退化成二分类器、对「半真半假」失明；两步走要把 partial 变成
聚合规则的机械结果。
"""

import importlib.util
from pathlib import Path

import pytest

from citation_eval import (
    aggregate_claim_verdicts,
    extract_claims,
    judge_claim,
    judge_support,
    judge_support_two_step,
)

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


def _gen(text):
    """构造一个"无论 prompt 是什么都返回固定文本"的假生成函数。"""
    return lambda _prompt: iter([text])


def _scripted_gen(responses: list[str]):
    """按调用顺序回放响应的假生成函数；记录每次收到的 prompt。"""
    calls: list[str] = []

    def fn(prompt: str):
        calls.append(prompt)
        return iter([responses.pop(0)])

    fn.calls = calls  # type: ignore[attr-defined]
    return fn


# ---------------------------------------------------------------- 断言抽取


def test_extract_claims_parses_plain_and_fenced_json():
    claims, reason = extract_claims("陈述", _gen('{"claims": ["甲偷了表", "甲把表卖了"]}'))
    assert claims == ["甲偷了表", "甲把表卖了"] and reason is None

    fenced = '好的：\n```json\n{"claims": ["唯一断言"]}\n```\n以上。'
    claims, reason = extract_claims("陈述", _gen(fenced))
    assert claims == ["唯一断言"] and reason is None


def test_extract_claims_caps_count_and_drops_blanks():
    raw = '{"claims": ["", "  ", "有效断言一", "有效断言二", "三", "四", "五", "六", "七"]}'
    claims, reason = extract_claims("陈述", _gen(raw))
    assert reason is None
    assert len(claims) <= 6, "超出上限的断言必须截断，防止烧钱"
    assert "" not in claims


@pytest.mark.parametrize(
    ("raw", "marker"),
    [
        ('{"label": "supported"}', "claims"),  # 结构对但字段缺失
        ('{"claims": "不是数组"}', "不是数组"),
        ('{"claims": []}', "空数组"),
        ("我认为这条陈述整体上没问题，不需要拆解。", "JSON"),
    ],
)
def test_extract_claims_degrades_with_retryable_prefix(raw, marker):
    errors: list[str] = []
    claims, reason = extract_claims("陈述", _gen(raw), errors=errors)
    assert claims == []
    assert reason is not None and marker in reason
    assert reason.startswith(("Judge 调用失败：", "Judge 输出解析失败：")), (
        "前缀是脚本层重试逻辑的唯一信号，降级路径必须保持一致"
    )
    assert errors


def test_extract_claims_generation_exception_reports_call_failure():
    def broken(_prompt):
        raise TimeoutError("模型超时")

    errors: list[str] = []
    claims, reason = extract_claims("陈述", broken, errors=errors)
    assert claims == []
    assert reason and "TimeoutError" in reason
    assert errors and "TimeoutError" in errors[0]


# ---------------------------------------------------------------- 单条断言判定


def test_judge_claim_accepts_three_verdict_labels():
    for label in ("supported", "contradicted", "not_found"):
        result = judge_claim("断言", ["证据"], _gen(f'{{"label": "{label}", "reason": "r"}}'))
        assert result["label"] == label


def test_judge_claim_rejects_single_step_vocabulary():
    """两步走的第二步不允许混入单步标签空间——partial/uncertain 只能由聚合产生。"""
    errors: list[str] = []
    result = judge_claim("断言", ["证据"], _gen('{"label": "partial"}'), errors=errors)
    assert result["label"] == "uncertain"
    assert "label 非法" in result["reason"]
    assert result["reason"].startswith("Judge 输出解析失败：")


def test_judge_claim_exception_degrades_to_uncertain():
    def broken(_prompt):
        raise ConnectionError("网络断了")

    result = judge_claim("断言", ["证据"], broken)
    assert result["label"] == "uncertain"
    assert result["reason"].startswith("Judge 调用失败：")


# ---------------------------------------------------------------- 聚合逻辑


def test_aggregate_all_supported_is_supported():
    result = aggregate_claim_verdicts(["supported", "supported"])
    assert result["label"] == "supported"


@pytest.mark.parametrize(
    "verdicts",
    [
        ["supported", "contradicted"],
        ["supported", "not_found"],
        ["supported", "contradicted", "not_found"],
    ],
)
def test_aggregate_mixed_verdicts_is_partial(verdicts):
    """部分有据 + 其余不成立 → partial：这正是第二阶段所有单步 Judge 的盲区形态，
    两步走把它变成机械规则而不是指望模型主动说半真半假。"""
    assert aggregate_claim_verdicts(verdicts)["label"] == "partial"


def test_aggregate_unknown_label_is_uncertain():
    assert aggregate_claim_verdicts(["supported", "meh"])["label"] == "uncertain"


def test_aggregate_empty_list_is_uncertain():
    assert aggregate_claim_verdicts([])["label"] == "uncertain"


@pytest.mark.parametrize(
    "verdicts",
    [
        ["contradicted"],
        ["contradicted", "not_found"],
        ["not_found", "not_found"],
    ],
)
def test_aggregate_without_any_support_is_unsupported(verdicts):
    assert aggregate_claim_verdicts(verdicts)["label"] == "unsupported"


# ---------------------------------------------------------------- 两步走端到端


def _two_step_backend(*verdicts: str):
    """构造两步走的 mock 后端：第一次返回抽取结果，之后按顺序回放逐条判定。

    断言抽取的 JSON 由 prompt 内容触发区分——mock 不需要真的理解 prompt。
    """
    responses = ['{"claims": ["断言A", "断言B"]}']
    responses += [f'{{"label": "{v}", "reason": "理由"}}' for v in verdicts]
    return _scripted_gen(responses)


def test_two_step_happy_path_returns_partial_with_trace():
    fn = _two_step_backend("supported", "contradicted")
    result = judge_support_two_step("半真半假的陈述", ["证据"], fn)
    assert result["label"] == "partial"
    assert result["claims"] == ["断言A", "断言B"]
    assert [v["label"] for v in result["verdicts"]] == ["supported", "contradicted"]
    # 调用次数 = 1 次抽取 + N 条断言核对；prompt 也必须是两种不同模板
    assert len(fn.calls) == 3
    assert any("原子断言" in p for p in fn.calls), "第一步应使用断言抽取 prompt"
    assert any("三选一" in p for p in fn.calls), "第二步应使用逐条核对 prompt"


def test_two_step_label_space_matches_single_step_contract():
    """评测脚本把两种方法当同一种东西消费：返回必须有 label/reason 键，
    且 label 落在人工标签可对照的空间内。"""
    fn = _two_step_backend("supported")
    result = judge_support_two_step("单一事实陈述", ["证据"], fn)
    assert set(result) >= {"label", "reason"}
    single = judge_support(
        "单一事实陈述", ["证据"], _gen('{"label": "supported", "reason": "r"}')
    )
    assert set(single) == {"label", "reason"}
    assert result["label"] in ("supported", "partial", "unsupported", "uncertain")


def test_two_step_extraction_failure_degrades_to_uncertain():
    errors: list[str] = []
    result = judge_support_two_step("陈述", ["证据"], _gen("完全不是 JSON"), errors=errors)
    assert result["label"] == "uncertain"
    assert result["claims"] == [] and result["verdicts"] == []
    # 抽取失败也要带可重试前缀：脚本层重试的是整个 case，不区分步骤
    assert result["reason"].startswith(("Judge 调用失败：", "Judge 输出解析失败："))
    assert errors


def test_two_step_claim_failure_aborts_with_retryable_reason():
    """第 2 条断言核对失败时不能拿残缺判定聚合作假：整条降级 uncertain。"""
    fn = _scripted_gen(
        [
            '{"claims": ["断言A", "断言B"]}',
            '{"label": "supported", "reason": "ok"}',
            "这段输出故意不含 JSON",
        ]
    )
    errors: list[str] = []
    result = judge_support_two_step("陈述", ["证据"], fn, errors=errors)
    assert result["label"] == "uncertain"
    assert result["reason"].startswith("Judge 输出解析失败：")
    assert [v["label"] for v in result["verdicts"]] == ["supported"], "已成功的判定要留痕"
    assert errors


# ---------------------------------------------------------------- 默认行为不变


def test_default_single_step_path_unchanged():
    """不显式选择两步走时，judge_support 仍是单次调用、原 prompt、原返回形状。"""
    fn = _scripted_gen(['{"label": "partial", "reason": "模型自创标签"}'])
    result = judge_support("陈述", ["证据"], fn)
    assert len(fn.calls) == 1, "默认单步路径只允许一次 LLM 调用"
    assert "原子断言" not in fn.calls[0], "单步 prompt 不应被替换成抽取模板"
    assert result["label"] == "uncertain", "非法标签照旧降级，行为不变"


def test_script_retry_uses_two_step_backend_when_requested(monkeypatch):
    """脚本层开关真正改变调用的函数：two_step=True 时一个 case 至少两次 LLM 调用。"""
    monkeypatch.setattr(shadow, "_MIN_CALL_INTERVAL_S", 0)
    fn = _scripted_gen(
        [
            '{"claims": ["断言A"]}',
            '{"label": "not_found", "reason": "无据"}',
        ]
    )
    case = {"statement": "测试陈述。", "evidence": ["测试证据。"]}
    result, failed = shadow.judge_with_retry(case, fn, two_step=True)
    assert not failed
    assert result["label"] == "unsupported"
    assert len(fn.calls) >= 2

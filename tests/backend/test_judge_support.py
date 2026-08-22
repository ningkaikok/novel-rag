"""M3.5 忠实度影子接口：judge_support 的解析与异常降级，rule_support 基线。

全部用假生成函数 mock，绝不调用真实 LLM。
"""

from citation_eval import judge_support, rule_support


def _gen(text):
    """构造一个"无论 prompt 是什么都返回固定文本"的假生成函数。"""
    return lambda _prompt: iter([text])


def test_parses_plain_json_output():
    fn = _gen('{"label": "supported", "reason": "证据明确说明埋深四点二米"}')
    result = judge_support("承压水埋深四点二米。", ["……埋深只有四点二米……"], fn)
    assert result == {
        "label": "supported",
        "reason": "证据明确说明埋深四点二米",
    }


def test_parses_code_fenced_and_noisy_output():
    """模型常在 JSON 外包 markdown 围栏和客套话，解析必须剥掉。"""
    raw = (
        "好的，以下是判断结果：\n"
        "```json\n"
        '{"label": "unsupported", "reason": "时间顺序颠倒"}\n'
        "```\n"
        "希望这能帮到你。"
    )
    result = judge_support("陈述", ["证据"], _gen(raw))
    assert result["label"] == "unsupported"
    assert result["reason"] == "时间顺序颠倒"


def test_invalid_label_degrades_to_uncertain():
    """label 不在三值之内（比如模型自创 partial）必须降级为 uncertain，
    而不是把非法标签透传给下游——影子评测宁要不确定也不要错标签。"""
    raw = '{"label": "partial", "reason": "一半对"}'
    errors: list[str] = []
    result = judge_support("陈述", ["证据"], _gen(raw), errors=errors)
    assert result["label"] == "uncertain"
    assert "label 非法" in result["reason"]
    assert errors and "partial" in errors[0]


def test_generation_exception_degrades_to_uncertain():
    def broken(_prompt):
        raise TimeoutError("模型超时")

    errors: list[str] = []
    result = judge_support("陈述", ["证据"], broken, errors=errors)
    assert result["label"] == "uncertain"
    assert "TimeoutError" in result["reason"]
    # 失败原因不能被静默吞掉
    assert any("TimeoutError" in e for e in errors)


def test_non_json_output_degrades_to_uncertain():
    raw = "我认为这段陈述是支持的，因为证据里写得很清楚。"
    result = judge_support("陈述", ["证据"], _gen(raw))
    assert result["label"] == "uncertain"
    assert "JSON" in result["reason"]
    assert "我认为" in result["reason"], "降级原因里应保留原始输出片段供排查"


# ---------------------------------------------------------------- 规则基线


def test_rule_support_high_overlap_is_supported():
    result = rule_support(
        "七号井位的承压水埋深只有四点二米。",
        [
            "上级要求他们把七号井位的坐标重新测定",
            "数据证实：这里的承压水埋深只有四点二米，水质达到饮用标准，完全可以打井。",
        ],
    )
    assert result["label"] == "supported"


def test_rule_support_zero_overlap_is_unsupported():
    result = rule_support(
        "雾隐山庄四周种满了桃树。",
        ["队伍里还有三个人：司机兼机械师巴特尔……"],
    )
    assert result["label"] == "unsupported"


def test_rule_support_middle_overlap_is_uncertain():
    statement = "陆知微是队长，他还会修卡车。"
    evidence = ["地质勘探队的营地扎在沙丘背风的一侧，队长陆知微正在帐篷里核对第二天的路线。"]
    result = rule_support(statement, evidence)
    assert result["label"] == "uncertain"


def test_rule_support_short_or_empty_input_is_uncertain():
    assert rule_support("好", ["证据"])["label"] == "uncertain"
    assert rule_support("正常长度的陈述。", [])["label"] == "uncertain"

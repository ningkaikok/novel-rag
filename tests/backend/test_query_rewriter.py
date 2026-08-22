"""多轮查询改写的单元测试。不调真实 LLM，用假生成函数验证调度逻辑。"""

from query_rewriter import format_history, needs_rewrite, rewrite_query

HISTORY = [
    {"role": "user", "content": "韩立的师父是谁？"},
    {"role": "assistant", "content": "韩立的师父是李化元。"},
]


def fake_gen(answer):
    def gen(prompt):
        yield answer

    return gen


def test_first_turn_never_rewrites():
    """第一轮没有上文，没有指代可解析——不该白花一次 LLM 调用。"""
    assert not needs_rewrite("他是谁", has_history=False)


def test_pronoun_question_needs_rewrite():
    assert needs_rewrite("他后来怎么样了？", has_history=True)


def test_short_question_needs_rewrite():
    """ "后来呢"这类极短问题基本都依赖上文。"""
    assert needs_rewrite("后来呢", has_history=True)


def test_self_contained_question_skipped():
    """本来就说清楚了主语的问题不该触发改写。

    踩过的坑：长度阈值一开始设成 12，而"雾隐山庄的庄主得了什么病"正好 12 个字，
    被误判成需要改写，白花一次 LLM 调用。阈值降到 8 后修复。
    """
    assert not needs_rewrite("雾隐山庄的庄主得了什么病", has_history=True)


def test_rewrite_resolves_pronoun():
    result = rewrite_query("他后来怎么样了？", HISTORY, fake_gen("李化元后来怎么样了？"))
    assert result == "李化元后来怎么样了？"


def test_cosmetic_rewrite_treated_as_no_change():
    """改写结果只差标点/虚词时，当作没改。

    踩过的坑：模型把"雾隐山庄的庄主得了什么病"改成"雾隐山庄庄主得了什么病"
    （只去掉一个「的」），字符串不同于是界面显示"补全指代后按…检索"——
    可用户根本没写任何指代，纯属误导。
    """
    original = "雾隐山庄的庄主得了什么病呢"
    result = rewrite_query(original, HISTORY, fake_gen("雾隐山庄庄主得了什么病"))
    assert result == original, "实质相同的改写应该退回原问题"


def test_failure_falls_back_to_original():
    """改写失败必须退回原问题，不能让提问不可用。"""

    def boom(prompt):
        raise RuntimeError("模型限流")

    errors: list[str] = []
    result = rewrite_query("他后来怎么样了？", HISTORY, boom, errors)

    assert result == "他后来怎么样了？"
    assert len(errors) == 1 and "限流" in errors[0]  # 但原因保留下来了


def test_rambling_rewrite_rejected():
    """模型啰嗦了一大段解释时退回原问题——坏的改写比不改写更糟，
    它会把检索引到完全错误的方向。
    """
    rambling = "好的，我来帮你改写这个问题。" + "根据上下文分析，" * 20
    result = rewrite_query("他后来怎么样了？", HISTORY, fake_gen(rambling))
    assert result == "他后来怎么样了？"


def test_history_truncates_long_answers():
    """助手的回答可能上千字，全塞进改写提示词里既慢又容易喧宾夺主。"""
    turns = [{"role": "assistant", "content": "很长的回答。" * 200}]
    formatted = format_history(turns, max_chars=50)
    assert len(formatted) < 100
    assert "…" in formatted


def test_history_keeps_only_recent_turns():
    turns = [{"role": "user", "content": f"第{i}个问题"} for i in range(20)]
    formatted = format_history(turns, max_turns=2)
    assert "第19个问题" in formatted
    assert "第0个问题" not in formatted

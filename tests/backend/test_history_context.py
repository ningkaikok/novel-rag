"""回答 prompt 携带对话历史的测试（M3.6 阶段二）。

在此之前，最终回答的 prompt 里只有「当前问题 + 检索证据」，历史仅用于查询改写，
所以"再展开讲讲""你刚才说的第二点"这类追问必然失效。这里测三件事：

1. 预算裁剪的行为是可解释的（带了几轮、丢了什么、为什么丢）
2. 历史和证据在 prompt 里严格分层——历史不能被当成可引用的事实来源
3. 不传历史时行为与改造前完全一致（自由问答、Agent Lab、评测脚本都不传）
"""

from chunk_model import SourceChunk
from generation_mixin import (
    PROMPT_TEMPLATE_WITH_HISTORY,
    build_history_block,
)


def _turns(n: int, content: str = "内容") -> list[dict]:
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"{content}{i}"}
        for i in range(n)
    ]


def test_keeps_only_the_most_recent_turns():
    """超过轮数上限时丢最旧的——指代对象几乎总在紧邻的一两轮里。"""
    text, trace = build_history_block(_turns(10), max_turns=4)

    assert trace["turns_used"] == 4
    assert trace["turns_available"] == 10
    assert "内容9" in text, "最近一轮必须留下"
    assert "内容0" not in text, "最旧的应该被丢掉"
    assert trace["truncated"] is True
    assert "4 轮上限" in trace["reason"]


def test_long_turn_is_truncated_not_dropped():
    """单轮太长时截断而不是整轮丢弃：知道"上一轮在聊什么"通常开头几句就够。"""
    turns = [{"role": "assistant", "content": "顾长风" * 500}]
    text, trace = build_history_block(turns, per_turn_chars=50)

    assert trace["turns_used"] == 1, "不该因为长就把这一轮丢掉"
    assert "…（略）" in text
    assert len(text) < 200
    assert "按 50 字截断" in trace["reason"]


def test_char_budget_drops_oldest_until_it_fits():
    """字数预算超了继续从最旧的丢，直到装得下。"""
    turns = [{"role": "user", "content": "问" * 100} for _ in range(6)]
    text, trace = build_history_block(turns, max_turns=6, max_chars=250, per_turn_chars=100)

    assert len(text) <= 250 + 20  # 允许角色前缀的少量开销
    assert trace["turns_used"] < 6
    assert "字预算" in trace["reason"]


def test_always_keeps_at_least_the_latest_turn():
    """哪怕最近一轮自己就超预算，也不能压成空的——那等于没有上下文。"""
    turns = [{"role": "user", "content": "问" * 400}]
    text, trace = build_history_block(turns, max_chars=10, per_turn_chars=400)

    assert trace["turns_used"] == 1
    assert text, "压成空串还不如不带历史，至少要留最近一轮"


def test_reason_explains_every_truncation():
    """「每次压缩都能在 trace 中解释」是本阶段的验收要求，不能是黑箱。"""
    _, trace = build_history_block(_turns(3))
    assert trace["reason"] == "未截断"

    _, trace = build_history_block(_turns(10), max_turns=2)
    assert "丢弃最旧" in trace["reason"]


def test_empty_history_produces_nothing():
    text, trace = build_history_block([])
    assert text == ""
    assert trace["turns_used"] == 0


def test_turns_without_content_are_ignored():
    """落库的轮次可能是空的（比如中断在第一个 token 之前）。"""
    turns = [{"role": "user", "content": ""}, {"role": "assistant", "content": "有内容"}]
    text, trace = build_history_block(turns)

    assert trace["turns_available"] == 1
    assert "有内容" in text


class _Rag:
    """只借 build_prompt，不碰检索和图线索。"""

    from generation_mixin import GenerationMixin

    build_prompt = GenerationMixin.build_prompt

    def _graph_hint(self, question):
        return ""


def _sources() -> list[SourceChunk]:
    return [SourceChunk("雾隐山庄", 3, "顾长风中的是蚀骨散", 0.1, "第三章 药方与往事")]


def test_prompt_without_history_is_unchanged():
    """不传历史时必须走原模板——自由问答、Agent Lab、评测脚本都依赖这个行为。"""
    prompt = _Rag().build_prompt("顾长风得了什么病？", _sources())

    assert "对话背景" not in prompt
    assert "蚀骨散" in prompt


def test_prompt_with_history_separates_background_from_evidence():
    """历史和证据必须分层：历史只帮理解问题，不能当成可引用的事实来源。

    这是整个 M3.6 最容易出事的地方——模型一旦拿历史当依据，回答里的 [n] 就不再
    对应真实片段，本项目最核心的引用可核验性会被直接破坏。
    """
    history = [
        {"role": "user", "content": "顾长风得了什么病？"},
        {"role": "assistant", "content": "中了蚀骨散[1]。"},
    ]
    prompt = _Rag().build_prompt("再展开讲讲", _sources(), history=history)

    assert "对话背景" in prompt
    assert "顾长风得了什么病？" in prompt, "历史内容要真的进 prompt"
    # 三条硬约束都要在
    assert "不能" in prompt and "引用它" in prompt
    assert "以原文片段为准" in prompt
    # 证据部分照旧
    assert "[1]" in prompt and "第三章" in prompt


def test_history_template_declares_evidence_precedence():
    """模板文本本身要写死"冲突时以原文为准"，防止后续编辑顺手删掉。"""
    assert "以原文片段为准" in PROMPT_TEMPLATE_WITH_HISTORY
    assert "只有编号原文片段才是可引用的证据" in PROMPT_TEMPLATE_WITH_HISTORY


def test_facts_text_alone_triggers_the_history_template():
    """结构化事实不需要逐字历史陪衬——只有它，也该走带「对话背景」段的模板。

    真实场景：摘要/事实按阈值触发时，可能只有 facts_text 没有 history。
    """
    prompt = _Rag().build_prompt("他后来呢", _sources(), facts_text="当前小说：《雾隐山庄》")

    assert "对话背景" in prompt
    assert "当前小说：《雾隐山庄》" in prompt


def test_facts_text_is_labeled_separately_from_verbatim_history():
    """三种背景来源要分开标注：逐字历史最可信，摘要次之，结构化事实是另一种
    可信度（查库/精确匹配得到，不会漂移，但仍不是可引用证据）。"""
    history = [{"role": "user", "content": "顾长风得了什么病？"}]
    prompt = _Rag().build_prompt(
        "再展开讲讲", _sources(), history=history, facts_text="提到过的人物：顾长风"
    )

    assert "（已知信息）提到过的人物：顾长风" in prompt
    assert "顾长风得了什么病？" in prompt, "逐字历史不能被结构化事实顶掉"


def test_facts_text_does_not_count_toward_the_char_budget():
    """和摘要一样，结构化事实不占 max_chars 预算——它很短，抢预算没有意义，
    反而可能把最近几轮原话挤掉。"""
    turns = [{"role": "user", "content": "问" * 100} for _ in range(3)]
    text, trace = build_history_block(
        turns, max_chars=310, facts_text="当前小说：《雾隐山庄》；提到过的人物：顾长风、沈砚之"
    )

    assert "当前小说" in text
    assert trace["facts_used"] is True
    assert "结构化已知信息" in trace["reason"]

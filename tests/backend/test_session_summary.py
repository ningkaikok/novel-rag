"""滚动会话摘要的回归测试（M3.6 阶段三/四）。

路线图给这一阶段点名了三类必须防住的回归，本文件按这三类组织：

1. **摘要漂移**——摘要是滚动累积的，一旦膨胀或掺进脑补内容，会在之后每一轮都
   以"背景"的身份出现，比一次答错持久得多
2. **事实丢失**——最近几轮的原话不能因为有了摘要就被替换掉；摘要只补更早的部分
3. **跨轮引用错误**——编号每轮独立重编，上一轮的 [1] 和这一轮的 [1] 是两段原文

外加两条成本约束：短会话一次模型都不调；生成失败不能让长会话突然失忆。
全部不连数据库、不调模型。
"""

import backend.main as main
from generation_mixin import build_history_block
from session_summary import build_summary, should_update, turns_to_summarize


def _turns(n: int) -> list[dict]:
    return [
        {
            "turn_index": i,
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"第{i}轮内容",
        }
        for i in range(n)
    ]


def _boom(_prompt):
    raise RuntimeError("模型限流")


# ------------------------------------------------------------- 成本：何时才更新


def test_short_session_never_triggers_a_summary():
    """绝大多数会话都是短的，它们一次额外模型调用都不该付。"""
    turns = _turns(4)
    pending = turns_to_summarize(turns, covered_through=-1, window_turns=6)

    assert pending == [], "全部还在逐字窗口里，没有任何一轮需要摘要"
    assert should_update(pending, every=4) is False


def test_update_waits_until_enough_turns_fell_out_of_the_window():
    """攒够阈值才更新——否则等于给每一轮追问都加一次模型调用。"""
    turns = _turns(9)  # 窗口 6 轮，掉出 3 轮
    pending = turns_to_summarize(turns, covered_through=-1, window_turns=6)

    assert [t["turn_index"] for t in pending] == [0, 1, 2]
    assert should_update(pending, every=4) is False, "只掉出 3 轮，还不到阈值"
    assert should_update(pending, every=2) is True


def test_covered_turns_are_not_summarized_twice():
    """滚动的含义：只处理上一版摘要还没覆盖到的部分，不重复烧钱也不重复计入。"""
    turns = _turns(14)
    pending = turns_to_summarize(turns, covered_through=3, window_turns=6)

    assert [t["turn_index"] for t in pending] == [4, 5, 6, 7]


def test_window_and_summary_partition_the_history_exactly():
    """摘要负责的 + 逐字窗口带的 = 全部历史，不重不漏。

    重叠会让同一轮既被压缩又被原样带上（浪费预算且自相矛盾），
    漏掉则是真正的事实丢失。
    """
    turns = _turns(12)
    pending = turns_to_summarize(turns, covered_through=-1, window_turns=6)
    text, _ = build_history_block(turns, max_turns=6, max_chars=10_000)

    summarized = {t["turn_index"] for t in pending}
    verbatim = {i for i in range(12) if f"第{i}轮内容" in text}
    assert summarized & verbatim == set(), "同一轮不能既进摘要又进逐字历史"
    assert summarized | verbatim == set(range(12)), "两边合起来必须覆盖全部历史"


# ------------------------------------------------------------------- 摘要漂移


def test_summary_is_hard_truncated_even_if_the_model_ignores_the_limit():
    """字数要求在提示词里只是软约束。摘要是每轮都要付的固定开销，
    涨上去就再也下不来，所以必须有机械兜底。"""
    summary = build_summary(None, _turns(4), lambda _p: iter(["漂" * 5000]), max_chars=100)

    assert summary is not None
    assert len(summary) <= 100 + len("…（略）")


def test_summary_does_not_replace_the_most_recent_turns():
    """有摘要不等于可以不带原话——最近几轮必须仍然逐字进 prompt。"""
    text, trace = build_history_block(
        _turns(3), summary="用户在读《雾隐山庄》，关注顾长风的病"
    )

    assert "第2轮内容" in text, "最近一轮的原话不能被摘要顶掉"
    assert "（更早对话的摘要）" in text
    assert trace["summary_used"] is True
    assert "摘要" in trace["reason"], "带没带摘要要能在 trace 里看出来"


def test_summary_sits_above_the_verbatim_turns():
    """顺序不是随意的：摘要更旧、可信度更低，放在前面，逐字原话紧挨着问题。"""
    text, _ = build_history_block(_turns(2), summary="更早的背景")

    assert text.index("更早的背景") < text.index("第0轮内容")


# ----------------------------------------------------------------- 跨轮引用错误


def test_citation_numbers_from_previous_answers_never_enter_the_background():
    """上一轮的 [1] 和这一轮的 [1] 是两段完全不同的原文。

    把带编号的旧回答原样放进背景，等于递给模型一个看起来合法、实际指向别处的
    编号。编号对"理解用户在问什么"毫无帮助，抹掉零损失。
    """
    turns = [
        {"turn_index": 0, "role": "user", "content": "顾长风得了什么病？"},
        {"turn_index": 1, "role": "assistant", "content": "中了蚀骨散[1]，药方在[2][3]。"},
    ]
    text, _ = build_history_block(turns)

    assert "[1]" not in text and "[2]" not in text and "[3]" not in text
    assert "蚀骨散" in text, "只抹编号，事实本身要留下"


def test_citation_numbers_are_stripped_from_the_summary_too():
    """摘要提示词里已经要求不要写编号，但那是软约束，模型经常照写不误。"""
    text, _ = build_history_block([], summary="讨论了蚀骨散[1]和解药[2]")

    assert "[1]" not in text and "[2]" not in text
    assert "解药" in text


# ------------------------------------------------------------------ 失败降级


def test_generation_failure_returns_none_and_reports_why():
    errors: list[str] = []
    assert build_summary("旧摘要", _turns(4), _boom, errors=errors) is None
    assert errors and "RuntimeError" in errors[0], "静默失效比失败本身更难查"


def test_empty_model_output_is_rejected_not_stored():
    """空摘要一旦落库，会把上一版有效摘要覆盖掉——那是真正的事实丢失。"""
    errors: list[str] = []
    assert build_summary("旧摘要", _turns(4), lambda _p: iter(["   "]), errors=errors) is None
    assert errors


def test_refresh_falls_back_to_previous_summary_when_generation_fails(monkeypatch):
    """一次模型调用失败不该让长会话突然失忆，也不该阻塞回答。"""
    monkeypatch.setattr(
        main,
        "load_session_summary",
        lambda _sid: {"summary": "上一版摘要", "covers_through": -1},
    )
    monkeypatch.setattr(
        main,
        "save_session_summary",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("失败时不该落库")),
    )
    monkeypatch.setattr(main, "_generate_for_model", lambda _p, _m: _boom(_p))

    errors: list[str] = []
    assert main._refresh_session_summary("s-1", _turns(12), errors) == "上一版摘要"
    assert errors


def test_refresh_still_answers_when_the_summary_table_is_unreadable(monkeypatch):
    """读摘要失败 = 退回"只带最近几轮原文"，也就是引入摘要之前的行为。"""
    monkeypatch.setattr(
        main,
        "load_session_summary",
        lambda _sid: (_ for _ in ()).throw(RuntimeError("表不存在")),
    )

    errors: list[str] = []
    assert main._refresh_session_summary("s-1", _turns(12), errors) is None
    assert errors


def test_refresh_uses_the_new_summary_even_if_persisting_it_fails(monkeypatch):
    """落库失败只意味着下一轮要重算一次，本轮算出来的摘要照样能用。"""
    monkeypatch.setattr(main, "load_session_summary", lambda _sid: None)
    monkeypatch.setattr(
        main,
        "save_session_summary",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("数据库断开")),
    )
    monkeypatch.setattr(main, "_generate_for_model", lambda _p, _m: iter(["新摘要"]))

    errors: list[str] = []
    assert main._refresh_session_summary("s-1", _turns(12), errors) == "新摘要"
    assert any("落库" in e for e in errors)


def test_ask_does_not_touch_the_summary_when_the_switch_is_off(client, monkeypatch):
    """默认关闭时必须是**零开销**：既不读摘要表，也不调模型。"""
    from tests.backend.test_endpoints import _FakeRag

    main.state["rag"] = _FakeRag()
    main.state["model"] = "fake-model"
    monkeypatch.setattr(main, "QUERY_REWRITE_ENABLED", False)
    monkeypatch.setattr(main, "load_turns", lambda _sid: _turns(12))
    monkeypatch.setattr(
        main,
        "load_session_summary",
        lambda _sid: (_ for _ in ()).throw(AssertionError("开关关着就不该碰摘要表")),
    )
    monkeypatch.setattr(main, "generate_ollama_prompt_stream", lambda p, model: iter(["好"]))

    resp = client.post("/api/ask", json={"question": "他后来呢", "session_id": "s-off"})

    assert resp.status_code == 200
    assert "摘要" not in resp.text


def test_ask_carries_the_summary_into_the_prompt_when_enabled(client, monkeypatch):
    seen: dict = {}

    from tests.backend.test_endpoints import _FakeRag

    class _Rag(_FakeRag):
        def build_prompt(self, question, sources, history=None, summary=None):
            seen["summary"] = summary
            return f"问题：{question}"

    main.state["rag"] = _Rag()
    main.state["model"] = "fake-model"
    monkeypatch.setattr(main, "QUERY_REWRITE_ENABLED", False)
    monkeypatch.setattr(main, "HISTORY_SUMMARY_ENABLED", True)
    monkeypatch.setattr(main, "load_turns", lambda _sid: _turns(12))
    monkeypatch.setattr(main, "load_session_summary", lambda _sid: None)
    monkeypatch.setattr(main, "save_session_summary", lambda *a, **k: None)
    monkeypatch.setattr(main, "_generate_for_model", lambda _p, _m: iter(["更早在聊雾隐山庄"]))
    monkeypatch.setattr(main, "generate_ollama_prompt_stream", lambda p, model: iter(["好"]))

    resp = client.post("/api/ask", json={"question": "他后来呢", "session_id": "s-on"})

    assert resp.status_code == 200
    assert seen["summary"] == "更早在聊雾隐山庄"
    assert "摘要" in resp.text, "带了摘要就要在 trace 的「对话背景」里说明"


def test_refresh_records_how_far_the_summary_covers(monkeypatch):
    """covers_through 记错了，滚动就会重复摘要或漏掉几轮。"""
    saved: dict = {}
    monkeypatch.setattr(main, "load_session_summary", lambda _sid: None)
    monkeypatch.setattr(
        main,
        "save_session_summary",
        lambda sid, summary, covers, model: saved.update(covers=covers, summary=summary),
    )
    monkeypatch.setattr(main, "_generate_for_model", lambda _p, _m: iter(["新摘要"]))

    main._refresh_session_summary("s-1", _turns(12), [])

    # 窗口 6 轮 → 掉出 0..5，摘要覆盖到 5
    assert saved["covers"] == 12 - main.HISTORY_MAX_TURNS - 1

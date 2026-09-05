"""结构化会话事实：书名 + 人物（M3.6 唯一剩下的功能项）。

和滚动摘要不同，这里**不调用任何模型**——书名来自这一轮真实检索到的证据本身，
人物名来自人物关系图里已确认存在的人名反查。要么查得到、要么精确匹配到，没有
"大概对"的中间态，所以默认随对话背景一起生效，不需要开关。不连真实数据库。
"""

import session_facts
from session_facts import (
    SessionFacts,
    extract_session_facts,
    format_facts_line,
)


def _turn(content: str = "", sources: list[dict] | None = None) -> dict:
    return {"content": content, "sources": sources}


def _source(novel: str) -> dict:
    return {"novel": novel, "chunk_id": 0, "chapter_title": "第一章", "text": "……"}


class _Conn:
    def __init__(self, names: list[str]):
        self._names = names
        self.queries: list[tuple] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params):
        self.queries.append((sql, params))
        return self

    def fetchall(self):
        return [{"name": n} for n in self._names]


# ------------------------------------------------------------------- 书名


def test_no_sources_means_no_novel():
    """自由问答或还没检索过：没有证据就没有"当前小说"可言，不该瞎猜。"""
    facts = extract_session_facts([_turn("你好"), _turn("在的")])
    assert facts.novels == []


def test_current_novel_is_the_most_recently_retrieved_one():
    """ "当前小说"应该是最近一轮实际检索到的书，不是历史上第一次提到的书。"""
    turns = [
        _turn(sources=[_source("雾隐山庄")]),
        _turn(sources=[_source("青梧镇异闻")]),
    ]
    facts = extract_session_facts(turns)
    assert facts.novels == ["雾隐山庄", "青梧镇异闻"]
    assert facts.novels[-1] == "青梧镇异闻"


def test_switching_back_bumps_the_novel_to_most_recent():
    """跨书切换后又切回来：去重要按"最近一次提到"排序，不是"第一次提到"。"""
    turns = [
        _turn(sources=[_source("雾隐山庄")]),
        _turn(sources=[_source("青梧镇异闻")]),
        _turn(sources=[_source("雾隐山庄")]),
    ]
    facts = extract_session_facts(turns)
    assert facts.novels == ["青梧镇异闻", "雾隐山庄"], "雾隐山庄最近才被提到，该排在最后"


# ------------------------------------------------------------------- 人物


def test_no_novel_means_no_character_lookup(monkeypatch):
    """没有确定的书就不去查人物图——查了也没有范围意义，白花一次数据库调用。"""
    monkeypatch.setattr(
        session_facts,
        "connect",
        lambda: (_ for _ in ()).throw(AssertionError("没有书名就不该查图")),
    )
    facts = extract_session_facts([_turn("你好")])
    assert facts.characters == []


def test_only_names_actually_mentioned_in_the_conversation_are_kept(monkeypatch):
    """图里存在的人名很多，但只应该留下真的在这次对话里出现过的那几个——
    不能把整本书的人物表都塞给模型。"""
    monkeypatch.setattr(
        session_facts, "connect", lambda: _Conn(["顾长风", "沈砚之", "白先生"])
    )
    turns = [
        _turn("顾长风得了什么病", sources=[_source("雾隐山庄")]),
        _turn("沈砚之呢"),
    ]
    facts = extract_session_facts(turns)
    assert facts.characters == ["顾长风", "沈砚之"]
    assert "白先生" not in facts.characters


def test_rejected_edges_are_excluded_from_the_lookup(monkeypatch):
    """M4 的人工审核结论优先于一切自动判断：查询必须带上"排除已拒绝边"的过滤，
    和 generation_mixin._graph_hint 是同一条规则。"""
    conn = _Conn([])
    monkeypatch.setattr(session_facts, "connect", lambda: conn)

    extract_session_facts([_turn(sources=[_source("雾隐山庄")])])

    sql = conn.queries[0][0]
    assert "review_status" in sql and "rejected" in sql


def test_character_list_evicts_the_least_recently_mentioned_first(monkeypatch):
    """容量有上限：超出时按"最近提到"排序丢最旧的，和逐字历史同一套牺牲策略。"""
    monkeypatch.setattr(session_facts, "connect", lambda: _Conn(["甲", "乙", "丙", "丁"]))
    turns = [
        _turn(sources=[_source("雾隐山庄")]),
        _turn("甲"),
        _turn("乙"),
        _turn("丙"),
        _turn("丁"),
    ]

    facts = extract_session_facts(turns, max_characters=2)

    assert facts.characters == ["丙", "丁"], "该留的是最近提到的两个"


def test_a_name_mentioned_again_is_bumped_to_most_recent(monkeypatch):
    monkeypatch.setattr(session_facts, "connect", lambda: _Conn(["甲", "乙"]))
    turns = [
        _turn(sources=[_source("雾隐山庄")]),
        _turn("甲"),
        _turn("乙"),
        _turn("甲又出现了"),
    ]

    facts = extract_session_facts(turns, max_characters=1)

    assert facts.characters == ["甲"], "甲最近又被提到，不该被乙挤掉"


def test_graph_lookup_failure_degrades_to_novels_only(monkeypatch):
    """人物关系图可能没建（GRAPH_ENABLED=0）或库暂时不可用：这是纯增强信息，
    缺了应该退回"只有书名"，不能让整个提取失败。"""
    monkeypatch.setattr(
        session_facts, "connect", lambda: (_ for _ in ()).throw(RuntimeError("表不存在"))
    )
    facts = extract_session_facts([_turn(sources=[_source("雾隐山庄")])])

    assert facts.novels == ["雾隐山庄"]
    assert facts.characters == []


# ------------------------------------------------------------------- 渲染


def test_empty_facts_render_to_empty_string():
    assert format_facts_line(SessionFacts()) == ""


def test_single_novel_no_switch():
    text = format_facts_line(SessionFacts(novels=["雾隐山庄"]))
    assert text == "当前小说：《雾隐山庄》"


def test_novel_switch_mentions_the_earlier_book_too():
    text = format_facts_line(SessionFacts(novels=["雾隐山庄", "青梧镇异闻"]))
    assert "当前小说：《青梧镇异闻》" in text
    assert "雾隐山庄" in text, "切换过的书不该完全消失，只是不再是当前"


def test_characters_are_appended_after_the_novel():
    text = format_facts_line(
        SessionFacts(novels=["雾隐山庄"], characters=["顾长风", "沈砚之"])
    )
    assert text == "当前小说：《雾隐山庄》；提到过的人物：顾长风、沈砚之"


def test_characters_alone_without_a_novel():
    """理论上不会发生（没有书名就不查人物），但渲染函数本身不该因此崩溃。"""
    text = format_facts_line(SessionFacts(characters=["顾长风"]))
    assert text == "提到过的人物：顾长风"

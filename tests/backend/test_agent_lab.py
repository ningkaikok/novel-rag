"""Agent Lab 的有限步循环测试：不调用真实模型和 PostgreSQL。"""

import json

import pytest

import agent_lab
from rag import SourceChunk


def _source(chunk_id: int) -> SourceChunk:
    return SourceChunk("雾隐山庄", chunk_id, f"证据{chunk_id}", 0.0, "第一章")


class _FakeRag:
    def __init__(self):
        self.prompt = ""

    def build_prompt(self, question, sources, history=None):
        self.prompt = f"{question}|" + ",".join(str(source.chunk_id) for source in sources)
        return self.prompt


class _FakeToolbox:
    def __init__(self, _rag):
        pass

    def execute(self, name, args):
        assert name == "search_novels"
        assert args["query"] == "庄主是谁"
        return agent_lab.ToolResult("找到两个片段", [_source(2), _source(3)])


class _CatalogToolbox:
    def __init__(self, _rag):
        self.calls = []

    def execute(self, name, args):
        self.calls.append((name, args))
        if name == "query_library":
            return agent_lab.ToolResult(
                "书架包含：甲、乙、丙",
                facts={
                    "kind": "library_query",
                    "coverage": "complete",
                    "domain": "books",
                    "operation": "list",
                    "total": 3,
                    "items": ["甲", "乙", "丙"],
                },
            )
        if name == "search_novels":
            return agent_lab.ToolResult("找到局部片段", [_source(1)])
        raise AssertionError(f"未预期的工具调用：{name}")


def test_agent_searches_then_answers_with_selected_citations(monkeypatch):
    rag = _FakeRag()
    actions = iter(
        [
            {"reason": "先找原文", "tool": "search_novels", "args": {"query": "庄主是谁"}},
            {
                "reason": "证据足够",
                "tool": "answer_with_citations",
                "args": {"source_ids": ["S2"]},
            },
        ]
    )
    monkeypatch.setattr(agent_lab, "AgentToolbox", _FakeToolbox)

    events = list(
        agent_lab.run_agent(
            "庄主是谁",
            rag=rag,
            planner=lambda _prompt: json.dumps(next(actions), ensure_ascii=False),
            answerer=lambda _prompt: iter(["顾长风", "[1]"]),
            max_steps=5,
        )
    )

    steps = [value for kind, value in events if kind == "agent_step"]
    sources = next(value for kind, value in events if kind == "sources")
    assert [step["tool"] for step in steps] == [
        "search_novels",
        "answer_with_citations",
    ]
    assert steps[0]["source_ids"] == ["S1", "S2"]
    assert [source.chunk_id for source in sources] == [3]
    assert rag.prompt == "庄主是谁|3"
    assert [value for kind, value in events if kind == "token"] == ["顾长风", "[1]"]
    assert events[-1] == ("done", {})


def test_catalog_questions_use_complete_facts_not_retrieval_count(monkeypatch):
    """全集问题不能把 top-k 命中的书误当成书架总数。"""
    toolbox_holder = {}

    def _make_toolbox(rag):
        box = _CatalogToolbox(rag)
        toolbox_holder["box"] = box
        return box

    monkeypatch.setattr(agent_lab, "AgentToolbox", _make_toolbox)
    actions = iter(
        [
            # 即使规划器误选 search，coverage 门禁也应先列完整目录。
            {"reason": "先搜索", "tool": "search_novels", "args": {"query": "现在有几部小说"}},
            {
                "reason": "回答",
                "tool": "answer_with_citations",
                "args": {"source_ids": ["S1"]},
            },
        ]
    )

    events = list(
        agent_lab.run_agent(
            "现在一共有几部小说",
            rag=_FakeRag(),
            planner=lambda _prompt: json.dumps(next(actions), ensure_ascii=False),
            answerer=lambda _prompt: (_ for _ in ()).throw(
                AssertionError("完整目录事实应走确定性回答")
            ),
            max_steps=3,
        )
    )

    steps = [value for kind, value in events if kind == "agent_step"]
    assert steps[0]["tool"] == "query_library"
    answer = "".join(value for kind, value in events if kind == "token")
    assert "3 部小说" in answer
    assert all(book in answer for book in ["甲", "乙", "丙"])
    assert toolbox_holder["box"].calls[0][0] == "query_library"


class _EmptyToolbox:
    def __init__(self, _rag):
        pass

    def execute(self, _name, _args):
        return agent_lab.ToolResult("没有命中", [])


def test_agent_never_exceeds_step_limit_and_refuses_without_evidence(monkeypatch):
    monkeypatch.setattr(agent_lab, "AgentToolbox", _EmptyToolbox)
    action = json.dumps(
        {"reason": "继续找", "tool": "search_novels", "args": {"query": "不存在"}},
        ensure_ascii=False,
    )

    events = list(
        agent_lab.run_agent(
            "不存在的问题",
            rag=_FakeRag(),
            planner=lambda _prompt: action,
            answerer=lambda _prompt: (_ for _ in ()).throw(
                AssertionError("无证据时不能调用生成模型")
            ),
            max_steps=3,
        )
    )

    steps = [value for kind, value in events if kind == "agent_step"]
    assert len(steps) == 3
    assert max(step["step"] for step in steps) == 3
    assert not any(kind == "sources" for kind, _value in events)
    assert "无法给出有依据的回答" in next(value for kind, value in events if kind == "token")


def test_invalid_planner_output_falls_back_to_search(monkeypatch):
    monkeypatch.setattr(agent_lab, "AgentToolbox", _FakeToolbox)
    events = list(
        agent_lab.run_agent(
            "庄主是谁",
            rag=_FakeRag(),
            planner=lambda _prompt: "这不是 JSON",
            answerer=lambda _prompt: iter(["答案[1]"]),
            max_steps=3,
        )
    )

    first = next(value for kind, value in events if kind == "agent_step")
    assert first["tool"] == "search_novels"
    assert "规划格式无效" in first["reason"]


class _FakeConnCtx:
    """假的 `with connect() as conn:` 上下文管理器，只支持 `_resolve_novel` 用到的查询。"""

    def __init__(self, novels):
        self._rows = [{"novel": n} for n in novels]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, _sql, _params=None):
        return self

    def fetchall(self):
        return self._rows


_REAL_TITLE = "《诡秘之主》（精校版全本）作者：爱潜水的乌贼"


def test_resolve_novel_falls_back_to_typo_tolerant_match(monkeypatch):
    """规划器常把用户打错的书名原样传进来（用户问"闺蜜之主"，规划器就传
    novel="闺蜜之主"）。子串匹配对这种情况必然失败——"闺蜜之主"不是
    "《诡秘之主》…"的子串。主对话链路已经用编辑距离容差解决过这个问题，
    这里必须退回同一套逻辑，而不是让工具直接报错、逼得规划器在坏参数上空转。
    """
    monkeypatch.setattr(agent_lab, "connect", lambda: _FakeConnCtx([_REAL_TITLE]))
    toolbox = agent_lab.AgentToolbox(rag=None)
    assert toolbox._resolve_novel("闺蜜之主") == _REAL_TITLE


def test_resolve_novel_still_rejects_truly_unknown_titles(monkeypatch):
    """纠错要有边界——完全不相关的书名不能被容错逻辑误接受。"""
    monkeypatch.setattr(agent_lab, "connect", lambda: _FakeConnCtx([_REAL_TITLE]))
    toolbox = agent_lab.AgentToolbox(rag=None)
    with pytest.raises(ValueError):
        toolbox._resolve_novel("完全不相关的书名")


class _FlakyToolbox:
    """read_neighbors 永远失败，search_novels 永远成功——用来复现"规划器在同一个
    坏参数上反复重试、每次只改无关参数"的场景。
    """

    def __init__(self, _rag):
        self.calls: list[tuple[str, dict]] = []

    def execute(self, name, args):
        self.calls.append((name, dict(args)))
        if name == "read_neighbors":
            raise ValueError("无法唯一确定小说：闺蜜之主")
        if name == "search_novels":
            return agent_lab.ToolResult("找到证据", [_source(9)])
        raise AssertionError(f"未预期的工具调用：{name}")


def test_repeated_failures_are_blocked_even_when_args_differ(monkeypatch):
    """同一工具连续失败两次后，第三次哪怕参数变了（radius 1→3→0）也要被拦下来。

    之前的"重复动作检测"按完整 (tool, args) 精确匹配去重，radius 变了签名就
    不同，检测形同虚设——实测规划器会靠着这个漏洞，把 5 步预算里 3 步都烧在
    同一个打错的书名上（见 docs/grounding-verification.md 之前的对话记录）。
    """
    toolbox_holder: dict[str, _FlakyToolbox] = {}

    def _make_toolbox(rag):
        box = _FlakyToolbox(rag)
        toolbox_holder["box"] = box
        return box

    monkeypatch.setattr(agent_lab, "AgentToolbox", _make_toolbox)

    actions = iter(
        [
            {
                "reason": "读邻居，半径1",
                "tool": "read_neighbors",
                "args": {"novel": "闺蜜之主", "chunk_id": 1, "radius": 1},
            },
            {
                "reason": "读邻居，半径3",
                "tool": "read_neighbors",
                "args": {"novel": "闺蜜之主", "chunk_id": 1, "radius": 3},
            },
            {
                "reason": "读邻居，半径0",
                "tool": "read_neighbors",
                "args": {"novel": "闺蜜之主", "chunk_id": 1, "radius": 0},
            },
        ]
    )

    events = list(
        agent_lab.run_agent(
            "闺蜜之主里面哪些人是穿越过来的",
            rag=_FakeRag(),
            planner=lambda _prompt: json.dumps(next(actions), ensure_ascii=False),
            answerer=lambda _prompt: iter([]),
            max_steps=3,
        )
    )

    calls = toolbox_holder["box"].calls
    assert [name for name, _args in calls] == [
        "read_neighbors",
        "read_neighbors",
        "search_novels",
    ], "第三次不该再真的执行 read_neighbors——同一工具连续失败两次就该被拦住"

    steps = [value for kind, value in events if kind == "agent_step"]
    assert steps[-1]["tool"] == "search_novels"
    assert "反复失败" in steps[-1]["reason"]

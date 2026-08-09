"""Agent Lab 的有限步循环测试：不调用真实模型和 PostgreSQL。"""
import json

import agent_lab
from rag import SourceChunk


def _source(chunk_id: int) -> SourceChunk:
    return SourceChunk("雾隐山庄", chunk_id, f"证据{chunk_id}", 0.0, "第一章")


class _FakeRag:
    def __init__(self):
        self.prompt = ""

    def build_prompt(self, question, sources):
        self.prompt = f"{question}|" + ",".join(str(source.chunk_id) for source in sources)
        return self.prompt


class _FakeToolbox:
    def __init__(self, _rag):
        pass

    def execute(self, name, args):
        assert name == "search_novels"
        assert args["query"] == "庄主是谁"
        return agent_lab.ToolResult("找到两个片段", [_source(2), _source(3)])


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
    assert "无法给出有依据的回答" in next(
        value for kind, value in events if kind == "token"
    )


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

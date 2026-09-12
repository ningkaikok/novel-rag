"""Agent Lab 的有限步循环测试：不调用真实模型和 PostgreSQL。"""

import json

import pytest

import agent_lab
from rag import SourceChunk


def _source(chunk_id: int) -> SourceChunk:
    return SourceChunk("雾隐山庄", chunk_id, f"证据{chunk_id}", 0.0, "第一章")


class _NoDocumentsConn:
    """假的 postgres 连接：知识库里没有任何 V2 文档。

    ``run_agent`` 的兜底检索现在会查一次 ``knowledge_v2.documents`` 判断问题
    是不是在问某份已上传文档；这里让它查到"没有文档"，兜底行为回到默认的
    search_novels，和这份文件"不调用真实 PostgreSQL"的约定保持一致。
    """

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def execute(self, *_args):
        return self

    def fetchall(self):
        return []


@pytest.fixture(autouse=True)
def _no_v2_documents_by_default(monkeypatch):
    """兜底检索（``_fallback_search_action``）现在会查一次
    ``knowledge_v2.documents`` 判断问题是不是在问某份已上传文档；默认让它查到
    "没有文档"，保持这份文件"不连真实 PostgreSQL"的约定，不依赖本机数据库
    里恰好有什么内容。需要模拟"确实有匹配文档"的测试自己在测试体内覆盖
    ``agent_lab.connect``（覆盖会在该测试内生效，不影响其他测试）。
    """
    monkeypatch.setattr(agent_lab, "connect", lambda: _NoDocumentsConn())


class _FakeRag:
    def __init__(self):
        self.prompt = ""

    def build_prompt(self, question, sources, history=None, summary=None, facts_text=None):
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


def test_generation_failure_does_not_crash_the_generator(monkeypatch):
    """最终生成模型调用失败（比如 claude CLI 认证过期）必须被捕获，而不是让异常从
    run_agent 里逃逸——否则 SSE 连接会被异常直接中断，前端收不到 done 事件，
    界面会卡在"正在思考"上不再更新（这是本次要修的真实卡死场景）。
    """
    actions = iter(
        [
            {"reason": "先找原文", "tool": "search_novels", "args": {"query": "庄主是谁"}},
            {
                "reason": "证据足够",
                "tool": "answer_with_citations",
                "args": {"source_ids": ["S1"]},
            },
        ]
    )
    monkeypatch.setattr(agent_lab, "AgentToolbox", _FakeToolbox)

    def _failing_answerer(_prompt):
        raise RuntimeError("claude CLI 调用失败（exit 1）：OAuth session expired")

    events = list(
        agent_lab.run_agent(
            "庄主是谁",
            rag=_FakeRag(),
            planner=lambda _prompt: json.dumps(next(actions), ensure_ascii=False),
            answerer=_failing_answerer,
            max_steps=5,
        )
    )

    # 没有异常向上抛——list() 能跑完就是最重要的断言。
    assert events[-1] == ("done", {})
    answer = "".join(value for kind, value in events if kind == "token")
    assert "OAuth session expired" in answer


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


# ------------------------------------------------- 兜底检索：小说 vs 上传文档


class _OneDocumentConn:
    """假的 postgres 连接：知识库里恰好有一份指定标题的 V2 文档。"""

    def __init__(self, title):
        self._title = title

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def execute(self, *_args):
        return self

    def fetchall(self):
        return [{"title": self._title}]


def test_document_display_title_strips_extension_and_underscores():
    assert (
        agent_lab._document_display_title("方案C历史推演_2022决定2023入学.md")
        == "方案C历史推演 2022决定2023入学"
    )


def test_shares_long_substring_matches_meaningful_overlap():
    assert agent_lab._shares_long_substring(
        "方案C历史推演 内容总结", "方案C历史推演 2022决定2023入学"
    )


def test_shares_long_substring_rejects_unrelated_text():
    assert not agent_lab._shares_long_substring("庄主是谁", "方案C历史推演 2022决定2023入学")


def test_fallback_search_action_prefers_documents_when_title_matches(monkeypatch):
    """兜底检索不能再硬编码猜"一定是小说"：问题明确提到了已上传文档的标题，
    就该选 search_documents，而不是对着一份文档问题去搜小说。
    """
    monkeypatch.setattr(
        agent_lab, "connect", lambda: _OneDocumentConn("方案C历史推演_2022决定2023入学.md")
    )

    action = agent_lab._fallback_search_action("方案C历史推演 内容总结", "测试")

    assert action["tool"] == "search_documents"


def test_fallback_search_action_defaults_to_novels_without_a_match(monkeypatch):
    """问题没提到任何已知文档标题时，兜底仍然默认小说——这是改动前的行为，
    不应该因为加了文档检索就对普通小说问题变得不确定。
    """
    monkeypatch.setattr(
        agent_lab, "connect", lambda: _OneDocumentConn("方案C历史推演_2022决定2023入学.md")
    )

    action = agent_lab._fallback_search_action("庄主是谁", "测试")

    assert action["tool"] == "search_novels"


class _CatalogLoopToolbox:
    """模拟规划器在 query_library 上反复兜圈子：域名一直换、也不报错，但从来
    没有真正检索过内容——这是实测复现过的真实故障（截图：5 步全部选中
    query_library，从未尝试过 search_documents，最终无法给出答案）。
    """

    def __init__(self, _rag):
        self.calls: list[tuple[str, dict]] = []

    def execute(self, name, args):
        self.calls.append((name, dict(args)))
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
        if name == "search_documents":
            return agent_lab.ToolResult(
                "检索到 1 个相关文档片段",
                [_source(0)],
            )
        raise AssertionError(f"未预期的工具调用：{name}")


def test_repeated_query_library_picks_are_capped_before_exhausting_budget(monkeypatch):
    """query_library 是一次性目录工具：规划器反复选中它（哪怕参数不同、哪怕
    每次都"成功"）不该烧光整个步数预算——第 3 次选中就该被拦下来，强制换成
    真正的检索（并按标题匹配正确路由到 search_documents，而不是硬猜小说）。
    """
    toolbox_holder: dict[str, _CatalogLoopToolbox] = {}

    def _make_toolbox(rag):
        box = _CatalogLoopToolbox(rag)
        toolbox_holder["box"] = box
        return box

    monkeypatch.setattr(agent_lab, "AgentToolbox", _make_toolbox)
    monkeypatch.setattr(
        agent_lab, "connect", lambda: _OneDocumentConn("方案C历史推演_2022决定2023入学.md")
    )

    actions = iter(
        [
            {"reason": "先看看目录", "tool": "query_library", "args": {"domain": "books"}},
            {"reason": "再看看章节", "tool": "query_library", "args": {"domain": "chapters"}},
            {"reason": "还是想看目录", "tool": "query_library", "args": {"domain": "chunks"}},
        ]
    )

    events = list(
        agent_lab.run_agent(
            "方案C历史推演 内容总结",
            rag=_FakeRag(),
            planner=lambda _prompt: json.dumps(next(actions), ensure_ascii=False),
            answerer=lambda _prompt: iter(["总结内容"]),
            max_steps=5,
        )
    )

    tool_sequence = [name for name, _args in toolbox_holder["box"].calls]
    assert tool_sequence.count("query_library") <= 2, "一次性目录工具最多信任两次"
    assert "search_documents" in tool_sequence, "反复卡住后必须换成真正的检索"
    answer = "".join(value for kind, value in events if kind == "token")
    assert answer == "总结内容", "应该真正拿到证据并生成回答，而不是耗光步数后拒答"

"""动作解析的埋点（路线图 M3.2.1 前置项）。

M3.2.1 想把 JSON 动作协议换成首行标签协议。三条理由（不能流式、正则兜底是猜、
解析失败没有回路）在道理上都成立，但**在自家模型上多久出一次从来没测过**。
路线图要求先埋点、用真实失败率决定排期。这里测的是埋点本身可信：

1. 四档 parse_mode 分得清（尤其 regex——它是"成功了但成功得可疑"）
2. 规划器没参与的步骤不计入，否则会把失败率冲淡成假结论
3. 埋点跟着步骤落库，事后能聚合

不连数据库、不调模型。
"""

import json

import pytest

import agent_lab
from agent_lab import ActionParseError, _parse_action
from scripts.agent_parse_stats import _report, collect

_ACTION = {"reason": "先找原文", "tool": "search_novels", "args": {"query": "庄主"}}


class _Toolbox:
    def __init__(self, _rag):
        pass

    def execute(self, _name, _args):
        return agent_lab.ToolResult("找到证据", [])


class _SourceToolbox:
    """会真的返回证据，用来触发"最后一步强制收尾"这条分支。"""

    def __init__(self, _rag):
        pass

    def execute(self, _name, _args):
        from chunk_model import SourceChunk

        return agent_lab.ToolResult(
            "找到证据", [SourceChunk("雾隐山庄", 1, "原文", 0.0, "第一章")]
        )


class _CatalogToolbox:
    def __init__(self, _rag):
        pass

    def execute(self, _name, _args):
        return agent_lab.ToolResult(
            "书架包含：甲",
            facts={
                "kind": "library_query",
                "coverage": "complete",
                "domain": "books",
                "operation": "list",
                "total": 1,
                "items": ["甲"],
            },
        )


class _FakeRag:
    def build_prompt(self, question, sources, history=None, summary=None):
        return f"问题：{question}"


# ----------------------------------------------------------------- 四档分得清


def test_bare_json_is_strict():
    action, mode = _parse_action(json.dumps(_ACTION, ensure_ascii=False))
    assert mode == "strict"
    assert action["tool"] == "search_novels"


def test_markdown_fence_is_counted_separately():
    """剥围栏本身无害，但它说明模型没遵守"只输出 JSON"——要能和 strict 分开看。"""
    raw = "```json\n" + json.dumps(_ACTION, ensure_ascii=False) + "\n```"
    _, mode = _parse_action(raw)
    assert mode == "fenced"


def test_prose_around_json_falls_back_to_regex():
    """这是最值得看的一档：解析成功了，但靠的是"抓第一个花括号"。"""
    raw = "好的，我打算先检索：" + json.dumps(_ACTION, ensure_ascii=False) + " 这样比较稳妥。"
    action, mode = _parse_action(raw)
    assert mode == "regex"
    assert action["tool"] == "search_novels"


def test_failures_carry_an_aggregatable_category():
    """失败要能分类，否则统计出来只有一个"失败"没法指导改法。"""
    with pytest.raises(ActionParseError) as no_json:
        _parse_action("我觉得应该先搜索一下")
    assert no_json.value.category == "no_json"

    with pytest.raises(ActionParseError) as bad_shape:
        _parse_action('{"tool": "search_novels", "args": "不是对象"}')
    assert bad_shape.value.category == "bad_shape"

    with pytest.raises(ActionParseError) as broken:
        _parse_action("前言 {tool: 没引号} 后语")
    assert broken.value.category == "invalid_json"


def test_valid_json_with_wrong_shape_is_not_rescued_by_the_regex_path():
    """JSON 合法但形状不对，换个剥法也不会变对——不能让正则兜底把它掩盖成 regex。"""
    with pytest.raises(ActionParseError) as exc:
        _parse_action('```json\n{"tool": "x", "args": 1}\n```')
    assert exc.value.category == "bad_shape"


# --------------------------------------------------------- 埋点跟着步骤走


def _run(planner_output: str, monkeypatch) -> list[dict]:
    monkeypatch.setattr(agent_lab, "AgentToolbox", _Toolbox)
    events = list(
        agent_lab.run_agent(
            "雾隐山庄的庄主是谁",
            rag=None,
            planner=lambda _p: planner_output,
            answerer=lambda _p: iter([]),
            max_steps=3,
        )
    )
    return [value for kind, value in events if kind == "agent_step"]


def test_每一步都带上解析方式(monkeypatch):
    steps = _run("```json\n" + json.dumps(_ACTION, ensure_ascii=False) + "\n```", monkeypatch)
    assert steps and all(step["parse_mode"] == "fenced" for step in steps)


def test_failed_steps_record_the_category(monkeypatch):
    steps = _run("这不是 JSON", monkeypatch)
    assert steps[0]["parse_mode"] == "failed:no_json"
    assert "规划格式无效" in steps[0]["reason"], "降级原因仍要对用户可见"


def test_overridden_actions_still_record_the_parse_that_happened(monkeypatch):
    """目录门禁会丢掉规划器选的动作，但**解析这件事确实发生了**。

    埋点问的是"解析多久失败一次"，不是"这个动作最后有没有被采用"。把这类步骤
    记成 None 会让分母莫名其妙地少掉一批真实样本。
    """
    monkeypatch.setattr(agent_lab, "AgentToolbox", _CatalogToolbox)

    steps = [
        value
        for kind, value in agent_lab.run_agent(
            "现在一共有几部小说",  # 目录问题：第一步被 coverage 门禁强制改写
            rag=None,
            planner=lambda _p: json.dumps(_ACTION, ensure_ascii=False),
            answerer=lambda _p: iter([]),
            max_steps=3,
        )
        if kind == "agent_step"
    ]

    assert steps[0]["tool"] == "query_library", "动作被门禁换掉了"
    assert steps[0]["parse_mode"] == "strict", "但解析成功这件事仍要如实记下"


def test_steps_where_the_planner_was_never_called_are_not_counted(monkeypatch):
    """最后一步的强制收尾根本没调规划器，算进失败率会把统计冲淡成假结论。"""
    monkeypatch.setattr(agent_lab, "AgentToolbox", _SourceToolbox)
    # 两步用不同参数，避开"重复动作"拦截，让循环真的走到第三步的强制收尾
    actions = iter(
        [
            json.dumps({**_ACTION, "args": {"query": "庄主"}}, ensure_ascii=False),
            json.dumps({**_ACTION, "args": {"query": "顾长风"}}, ensure_ascii=False),
        ]
    )

    steps = [
        value
        for kind, value in agent_lab.run_agent(
            "雾隐山庄的庄主是谁",
            rag=_FakeRag(),
            planner=lambda _p: next(actions),
            answerer=lambda _p: iter(["答案"]),
            max_steps=3,
        )
        if kind == "agent_step"
    ]

    assert steps[-1]["tool"] == "answer_with_citations"
    assert steps[-1]["parse_mode"] is None, "规划器没参与的步骤不该被计入解析统计"
    assert steps[0]["parse_mode"] == "strict"


# ------------------------------------------------------------------ 聚合脚本


class _Conn:
    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, _sql, _params=()):
        return self

    def fetchall(self):
        return self._rows


def test_stats_ignores_steps_without_a_parse_mode(monkeypatch):
    """旧记录没有这个字段、强制步骤是 None——两者都不能算进分母。"""
    import scripts.agent_parse_stats as stats

    rows = [
        {
            "run_config": {"generate_model": "qwen2.5:7b"},
            "agent_steps": [
                {"parse_mode": "strict"},
                {"parse_mode": "regex"},
                {"parse_mode": None},  # 强制收尾
                {},  # 埋点上线前的旧记录
            ],
        }
    ]
    monkeypatch.setattr(stats, "connect", lambda: _Conn(rows))

    modes, _ = collect()

    assert sum(modes.values()) == 2, "分母只能是规划器真正产出的步骤"
    assert modes["regex"] == 1


def test_report_highlights_suspicious_share(monkeypatch):
    import scripts.agent_parse_stats as stats

    rows = [
        {
            "run_config": None,
            "agent_steps": [{"parse_mode": "strict"}] * 8
            + [{"parse_mode": "regex"}, {"parse_mode": "failed:no_json"}],
        }
    ]
    monkeypatch.setattr(stats, "connect", lambda: _Conn(rows))

    text = _report(collect()[0])

    assert "20.0%" in text, "regex 和 failed 要合并成一个可判读的比例"
    assert "failed:no_json" in text


def test_report_says_so_when_there_is_nothing_to_measure(monkeypatch):
    """没有数据要说清楚是"还没人用过"还是"旧记录"，不能显示成 0% 让人误判。"""
    import scripts.agent_parse_stats as stats

    monkeypatch.setattr(stats, "connect", lambda: _Conn([]))
    assert "没有可统计的步骤" in _report(collect()[0])

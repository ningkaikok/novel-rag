"""tool_spec 注册表测试：不连数据库，只验证静态元数据与类型约束。

覆盖三块（对应 M6.1 前置项的验收）：
1. TOOL_REGISTRY 完整性——五个 Agent Lab 工具都在，元数据形状合法；
2. 版权红线——SourceRef 的 80 字摘录约束真的生效；
3. 一致性——MCP 服务器的实际注册与 Registry 逐项对得上。
"""

import asyncio
import inspect
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from tool_spec import (
    EXCERPT_MAX_CHARS,
    TOOL_REGISTRY,
    AnswerWithCitationsV1,
    SourceRef,
    ToolResultV1,
    ToolSpec,
    get_tool_spec,
)

EXPECTED_TOOLS = {
    "list_books",
    "search_novels",
    "read_neighbors",
    "get_chapter",
    "answer_with_citations",
}


# ---- 1. 注册表完整性 --------------------------------------------------------


def test_registry_covers_five_agent_lab_tools():
    assert set(TOOL_REGISTRY) == EXPECTED_TOOLS


def test_every_spec_is_well_formed():
    for name, spec in TOOL_REGISTRY.items():
        assert isinstance(spec, ToolSpec)
        assert spec.name == name, "注册键应与 spec.name 一致"
        assert spec.description, name
        # Agent Lab 五个工具全部只读；风险只允许 low/medium，没有写操作
        assert spec.readonly is True, name
        assert spec.risk_level in {"low", "medium"}, name
        assert spec.timeout_s > 0, name
        params = spec.params_json_schema
        assert params["type"] == "object" and "properties" in params, name
        assert set(params["required"]) <= set(params["properties"]), name
        result = spec.result_schema
        assert result["type"] == "object" and "properties" in result, name


def test_answer_tool_has_its_own_result_schema():
    query_result = ToolResultV1.model_json_schema()
    for name in ("list_books", "search_novels", "read_neighbors", "get_chapter"):
        assert TOOL_REGISTRY[name].result_schema == query_result, name
    answer_spec = TOOL_REGISTRY["answer_with_citations"]
    assert answer_spec.result_schema == AnswerWithCitationsV1.model_json_schema()
    assert answer_spec.result_schema != query_result
    # 回答的引用复用同一个 SourceRef 定义（$defs 里同名同形），红线只有一份
    assert answer_spec.result_schema["$defs"] == query_result["$defs"]


def test_registry_is_immutable_snapshot():
    """M6.1 要求运行时读取不可变快照，先从注册表本身不可改做起。"""
    with pytest.raises(TypeError):
        TOOL_REGISTRY["list_books"] = TOOL_REGISTRY["search_novels"]  # type: ignore[index]


def test_get_tool_spec_rejects_unknown_name():
    with pytest.raises(KeyError, match="nope"):
        get_tool_spec("nope")


# ---- 参数 schema 与方法签名的一致性 -----------------------------------------


def _signature_params(method) -> tuple[set[str], dict[str, object]]:
    """从 AgentToolbox 方法签名推导参数名/必填集合/默认值，用于交叉核对。"""
    sig = inspect.signature(method)
    names: set[str] = set()
    defaults: dict[str, object] = {}
    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        names.add(param_name)
        if param.default is not inspect.Parameter.empty:
            defaults[param_name] = param.default
    return names - set(defaults), defaults


@pytest.mark.parametrize(
    ("tool_name", "method_name"),
    [
        ("list_books", "list_books"),
        ("search_novels", "search_novels"),
        ("read_neighbors", "read_neighbors"),
        ("get_chapter", "get_chapter"),
    ],
)
def test_param_names_match_agent_lab_signatures(tool_name: str, method_name: str):
    """除改名转发外，Registry 的参数名与 AgentToolbox 签名一一对应。"""
    from agent_lab import AgentToolbox  # 延迟导入：避免本文件顶部就拉起依赖链

    required, _defaults = _signature_params(getattr(AgentToolbox, method_name))
    props = TOOL_REGISTRY[tool_name].params_json_schema["properties"]
    assert set(props) == required | set(_defaults)


def test_param_bounds_match_implementation_clamps():
    """边界是对照方法体内的夹取逻辑手工核对的，这里固化成断言防漂移。

    - search_novels：top_k = max(1, min(limit, 8))
    - read_neighbors：radius = max(0, min(radius, 3))
    - get_chapter：limit = max(1, min(int(limit), 12))
    """
    props_of = lambda n: TOOL_REGISTRY[n].params_json_schema["properties"]  # noqa: E731

    limit = props_of("search_novels")["limit"]
    assert (limit["minimum"], limit["maximum"], limit["default"]) == (1, 8, 5)

    radius = props_of("read_neighbors")["radius"]
    assert (radius["minimum"], radius["maximum"], radius["default"]) == (0, 3, 1)

    chapter_limit = props_of("get_chapter")["limit"]
    assert (chapter_limit["minimum"], chapter_limit["maximum"], chapter_limit["default"]) == (
        1,
        12,
        8,
    )

    source_ids = TOOL_REGISTRY["answer_with_citations"].params_json_schema["properties"][
        "source_ids"
    ]
    assert source_ids["type"] == "array" and source_ids["items"]["type"] == "string"


# ---- 2. 版权红线：80 字摘录约束 ---------------------------------------------


def _source_ref(excerpt: str) -> SourceRef:
    return SourceRef(novel="雾隐山庄", chapter="第一章", chunk_id=1, excerpt=excerpt)


def test_source_ref_accepts_excerpt_at_exact_limit():
    boundary = "长" * EXCERPT_MAX_CHARS
    assert len(_source_ref(boundary).excerpt) == EXCERPT_MAX_CHARS


def test_source_ref_rejects_excerpt_over_limit():
    with pytest.raises(ValidationError):
        _source_ref("长" * (EXCERPT_MAX_CHARS + 1))


def test_tool_result_v1_shape_and_defaults():
    result = ToolResultV1(summary="摘要")
    assert result.schema_version == "1"
    assert result.facts == {} and result.sources == []
    with_sources = ToolResultV1(
        summary="摘要",
        facts={"kind": "book_catalog"},
        sources=[_source_ref("短摘录")],
    )
    assert with_sources.sources[0].novel == "雾隐山庄"


def test_answer_result_v1_carries_citations():
    answer = AnswerWithCitationsV1(answer="答案是……", citations=[_source_ref("依据")])
    assert answer.schema_version == "1"
    assert answer.citations[0].chunk_id == 1


# ---- 3. MCP 注册与 Registry 的一致性 ----------------------------------------


def _mcp_server():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
    import mcp_server

    return mcp_server


def test_mcp_registration_matches_registry():
    mcp_server = _mcp_server()
    tools = {t.name: t for t in asyncio.run(mcp_server.server.list_tools())}
    # MCP 只暴露数据查询四件套；answer_with_citations 需要 LLM，属 Agent 循环
    assert set(tools) == EXPECTED_TOOLS - {"answer_with_citations"}

    for name, tool in tools.items():
        spec = TOOL_REGISTRY[name]
        # 名称与描述必须来自同一份 Registry，不允许手写第二份
        assert tool.description == spec.description, name

        rename = mcp_server._MCP_PARAM_ALIASES.get(name, {})
        registry_props = spec.params_json_schema["properties"]
        expected = {rename.get(key, key): value for key, value in registry_props.items()}
        actual_props = tool.input_schema.get("properties", {})
        # MCP 是 Agent Lab 接口的子集（例如 search 未暴露 novel 过滤）
        assert set(actual_props) <= set(expected), name
        for param, meta in actual_props.items():
            assert meta["type"] == expected[param]["type"], (name, param)

        expected_required = {
            rename.get(key, key)
            for key in spec.params_json_schema["required"]
            if rename.get(key, key) in actual_props
        }
        assert set(tool.input_schema.get("required", [])) == expected_required, name


def test_mcp_output_schema_matches_tool_result_v1():
    mcp_server = _mcp_server()
    for tool in asyncio.run(mcp_server.server.list_tools()):
        output_props = (tool.output_schema or {}).get("properties", {})
        assert set(output_props) == set(ToolResultV1.model_json_schema()["properties"]), (
            tool.name
        )

"""统一 ToolSpec：Agent Lab 工具接口的正式化定义（路线图 M6.1 前置项）。

MCP PoC（scripts/mcp_server.py）用真实客户端验证过 ToolResult 的投影形状后，
这里把结论沉淀为唯一事实来源：

- ``SourceRef`` / ``ToolResultV1``：查询类工具的统一返回形状，80 字摘录红线
  由类型约束直接保证；
- ``AnswerWithCitationsV1``：回答型工具单独的结果 schema；
- ``ToolSpec``：一个工具的完整静态描述——名称、描述、参数/结果 JSON Schema、
  只读属性、风险等级、超时；
- ``TOOL_REGISTRY``：五个 Agent Lab 工具的不可变注册表。MCP 服务器从它生成
  注册；后续 M6.1 正式项的权限/启停和 M6.2 的 Tool Gateway 也以它为准。

参数 schema 从 ``agent_lab.AgentToolbox`` 各方法签名推导，再对照方法体内的
夹取逻辑手工核对（limit/radius 的上下界以实现为准，见各条目的注释）。
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal

from pydantic import BaseModel, Field

# 结果形状版本号：结构变更时递增，消费方按它判断能否直接读取 facts/sources。
TOOL_RESULT_SCHEMA_VERSION: Literal["1"] = "1"

# 版权红线：sources 里每条摘录最长 80 字，完整原文只能凭定位信息自行查库。
EXCERPT_MAX_CHARS = 80


class SourceRef(BaseModel):
    """证据定位信息 + 截断摘录。

    novel/chapter/chunk_id 让消费方能自行取回完整原文；excerpt 超过
    ``EXCERPT_MAX_CHARS`` 时构造直接校验失败，而不是静默外泄原文。
    """

    novel: str
    chapter: str
    chunk_id: int
    excerpt: str = Field(max_length=EXCERPT_MAX_CHARS)


class ToolResultV1(BaseModel):
    """查询类工具的统一返回形状（list_books/search_novels/read_neighbors/get_chapter）。

    summary 给规划器快速阅读；facts 存可机器校验的结构化事实；sources 携带
    定位信息与截断摘录。
    """

    schema_version: Literal["1"] = TOOL_RESULT_SCHEMA_VERSION
    summary: str
    facts: dict[str, object] = Field(default_factory=dict)
    sources: list[SourceRef] = Field(default_factory=list)


class AnswerWithCitationsV1(BaseModel):
    """answer_with_citations 单独的结果 schema：最终回答文本 + 引用定位。

    与查询类不同，它的产出是面向用户的完整答案；引用复用 ``SourceRef``，
    摘录红线由同一个类型保证。
    """

    schema_version: Literal["1"] = TOOL_RESULT_SCHEMA_VERSION
    answer: str
    citations: list[SourceRef] = Field(default_factory=list)


class ToolSpec(BaseModel):
    """单个工具的静态描述——M6.1 Tool Registry 的最小可行单元。

    权限（permission）、启停和版本快照等 Control Plane 能力留给 M6.1 正式项；
    这里先固化 MCP PoC 验证过有用的六类元数据。所有字段都是声明式的：
    工具实现仍可以是普通 Python 方法，不与 spec 绑定。
    """

    name: str
    description: str
    params_json_schema: dict[str, object]
    result_schema: dict[str, object]
    readonly: bool = True
    risk_level: Literal["low", "medium", "high"] = "low"
    timeout_s: int = 30


def _params(properties: dict[str, object], required: list[str]) -> dict[str, object]:
    """把属性表 + 必填清单包装成标准 JSON Schema 对象，保持注册表条目简洁。"""
    return {"type": "object", "properties": properties, "required": required}


# ---- 参数 schema：从 AgentToolbox 方法签名推导，边界对照方法内夹取逻辑核对 --

_LIST_BOOKS_PARAMS = _params({}, [])

# AgentToolbox.search_novels(query, novel=None, limit=5)，实现内夹取 top_k 到 1~8
_SEARCH_NOVELS_PARAMS = _params(
    {
        "query": {"type": "string", "description": "检索问题或关键词"},
        "novel": {
            "type": ["string", "null"],
            "description": "可选书名，限定在该书内检索",
        },
        "limit": {
            "type": "integer",
            "minimum": 1,
            "maximum": 8,
            "default": 5,
            "description": "返回片段数上限",
        },
    },
    ["query"],
)

# AgentToolbox.read_neighbors(novel, chunk_id, radius=1)，radius 夹取到 0~3
_READ_NEIGHBORS_PARAMS = _params(
    {
        "novel": {"type": "string", "description": "书名（支持模糊解析为库里真实书名）"},
        "chunk_id": {"type": "integer", "description": "中心片段编号"},
        "radius": {
            "type": "integer",
            "minimum": 0,
            "maximum": 3,
            "default": 1,
            "description": "前后各取多少个相邻片段",
        },
    },
    ["novel", "chunk_id"],
)

# AgentToolbox.get_chapter(novel, chapter_title, limit=8)，limit 夹取到 1~12
_GET_CHAPTER_PARAMS = _params(
    {
        "novel": {"type": "string", "description": "书名"},
        "chapter_title": {"type": "string", "description": "章节标题（子串匹配）"},
        "limit": {
            "type": "integer",
            "minimum": 1,
            "maximum": 12,
            "default": 8,
            "description": "最多返回多少段",
        },
    },
    ["novel", "chapter_title"],
)

# 规划器约定动作：从观察里挑证据编号（S1、S2…），见 agent_lab._PLANNER_PROMPT
_ANSWER_WITH_CITATIONS_PARAMS = _params(
    {
        "source_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": "要引用的证据编号，来自此前工具观察返回的 source_ids",
        }
    },
    ["source_ids"],
)

_QUERY_RESULT_SCHEMA = ToolResultV1.model_json_schema()
_ANSWER_RESULT_SCHEMA = AnswerWithCitationsV1.model_json_schema()

# 前四个描述沿用 MCP PoC 发布时客户端可见的文案；answer_with_citations 是
# Agent 循环内部的收尾动作，不暴露给 MCP。
TOOL_REGISTRY: Mapping[str, ToolSpec] = MappingProxyType(
    {
        "list_books": ToolSpec(
            name="list_books",
            description="列出书架上全部小说及各自片段数",
            params_json_schema=_LIST_BOOKS_PARAMS,
            result_schema=_QUERY_RESULT_SCHEMA,
            timeout_s=10,
        ),
        "search_novels": ToolSpec(
            name="search_novels",
            description="混合检索小说原文，返回最相关的片段（含 80 字摘录与定位信息）",
            params_json_schema=_SEARCH_NOVELS_PARAMS,
            result_schema=_QUERY_RESULT_SCHEMA,
            # embedding 模型可能现场首次载入（数秒），超时放宽
            timeout_s=30,
        ),
        "read_neighbors": ToolSpec(
            name="read_neighbors",
            description="按片段编号读取前后相邻片段，用于核对上下文",
            params_json_schema=_READ_NEIGHBORS_PARAMS,
            result_schema=_QUERY_RESULT_SCHEMA,
            timeout_s=10,
        ),
        "get_chapter": ToolSpec(
            name="get_chapter",
            description="按章节标题取某书的章节内容片段列表",
            params_json_schema=_GET_CHAPTER_PARAMS,
            result_schema=_QUERY_RESULT_SCHEMA,
            timeout_s=10,
        ),
        "answer_with_citations": ToolSpec(
            name="answer_with_citations",
            description="基于选中的证据片段生成带引用的最终回答",
            params_json_schema=_ANSWER_WITH_CITATIONS_PARAMS,
            result_schema=_ANSWER_RESULT_SCHEMA,
            # 触发一次 LLM 生成：有 token 成本且答案是模型产出而非数据库真值
            risk_level="medium",
            timeout_s=60,
        ),
    }
)


def get_tool_spec(name: str) -> ToolSpec:
    """按名称取 spec；未知工具报错时附上可用清单，方便排错。"""
    try:
        return TOOL_REGISTRY[name]
    except KeyError:
        raise KeyError(f"未知工具：{name}（可用：{', '.join(TOOL_REGISTRY)}）") from None

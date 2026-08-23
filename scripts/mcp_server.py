#!/usr/bin/env python3
"""novel-rag 的只读 MCP 服务器（PoC，stdio 传输）。

定位（路线图 M6.6 允许的提前项）
--------------------------------
把 Agent Lab 已有的只读工具用 MCP 协议暴露给任意兼容客户端（Claude Code、
Cursor 等）。**只是探针**：不接权限体系、不做发现机制。用真实客户端检验
ToolResult schema 设计的结论已反哺进 src/tool_spec.py（M6.1 前置项）——本文件
不再手写工具元数据，名称、描述、返回模型一律取自 TOOL_REGISTRY。

安全边界
--------
- 只读：仅暴露查询类工具，没有任何写操作；
- 版权红线：sources 里每条摘录截断到 80 字 + 定位信息（书/章/片段号），
  约束由 tool_spec.SourceRef 的类型定义保证——绝不整段输出原文。

运行
----
    uv run mcp dev scripts/mcp_server.py        # Inspector 调试
    # 或在 Claude Code / 客户端配置里注册：
    #   command: uv  args: ["run", "python", "scripts/mcp_server.py"]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mcp.server.mcpserver import MCPServer  # noqa: E402

from agent_lab import AgentToolbox, ToolResult  # noqa: E402
from config import DATABASE_URL  # noqa: E402
from tool_spec import (  # noqa: E402
    EXCERPT_MAX_CHARS,
    TOOL_RESULT_SCHEMA_VERSION,
    SourceRef,
    ToolResultV1,
    get_tool_spec,
)

server = MCPServer(
    name="novel-rag",
    version="0.1.0",
    instructions=(
        "中文小说书架的只读查询工具。回答事实问题前先用 search_novels 检索原文，"
        "引用时给出 novel/chapter/chunk_id 定位信息。"
    ),
)

# 工具实现复用 Agent Lab 的 AgentToolbox：同一套 SQL 与版权红线只维护一份。
# rag=None 时 search 类工具不可用（无索引环境），list/query 类仍可工作。
_toolbox: AgentToolbox | None = None


def _toolbox_lazy(with_search: bool) -> AgentToolbox:
    """按需构建工具箱。

    rag 只在真正要检索时才加载（embedding 模型首次载入数秒）：
    list/query 这类纯 SQL 工具传 None 即可，AgentToolbox 仅存引用不校验。
    """
    global _toolbox
    if _toolbox is None:
        _toolbox = AgentToolbox(None)
    if with_search and _toolbox.rag is None:
        from embedder import load_embedder
        from rag import NovelRAG

        _toolbox.rag = NovelRAG(embedder=load_embedder())
    return _toolbox


def _to_payload(result: ToolResult) -> ToolResultV1:
    """ToolResult → MCP 结构化返回：摘要 + 定位化 sources + 可校验 facts。"""
    return ToolResultV1(
        schema_version=TOOL_RESULT_SCHEMA_VERSION,
        summary=result.summary,
        facts=dict(result.facts),
        sources=[
            SourceRef(
                novel=s.novel,
                chapter=s.chapter_title or "",
                chunk_id=s.chunk_id,
                # 版权红线：这里主动截断而不是依赖校验报错——超长是常态而非异常，
                # 完整原文请按定位信息自行查库
                excerpt=(s.text or "")[:EXCERPT_MAX_CHARS],
            )
            for s in result.sources
        ],
    )


# ---------------------------------------------------------------------------
# 工具注册：名称/描述/结果模型全部来自 TOOL_REGISTRY，这里只剩两类不得不写的
# 内容——MCP SDK 从函数签名反射生成 inputSchema（v2 约束：必须写具体注解，
# 不能用 **kwargs），所以每个工具要有一层具体签名的薄函数；以及对外参数名到
# Agent Lab 形参的改名转发。参数越界不再在本层重复夹取：Registry 记录的是
# Agent Lab 接口的边界，最终统一由 AgentToolbox 实现内的夹取逻辑兜底。
# ---------------------------------------------------------------------------

# Registry 形参名 → 对外参数名（对外沿用 MCP PoC 发布时的约定，客户端配置已固化）
_MCP_PARAM_ALIASES = {"search_novels": {"query": "question"}}


def list_books() -> ToolResultV1:
    return _to_payload(_toolbox_lazy(with_search=False).list_books())


def search_novels(question: str, limit: int = 5) -> ToolResultV1:
    # 对外参数名是 question；AgentToolbox.search_novels 的形参是 query
    result = _toolbox_lazy(with_search=True).search_novels(query=question, limit=limit)
    return _to_payload(result)


def read_neighbors(novel: str, chunk_id: int, radius: int = 1) -> ToolResultV1:
    result = _toolbox_lazy().read_neighbors(novel=novel, chunk_id=chunk_id, radius=radius)
    return _to_payload(result)


def get_chapter(novel: str, chapter_title: str, limit: int = 8) -> ToolResultV1:
    result = _toolbox_lazy().get_chapter(novel=novel, chapter_title=chapter_title, limit=limit)
    return _to_payload(result)


_TOOLBOX_CALLS = {
    "list_books": list_books,
    "search_novels": search_novels,
    "read_neighbors": read_neighbors,
    "get_chapter": get_chapter,
}

for _name, _fn in sorted(_TOOLBOX_CALLS.items()):
    # answer_with_citations 有意不注册：它需要 LLM 回答器，属于 Agent 循环的
    # 收尾动作，不在只读数据查询面上（spec 仍收录在 Registry 里供 Gateway 用）
    spec = get_tool_spec(_name)
    server.tool(name=spec.name, description=spec.description, structured_output=True)(_fn)


def main() -> None:
    import asyncio

    print(f"novel-rag MCP PoC 启动（db={DATABASE_URL.split('@')[-1]}）", file=sys.stderr)
    # run_stdio_async 返回协程，必须显式驱动（踩过：直接调用只会产生
    # "coroutine was never awaited"，服务器静默退出、客户端报 Connection closed）
    asyncio.run(server.run_stdio_async())


if __name__ == "__main__":
    main()

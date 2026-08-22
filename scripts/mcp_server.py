#!/usr/bin/env python3
"""novel-rag 的只读 MCP 服务器（PoC，stdio 传输）。

定位（路线图 M6.6 允许的提前项）
--------------------------------
把 Agent Lab 已有的只读工具用 MCP 协议暴露给任意兼容客户端（Claude Code、
Cursor 等）。**只是探针**：不接权限体系、不做发现机制，目的是用真实客户端
检验 ToolResult 的 schema 设计，反哺 M6.1 的 ToolSpec 正式化。

安全边界
--------
- 只读：仅暴露查询类工具，没有任何写操作；
- 版权红线：sources 里每条摘录截断到 80 字 + 定位信息（书/章/片段号），
  与 tests/run_qa_tests.py 的落盘策略一致——绝不整段输出原文。

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
from pydantic import BaseModel  # noqa: E402


class SourceRef(BaseModel):
    """定位信息 + 80 字摘录（版权红线见模块 docstring）。"""

    novel: str
    chapter: str
    chunk_id: int
    excerpt: str


class ToolPayload(BaseModel):
    """所有工具的统一返回形状——ToolResult 的 MCP 投影。"""

    schema_version: str
    summary: str
    facts: dict
    sources: list[SourceRef]


from agent_lab import AgentToolbox  # noqa: E402
from config import DATABASE_URL  # noqa: E402

SCHEMA_VERSION = "1"

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


def _to_payload(result):
    """ToolResult → MCP 结构化返回：摘要 + 定位化 sources + 可校验 facts。"""
    return ToolPayload(
        schema_version=SCHEMA_VERSION,
        summary=result.summary,
        facts=dict(result.facts),
        sources=[
            SourceRef(
                novel=s.novel,
                chapter=s.chapter_title or "",
                chunk_id=s.chunk_id,
                # 版权红线：摘录 ≤80 字，完整原文请按定位信息自行查库
                excerpt=(s.text or "")[:80],
            )
            for s in result.sources
        ],
    )


@server.tool(
    name="list_books",
    description="列出书架上全部小说及各自片段数",
    structured_output=True,
)
def list_books() -> ToolPayload:
    return _to_payload(_toolbox_lazy(with_search=False).list_books())


@server.tool(
    name="search_novels",
    description="混合检索小说原文，返回最相关的片段（含 80 字摘录与定位信息）",
    structured_output=True,
)
def search_novels(question: str, limit: int = 5) -> ToolPayload:
    # 对外参数名用 question（对客户端更自然）；AgentToolbox 的形参是 query
    result = _toolbox_lazy(with_search=True).search_novels(
        query=question, limit=max(1, min(limit, 10))
    )
    return _to_payload(result)


@server.tool(
    name="read_neighbors",
    description="按片段编号读取前后相邻片段，用于核对上下文",
    structured_output=True,
)
def read_neighbors(novel: str, chunk_id: int, radius: int = 1) -> ToolPayload:
    result = _toolbox_lazy().read_neighbors(
        novel=novel, chunk_id=chunk_id, radius=max(1, min(radius, 3))
    )
    return _to_payload(result)


@server.tool(
    name="get_chapter",
    description="按章节标题取某书的章节内容片段列表",
    structured_output=True,
)
def get_chapter(novel: str, chapter_title: str, limit: int = 8) -> ToolPayload:
    result = _toolbox_lazy().get_chapter(
        novel=novel, chapter_title=chapter_title, limit=min(limit, 20)
    )
    return _to_payload(result)


def main() -> None:
    import asyncio

    print(f"novel-rag MCP PoC 启动（db={DATABASE_URL.split('@')[-1]}）", file=sys.stderr)
    # run_stdio_async 返回协程，必须显式驱动（踩过：直接调用只会产生
    # "coroutine was never awaited"，服务器静默退出、客户端报 Connection closed）
    asyncio.run(server.run_stdio_async())


if __name__ == "__main__":
    main()

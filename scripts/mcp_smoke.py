#!/usr/bin/env python3
"""MCP PoC 冒烟验证：以真实客户端身份经 stdio 拉起服务器，列工具并调用一次。

用法：uv run python scripts/mcp_smoke.py
需要本机 PostgreSQL 可连；search 类调用还需要 embedding 模型缓存。
"""

import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PARAMS = StdioServerParameters(
    command="uv",
    args=["run", "python", "scripts/mcp_server.py"],
    cwd=str(__import__("pathlib").Path(__file__).resolve().parent.parent),
)


async def main() -> None:
    async with (
        stdio_client(PARAMS) as (read, write),
        ClientSession(read, write) as session,
    ):
            await session.initialize()
            tools = await session.list_tools()
            print("tools:", [t.name for t in tools.tools])

            result = await session.call_tool("list_books", {})
            payload = result.structured_content or {}
            # list_books 的书目清单放在 facts.books 里（AgentToolbox 的既有约定）
            books = (payload.get("facts") or {}).get("books", [])
            print("summary:", str(payload.get("summary"))[:60])
            print("list_books →", len(books), "本书")
            assert isinstance(books, list) and books, "list_books 应返回非空书架"

            first = books[0]
            print("第一本:", str(first)[:100])

            # 检索工具会现场加载 embedding 模型（首次数秒），验证摘录红线与结构化输出
            s = await session.call_tool(
                "search_novels", {"question": "韩立的绰号", "limit": 3}
            )
            sp = s.structured_content or {}
            srcs = sp.get("sources") or []
            print(
                "search_novels →",
                len(srcs),
                "条来源；摘录最长",
                max((len(x["excerpt"]) for x in srcs), default=0),
                "字",
            )
            assert all(len(x["excerpt"]) <= 80 for x in srcs), "摘录不得超过 80 字"
            assert len(srcs) > 0


if __name__ == "__main__":
    asyncio.run(main())

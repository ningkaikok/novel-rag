"""MCP PoC 服务器的单元测试：不连数据库，只测纯逻辑。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import mcp_server

import tool_spec


def test_four_readonly_tools_registered():
    import asyncio

    tools = asyncio.run(mcp_server.server.list_tools())
    assert {t.name for t in tools} == {
        "list_books",
        "search_novels",
        "read_neighbors",
        "get_chapter",
    }


def test_payload_truncates_excerpt_and_marks_schema_version():
    class _Chunk:
        novel = "书"
        chapter_title = "第一章"
        chunk_id = 7
        text = "长" * 200

    class _Result:
        summary = "摘要"
        facts = {"kind": "x"}
        sources = [_Chunk()]

    payload = mcp_server._to_payload(_Result())
    # 版本常量已迁到 tool_spec（M6.1 前置项），这里验证投影仍带版本号
    assert payload.schema_version == tool_spec.TOOL_RESULT_SCHEMA_VERSION == "1"
    assert len(payload.sources[0].excerpt) == 80  # 版权红线：摘录截断到 80 字
    assert payload.facts == {"kind": "x"}


def test_toolbox_lazy_defers_rag_loading():
    """list 类工具不得触发 embedding 模型加载（首次载入数秒）。"""
    tb = mcp_server._toolbox_lazy(with_search=False)
    assert tb.rag is None

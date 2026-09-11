import pytest

from agent_lab import ToolResult
from tool_gateway import ToolGateway, ToolGatewayError, validate_tool_args
from tool_spec import get_tool_spec


class _Toolbox:
    def __init__(self):
        self.calls = []

    def execute(self, name, args):
        self.calls.append((name, args))
        return ToolResult("ok")


def test_gateway_validates_schema_permissions_and_duplicate_calls():
    toolbox = _Toolbox()
    gateway = ToolGateway(toolbox, max_calls=3)

    gateway.execute("search_novels", {"query": "庄主是谁", "limit": 1})
    assert toolbox.calls == [("search_novels", {"query": "庄主是谁", "limit": 1})]

    with pytest.raises(ToolGatewayError, match="重复"):
        gateway.execute("search_novels", {"query": "庄主是谁", "limit": 1})
    with pytest.raises(ToolGatewayError, match="未知参数"):
        gateway.execute("search_novels", {"query": "庄主是谁", "extra": 1})
    with pytest.raises(ToolGatewayError, match="权限"):
        ToolGateway(_Toolbox(), permissions=set()).execute("search_novels", {"query": "x"})


def test_gateway_rejects_unknown_tools_and_invalid_types():
    with pytest.raises(ToolGatewayError) as unknown:
        ToolGateway(_Toolbox()).execute("shell", {})
    assert unknown.value.category == "unknown_tool"

    with pytest.raises(ToolGatewayError) as invalid:
        validate_tool_args(get_tool_spec("read_neighbors"), {"novel": "书", "chunk_id": "1"})
    assert invalid.value.category == "invalid_args"


def test_gateway_snapshot_contains_only_execution_metadata():
    gateway = ToolGateway(_Toolbox())
    gateway.execute("list_books", {})
    snapshot = gateway.snapshot()
    assert snapshot["calls"] == 1
    assert snapshot["audits"][0]["tool"] == "list_books"
    assert "query" not in snapshot["audits"][0]

"""RequestIDMiddleware 的单元测试：不依赖完整 FastAPI app，
直接拿一个假的下游 ASGI app 验证中间件本身的行为。

用 asyncio.run() 而不是引入 pytest-asyncio 依赖——这几个测试足够简单，
不需要额外的插件。
"""

import asyncio

from backend.context import request_id_var
from backend.middleware import RequestIDMiddleware


def _run(coro):
    return asyncio.run(coro)


def test_generates_request_id_when_missing():
    """没带 X-Request-ID 请求头：中间件自己生成一个，且下游 app 能通过
    contextvar 读到同一个值（日志 Filter 就是这么拿到它的）。
    """
    seen = {}

    async def downstream(scope, receive, send):
        seen["id"] = request_id_var.get()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    middleware = RequestIDMiddleware(downstream)
    sent = []

    async def send(message):
        sent.append(message)

    async def receive():
        return {"type": "http.request"}

    scope = {"type": "http", "method": "GET", "path": "/x", "headers": []}
    _run(middleware(scope, receive, send))

    assert seen["id"] and seen["id"] != "-"
    headers = dict(sent[0]["headers"])
    assert headers[b"x-request-id"].decode("latin-1") == seen["id"]


def test_honors_incoming_request_id_header():
    """客户端/上游网关自己传了 X-Request-ID：原样沿用，不生成新的——
    这样跨服务调用时同一条链路能用同一个 id 串起来。
    """

    async def downstream(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})

    middleware = RequestIDMiddleware(downstream)
    sent = []

    async def send(message):
        sent.append(message)

    async def receive():
        return {"type": "http.request"}

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/x",
        "headers": [(b"x-request-id", b"client-supplied-id")],
    }
    _run(middleware(scope, receive, send))

    headers = dict(sent[0]["headers"])
    assert headers[b"x-request-id"] == b"client-supplied-id"


def test_non_http_scope_passes_through_untouched():
    """lifespan 这类非 HTTP 的 scope：直接透传给下游，不做任何处理。"""
    called = {}

    async def downstream(scope, receive, send):
        called["yes"] = True

    middleware = RequestIDMiddleware(downstream)

    async def receive():
        return {}

    async def send(message):
        pass

    _run(middleware({"type": "lifespan"}, receive, send))
    assert called.get("yes") is True


def test_context_var_reset_after_request():
    """请求处理完之后，contextvar 应该恢复默认值——不会把这次的 id
    漏到下一次请求（或下一个测试）里。
    """

    async def downstream(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})

    middleware = RequestIDMiddleware(downstream)

    async def receive():
        return {"type": "http.request"}

    async def send(message):
        pass

    scope = {"type": "http", "method": "GET", "path": "/x", "headers": []}
    _run(middleware(scope, receive, send))

    assert request_id_var.get() == "-"

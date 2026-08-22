"""ASGI 中间件：给每个 HTTP 请求生成/透传一个 request-id，并打一条访问日志。

request-id 存进 contextvars，配合 logging_config.py 的 Filter 自动注入到这次
请求产生的每一条日志里——SSE 流式接口出问题时，能把"这次请求从进来到结束"的
日志串起来看，而不是只能靠时间戳肉眼猜。

刻意写成原始 ASGI 中间件（不是 Starlette 的 BaseHTTPMiddleware）：
BaseHTTPMiddleware 会把整个响应体收集完再转发给客户端，对 /api/ask 这种
SSE 流式接口是致命的——用户要等生成完全结束才能收到第一个字，打字机效果
和"点停止立刻收到已生成内容"都会失效。原始 ASGI 中间件只在
http.response.start 这一条消息上追加响应头，其余消息（包括每一个
http.response.body chunk）原样透传，不缓冲、不等待。
"""

import time
import uuid

from backend.context import request_id_var
from backend.logging_config import get_logger

logger = get_logger("access")


class RequestIDMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        # 只拦截 HTTP；lifespan、websocket 等 scope 没有"请求"概念，原样放行，
        # 避免 set/reset contextvar 用在错误的协议生命周期上。
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        incoming = headers.get(b"x-request-id")
        # 客户端/网关传了就沿用（便于跨服务串联），没传就自己生成一个短 id
        request_id = incoming.decode("latin-1") if incoming else uuid.uuid4().hex[:12]
        token = request_id_var.set(request_id)

        method = scope.get("method", "")
        path = scope.get("path", "")
        start = time.monotonic()
        status_holder: dict = {}

        # 用 dict 包一层 send：只在响应头消息上追加 x-request-id，body chunk
        # 原样透传（这是不缓冲 SSE 的关键，见模块 docstring）。status 不能直接
        # 从 send 拿到返回值，所以塞进 holder，供 finally 里的访问日志使用。
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_holder["status"] = message["status"]
                message.setdefault("headers", [])
                message["headers"].append((b"x-request-id", request_id.encode("latin-1")))
            await send(message)

        logger.info(f"{method} {path} 开始")
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            # 放在 finally：请求抛异常时也能记下耗时和状态，不会丢访问日志。
            # status_holder 里没有值说明响应头都没发出去就炸了，记 "?" 而不是猜。
            duration_ms = (time.monotonic() - start) * 1000
            status = status_holder.get("status", "?")
            logger.info(f"{method} {path} 结束 status={status} 耗时={duration_ms:.0f}ms")
            request_id_var.reset(token)

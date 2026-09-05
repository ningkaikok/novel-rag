"""统一的错误响应 schema：`{"error": {"code": ..., "message": ...}}`。

之前所有 `raise HTTPException(...)` 用的都是自由文本 detail，前端只能拿到
一整段文案，没法按错误类型做不同的 UI 反应（比如"索引未建立"和"模型不可用"
现在前端处理方式完全一样，只是弹出的文字不同）。

这里定义一个带 code 的 APIError，配合全局 exception handler——同时把
FastAPI 自己的 RequestValidationError（请求体不符合 Pydantic 模型时框架
自动抛出的）以及任何未被专门处理的异常，都归一化成同一个 envelope。
调用方（前端）不用区分"这是我们自己抛的错误"还是"框架校验失败"还是
"服务器炸了"，处理逻辑统一只读 `error.message`（想要更精细的判断可以读
`error.code`）。
"""

from enum import StrEnum

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from backend.logging_config import get_logger

logger = get_logger("errors")


class ErrorCode(StrEnum):
    """机器可读的错误码，是前后端之间的稳定契约。

    用 StrEnum：成员本身就是字符串，JSON 序列化不需要额外转换；前端拿 code
    做分支判断（如 index_task_running → 轮询任务卡片），拿 message 做展示。
    新增错误码对老前端是无损的（多一个没见过的值，退化为只显示 message），
    但**修改或删除已有 code 是破坏性变更**。
    """

    no_valid_files = "no_valid_files"
    file_too_large = "file_too_large"
    book_not_found = "book_not_found"
    index_not_ready = "index_not_ready"
    index_task_running = "index_task_running"
    index_task_not_found = "index_task_not_found"
    index_task_not_retryable = "index_task_not_retryable"
    model_unavailable = "model_unavailable"
    session_read_failed = "session_read_failed"
    session_clear_failed = "session_clear_failed"
    validation_error = "validation_error"
    internal_error = "internal_error"


class APIError(Exception):
    """业务代码里主动抛的错误。取代原来的 HTTPException，携带机器可读的 code。"""

    def __init__(self, status_code: int, code: ErrorCode, message: str):
        self.status_code = status_code
        self.code = code
        self.message = message
        super().__init__(message)


def _envelope(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


def register_exception_handlers(app: FastAPI) -> None:
    """在 app 上挂三个 handler，覆盖"我们自己抛的错误 / 框架校验失败 / 未预料的异常"
    这三种情况，保证不管哪种，调用方看到的响应体都是同一个形状。
    """

    @app.exception_handler(APIError)
    async def _handle_api_error(request: Request, exc: APIError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code, content=_envelope(exc.code, exc.message)
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        errors = exc.errors()
        # message 给一句人话摘要；details 保留 Pydantic 原始的逐字段错误，
        # 方便开发时调试具体是哪个字段、哪种校验没过。
        message = errors[0]["msg"] if errors else "请求参数不合法"
        content = _envelope(ErrorCode.validation_error, message)
        content["error"]["details"] = errors
        return JSONResponse(status_code=422, content=content)

    @app.exception_handler(Exception)
    async def _handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        # 兜底：任何没被专门处理的异常都不该把内部堆栈细节泄漏给调用方
        # （比如数据库连接字符串、文件路径这类信息），但要在服务端把完整
        # 异常记下来（logger.exception 会带上 traceback），方便排查。
        logger.exception(f"未处理的异常：{request.method} {request.url.path}")
        return JSONResponse(
            status_code=500,
            content=_envelope(ErrorCode.internal_error, "服务器内部错误"),
        )

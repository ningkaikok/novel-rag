"""集中配置 logging：格式、级别，以及把当前请求的 request-id 自动注入
每一条日志（配合 backend/middleware.py 的 RequestIDMiddleware）。

用独立的 "novel_rag" 具名 logger、且 propagate=False，不 touch root logger
——uvicorn 自己的 uvicorn.error/uvicorn.access 走它自己的 handler，
这里的配置不会和它互相干扰或导致日志打印两遍。
"""

import logging

from backend.context import request_id_var
from config import LOG_LEVEL


class RequestIDFilter(logging.Filter):
    """把 contextvar 里的 request-id 抄写到每条 LogRecord 上。

    Filter 是唯一能"在格式化前"往 record 塞自定义字段的钩子；Formatter 里
    的 %(request_id)s 依赖它。contextvar 天然按异步任务隔离，不会串号。
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # 默认值 "-"：启动/关闭、后台索引线程这些不在 HTTP 请求里的日志
        # 也能正常格式化（占位符而不是 KeyError），且一眼能看出"不属于任何请求"。
        record.request_id = request_id_var.get()
        return True


def configure_logging() -> None:
    # 整体覆盖式配置：直接替换 handlers 列表，重复调用也幂等（main.py 启动时
    # 调一次，测试里可能反复调），不会累积出多个 handler 导致日志打两遍。
    level = LOG_LEVEL.upper()
    handler = logging.StreamHandler()
    handler.addFilter(RequestIDFilter())
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s [%(request_id)s] %(name)s: %(message)s")
    )
    logger = logging.getLogger("novel_rag")
    logger.setLevel(level)
    logger.handlers = [handler]
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """在 "novel_rag" 这个具名 logger 下建子 logger，共享上面配置的 handler/级别。"""
    return logging.getLogger(f"novel_rag.{name}")

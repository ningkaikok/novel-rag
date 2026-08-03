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
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def configure_logging() -> None:
    level = LOG_LEVEL.upper()
    handler = logging.StreamHandler()
    handler.addFilter(RequestIDFilter())
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-8s [%(request_id)s] %(name)s: %(message)s"
        )
    )
    logger = logging.getLogger("novel_rag")
    logger.setLevel(level)
    logger.handlers = [handler]
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """在 "novel_rag" 这个具名 logger 下建子 logger，共享上面配置的 handler/级别。"""
    return logging.getLogger(f"novel_rag.{name}")

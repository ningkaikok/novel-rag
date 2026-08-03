"""跨请求共享的 contextvars。单独成一个模块，避免 middleware.py 和
logging_config.py 互相 import 造成循环依赖（middleware 要读/写它，
logging_config 的 Filter 要读它，两边都需要，谁都不该"拥有"对方）。
"""
from contextvars import ContextVar

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

"""跨请求共享的 contextvars。单独成一个模块，避免 middleware.py 和
logging_config.py 互相 import 造成循环依赖（middleware 要读/写它，
logging_config 的 Filter 要读它，两边都需要，谁都不该"拥有"对方）。
"""
from contextvars import ContextVar

# default="-"：请求生命周期之外的日志（启动/关闭、后台索引线程）拿不到
# request-id 时用占位符，保证 logging_config 的 Formatter 永远有值可填，
# 且 "-" 一眼能区分"不在任何请求里"。ContextVar 在 asyncio 下按任务隔离，
# 并发请求之间不会互相覆盖。
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

"""跨请求共享的 contextvars。单独成一个模块，避免 middleware.py 和
logging_config.py 互相 import 造成循环依赖（middleware 要读/写它，
logging_config 的 Filter 要读它，两边都需要，谁都不该"拥有"对方）。

三类 ID 的职责边界（路线图 M3.5-③）
------------------------------------
**request_id**（本模块）：一次 HTTP 请求的唯一标识。middleware 为每个进入的
请求生成，contextvar 透传给日志 Filter，让同一请求里所有日志行都带同一个前缀；
SSE 响应结束后即作废，不与任何持久状态绑定。

**run_id**：一次"逻辑运行"的串联标识——跨越多步/多次调用的单元。典型场景：
Agent 的多步工具循环、后台索引任务、未来的重试场景（重试会换 request_id，
但应共享 run_id 才能把几次尝试串成一条线）。本项目目前没有独立的 run_id
体系，职责被部分承担着：
    - 索引任务：``index_task_runs.id``（uuid）就是一次后台运行的 run_id，
      任务卡片、落库快照和重试链（retry_of）都用它串联；
    - Agent Lab：会话层用 ``session_id + turn_index`` 定位一轮对话；M3.5 起
      每个 agent_step 事件带上了轻量 ``run_id`` 字段，同一次运行的所有步骤
      共享一个值（见 backend/main.py 的 /api/agent/ask）。
等出现真正的跨进程重试/恢复需求时，再把这两个概念统一成显式的 run_id。

**trace_id**：跨服务分布式观测标识（OpenTelemetry 语境）。本项目当前是单进程
单体，没有第二个服务可串联，**刻意预留、暂不引入**——提前引入只会得到一个
永远等于 request_id 的冗余字段。
"""
from contextvars import ContextVar

# default="-"：请求生命周期之外的日志（启动/关闭、后台索引线程）拿不到
# request-id 时用占位符，保证 logging_config 的 Formatter 永远有值可填，
# 且 "-" 一眼能区分"不在任何请求里"。ContextVar 在 asyncio 下按任务隔离，
# 并发请求之间不会互相覆盖。
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

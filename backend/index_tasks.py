"""单进程后台索引任务管理器。

这是本地单用户项目，不需要 Celery/Redis；一个受锁保护的后台线程就足够。真正的
一致性由 ``src/postgres.py`` 的单书事务保证，任务管理器只负责生命周期、进度、
取消信号和重试所需的状态。服务重启时正在进行的数据库事务会回滚，下一次按
manifest 扫描即可继续未完成的书。

任务卡片状态通过可选的 ``store`` 落库（M3.3.5）：内存仍是运行时的唯一事实来源，
数据库只是"最后已知状态"的副本——重启后 ``current()/get()`` 会回退读库，让侧栏
还能看到上一次任务的卡片；遗留的 active 状态由启动时的 ``mark_interrupted()``
如实改成 failed。落库失败只打警告、绝不影响任务本身（数据正确性不依赖它）。

任务状态机（状态变化比线程实现本身更值得先看）：

    queued ──线程启动──→ running ──成功──→ completed
                            │  │
                     用户取消  └─异常──→ failed ──重试──→ 新任务
                            ↓
                       cancelling ──到达检查点──→ cancelled

``cancelling`` 不是取消失败，而是“信号已经收到，后台正走向安全检查点”。Python
无法安全地强杀一个正在执行模型推理或数据库 COPY 的线程，因此这里使用协作式取消：
任务定期调用 ``check_cancel``，发现 Event 后主动抛出 ``IndexCancelled``，数据库层
再通过事务回滚保证一致性。这也是很多 Agent 长任务实现取消/暂停时的基本模式。
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from backend.logging_config import get_logger
from ingest import IndexCancelled

logger = get_logger("index_tasks")

# 状态集合是各处判断的单一事实来源：并发闸门（同一时间只允许一个任务）看
# ACTIVE_STATUSES；进度回调是否还接受写入、重试入口是否放行，都看终态集合。
ACTIVE_STATUSES = {"queued", "running", "cancelling"}
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}

ProgressFn = Callable[[str, int, str], None]
CancelCheck = Callable[[], None]
BuildFn = Callable[[ProgressFn, CancelCheck], dict]


class TaskAlreadyRunning(RuntimeError):
    def __init__(self, task: dict[str, Any]):
        self.task = task
        super().__init__("已有索引任务正在运行")


class TaskNotFound(KeyError):
    pass


class TaskStore(Protocol):
    """任务快照的持久化接口；实现见文件末尾的 PostgresTaskStore。

    manager 只依赖这三个方法，测试里用一个内存字典就能伪造数据库——这也是
    Protocol（结构化类型）比继承更合适的原因：不需要 import 具体实现。
    """

    def save(self, snapshot: dict[str, Any]) -> None: ...

    def load_latest(self) -> dict[str, Any] | None: ...

    def load(self, task_id: str) -> dict[str, Any] | None: ...

    def mark_interrupted(self) -> int: ...


# 进度类更新的落库节流：阶段切换立即写，纯进度百分比最多每 2 秒写一次。
# 索引一本书会产生成百上千次 update 回调，每次都 UPSERT 数据库纯属浪费——
# 卡片显示只需要"最新进度"，不需要每一次中间值。
_PERSIST_MIN_INTERVAL = 2.0


class PostgresTaskStore:
    """TaskStore 的 PostgreSQL 实现，函数体只是对 postgres 模块的薄转发。

    postgres 的 import 放在方法内部：让本模块在没有数据库驱动的环境里也能
    被安全 import（单元测试不碰真库）。
    """

    def save(self, snapshot: dict[str, Any]) -> None:
        from postgres import save_index_task_run

        save_index_task_run(snapshot)

    def load_latest(self) -> dict[str, Any] | None:
        from postgres import load_latest_index_task_runs

        rows = load_latest_index_task_runs(limit=1)
        return rows[0] if rows else None

    def load(self, task_id: str) -> dict[str, Any] | None:
        from postgres import load_index_task_run

        return load_index_task_run(task_id)

    def mark_interrupted(self) -> int:
        from postgres import mark_interrupted_index_task_runs

        return mark_interrupted_index_task_runs()


@dataclass
class _Task:
    """任务的可变内部状态；API 调用方只能拿到 ``snapshot()`` 生成的普通字典。

    如果直接把这个对象交给 Web 层，序列化过程中后台线程可能同时改字段，前端就
    可能看到“状态已完成但 progress 还是 70”这类撕裂快照。
    """

    id: str
    force: bool
    retry_of: str | None = None
    status: str = "queued"
    stage: str = "queued"
    progress: int = 0
    message: str = "任务已排队"
    error: str | None = None
    result: dict[str, Any] | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    started_at: str | None = None
    finished_at: str | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    # 落库节流用的字段（不属于 API 快照）：上次写库的时间和阶段。
    _persisted_at: float = field(default=0.0, repr=False)
    _persisted_stage: str = ""

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "stage": self.stage,
            "progress": self.progress,
            "message": self.message,
            "error": self.error,
            "force": self.force,
            "retry_of": self.retry_of,
            "result": self.result,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class IndexTaskManager:
    """最多运行一个索引任务；所有公开返回值都是不可变快照。"""

    def __init__(self, store: TaskStore | None = None) -> None:
        self._lock = threading.RLock()
        # 可选的快照持久化（M3.3.5）：内存仍是运行时唯一事实来源，store 只保存
        # "最后已知状态"供重启恢复。默认 None（纯内存），由 FastAPI lifespan 在
        # 数据库就绪后调用 set_store() 接上。
        self._store = store
        self._tasks: dict[str, _Task] = {}
        # 单槽位设计：UI 只关心"最近一次任务"，轮询 /api/index-tasks/current 即可，
        # 不需要维护任务队列（本项目管理器最多同时跑一个）。
        self._latest_id: str | None = None
        self._threads: dict[str, threading.Thread] = {}

    def set_store(self, store: TaskStore) -> int:
        """启动阶段接上持久化，并把上次进程遗留的 active 任务标记为中断。

        必须在数据库 schema 就绪后调用。返回被标记的任务数（写进启动日志，
        让"为什么侧栏有个 failed 的卡片"有迹可循）。
        """
        with self._lock:
            self._store = store
        try:
            return store.mark_interrupted()
        except Exception:
            logger.warning("标记遗留索引任务失败（任务卡片可能显示旧状态）", exc_info=True)
            return 0

    def _persist(self, task: _Task) -> None:
        """把任务当前状态写入 store；任何失败都不影响任务运行。

        调用方必须已持有 self._lock。节流规则见 _PERSIST_MIN_INTERVAL：终态和
        阶段切换总是落库，纯进度更新最多每 2 秒一次。
        """
        if self._store is None:
            return
        terminal_or_stage = (
            task.status in TERMINAL_STATUSES or task.stage != task._persisted_stage
        )
        if (
            not terminal_or_stage
            and time.monotonic() - task._persisted_at < _PERSIST_MIN_INTERVAL
        ):
            return
        try:
            self._store.save(task.snapshot())
            task._persisted_at = time.monotonic()
            task._persisted_stage = task.stage
        except Exception:
            logger.warning("索引任务 %s 状态落库失败", task.id, exc_info=True)

    def current(self) -> dict[str, Any] | None:
        with self._lock:
            if self._latest_id is not None:
                return self._tasks[self._latest_id].snapshot()
            store = self._store
        # 进程刚重启、内存还没有任何任务：回退读库里最后一条已知状态，让侧栏
        # 不至于凭空丢掉上一张卡片。读不到（首次使用/没配数据库）就保持 None。
        if store is not None:
            try:
                return store.load_latest()
            except Exception:
                logger.warning("读取历史索引任务失败", exc_info=True)
        return None

    def get(self, task_id: str) -> dict[str, Any]:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is not None:
                return task.snapshot()
            store = self._store
        if store is not None:
            try:
                snapshot = store.load(task_id)
            except Exception:
                logger.warning("读取索引任务 %s 失败", task_id, exc_info=True)
                snapshot = None
            if snapshot is not None:
                return snapshot
        raise TaskNotFound(task_id)

    def active(self) -> dict[str, Any] | None:
        with self._lock:
            if self._latest_id is None:
                return None
            task = self._tasks[self._latest_id]
            return task.snapshot() if task.status in ACTIVE_STATUSES else None

    def start(
        self,
        build: BuildFn,
        *,
        force: bool = False,
        retry_of: str | None = None,
        prepare: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            active = self.active()
            if active is not None:
                raise TaskAlreadyRunning(active)
            # 上传落盘/删除文件和“占有任务槽”必须共享这把锁，否则两个并发请求
            # 可能都先改了文件，只有后一个成功启动任务，前一个响应却报冲突。
            if prepare is not None:
                prepare()
            task = _Task(id=str(uuid.uuid4()), force=force, retry_of=retry_of)
            self._tasks[task.id] = task
            self._latest_id = task.id
            self._persist(task)
            thread = threading.Thread(
                target=self._run,
                args=(task.id, build),
                name=f"novel-index-{task.id[:8]}",
                # daemon 线程不会阻止 Python 进程退出；真正的优雅停止仍由 shutdown()
                # 发取消信号并等待。daemon 只是最后一道兜底，不能代替事务回滚。
                daemon=True,
            )
            self._threads[task.id] = thread
            thread.start()
            return task.snapshot()

    def _run(self, task_id: str, build: BuildFn) -> None:
        with self._lock:
            task = self._tasks[task_id]
            task.status = "running"
            task.stage = "scan"
            task.message = "正在扫描小说目录"
            task.started_at = datetime.now(UTC).isoformat()
            self._persist(task)

        def update(stage: str, progress: int, message: str) -> None:
            with self._lock:
                current = self._tasks[task_id]
                # 终态之后不再接受写入，防止线程收尾阶段的迟到回调覆盖掉
                # cancelled/failed 等结果；progress 只增不减，避免阶段切换时
                # 回调乱序导致进度条倒退。
                if current.status not in TERMINAL_STATUSES:
                    current.stage = stage
                    current.progress = max(current.progress, min(100, int(progress)))
                    current.message = message
                    self._persist(current)

        def check_cancel() -> None:
            # Event 是跨线程传递“请停止”的信号，不承载业务状态。具体在哪些循环中
            # 检查由 ingest.build_index 决定，因此取消延迟取决于检查点的密度。
            if self._tasks[task_id].cancel_event.is_set():
                raise IndexCancelled("用户取消了索引任务")

        try:
            result = build(update, check_cancel)
        except IndexCancelled:
            with self._lock:
                task = self._tasks[task_id]
                task.status = "cancelled"
                task.stage = "cancelled"
                task.message = "任务已取消；已完成的书保持可用，当前书没有写入半套索引"
                task.finished_at = datetime.now(UTC).isoformat()
                self._persist(task)
        except Exception as exc:
            logger.exception("索引任务 %s 失败", task_id)
            with self._lock:
                task = self._tasks[task_id]
                task.status = "failed"
                task.stage = "failed"
                task.message = "索引任务失败，可安全重试"
                task.error = str(exc) or exc.__class__.__name__
                task.finished_at = datetime.now(UTC).isoformat()
                self._persist(task)
        else:
            with self._lock:
                task = self._tasks[task_id]
                task.status = "completed"
                task.stage = "complete"
                task.progress = 100
                task.message = "书架索引已更新"
                task.result = result
                task.finished_at = datetime.now(UTC).isoformat()
                self._persist(task)

    def cancel(self, task_id: str) -> dict[str, Any]:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise TaskNotFound(task_id)
            if task.status in ACTIVE_STATUSES:
                # 这里只做两件事：置位 Event（跨线程信号，见模块 docstring 的
                # 协作式取消说明）+ 在锁内把状态改成 cancelling。真正的收尾
                # （回滚、状态落定为 cancelled）由后台线程到达检查点后自己完成，
                # 本方法不等待、不代劳——HTTP 响应可以立刻返回给前端。
                task.cancel_event.set()
                task.status = "cancelling"
                task.message = "正在安全停止：当前数据库事务会回滚"
                self._persist(task)
            return task.snapshot()

    def reset_for_tests(self) -> None:
        """测试间清理终态任务；运行中的线程不能被静默遗弃。"""
        with self._lock:
            active = self.active()
            if active is not None:
                self._tasks[active["id"]].cancel_event.set()
                thread = self._threads.get(active["id"])
            else:
                thread = None
        if thread:
            thread.join(timeout=2)
        with self._lock:
            self._tasks.clear()
            self._threads.clear()
            self._latest_id = None

    def shutdown(self, timeout: float = 5) -> None:
        """服务退出时请求停止，并短暂等待任务走到安全检查点。"""
        with self._lock:
            active = self.active()
            if active is None:
                return
            task = self._tasks[active["id"]]
            task.cancel_event.set()
            task.status = "cancelling"
            task.message = "后端正在关闭，任务将在安全检查点停止"
            thread = self._threads.get(task.id)
        if thread:
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning("索引任务 %s 尚未停止，将由进程退出回滚连接", task.id)

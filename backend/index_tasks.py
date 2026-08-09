"""单进程后台索引任务管理器。

这是本地单用户项目，不需要 Celery/Redis；一个受锁保护的后台线程就足够。真正的
一致性由 ``src/postgres.py`` 的单书事务保证，任务管理器只负责生命周期、进度、
取消信号和重试所需的状态。服务重启时内存状态会消失，但正在进行的数据库事务会
回滚，下一次按 manifest 扫描即可继续未完成的书。

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
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable

from backend.logging_config import get_logger
from ingest import IndexCancelled

logger = get_logger("index_tasks")

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

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._tasks: dict[str, _Task] = {}
        self._latest_id: str | None = None
        self._threads: dict[str, threading.Thread] = {}

    def current(self) -> dict[str, Any] | None:
        with self._lock:
            if self._latest_id is None:
                return None
            return self._tasks[self._latest_id].snapshot()

    def get(self, task_id: str) -> dict[str, Any]:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise TaskNotFound(task_id)
            return task.snapshot()

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

        def update(stage: str, progress: int, message: str) -> None:
            with self._lock:
                current = self._tasks[task_id]
                if current.status not in TERMINAL_STATUSES:
                    current.stage = stage
                    current.progress = max(current.progress, min(100, int(progress)))
                    current.message = message

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
        except Exception as exc:
            logger.exception("索引任务 %s 失败", task_id)
            with self._lock:
                task = self._tasks[task_id]
                task.status = "failed"
                task.stage = "failed"
                task.message = "索引任务失败，可安全重试"
                task.error = str(exc) or exc.__class__.__name__
                task.finished_at = datetime.now(UTC).isoformat()
        else:
            with self._lock:
                task = self._tasks[task_id]
                task.status = "completed"
                task.stage = "complete"
                task.progress = 100
                task.message = "书架索引已更新"
                task.result = result
                task.finished_at = datetime.now(UTC).isoformat()

    def cancel(self, task_id: str) -> dict[str, Any]:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise TaskNotFound(task_id)
            if task.status in ACTIVE_STATUSES:
                task.cancel_event.set()
                task.status = "cancelling"
                task.message = "正在安全停止：当前数据库事务会回滚"
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

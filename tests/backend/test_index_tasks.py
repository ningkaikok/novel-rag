import threading
import time
from typing import Any

from backend.index_tasks import IndexTaskManager, TaskAlreadyRunning


class FakeStore:
    """用内存字典伪造数据库持久化，验证 manager 的落库时机与重启恢复语义。"""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}

    def save(self, snapshot: dict[str, Any]) -> None:
        self.rows[snapshot["id"]] = dict(snapshot)

    def load_latest(self) -> dict[str, Any] | None:
        if not self.rows:
            return None
        return max(self.rows.values(), key=lambda row: row["created_at"])

    def load(self, task_id: str) -> dict[str, Any] | None:
        return self.rows.get(task_id)

    def mark_interrupted(self) -> int:
        count = 0
        for row in self.rows.values():
            if row["status"] in {"queued", "running", "cancelling"}:
                row["status"] = "failed"
                count += 1
        return count


def _wait_terminal(manager: IndexTaskManager, timeout: float = 2) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        task = manager.current()
        if task and task["status"] in {"completed", "failed", "cancelled"}:
            return task
        time.sleep(0.01)
    raise AssertionError("task did not finish")


def test_task_reports_progress_and_result():
    manager = IndexTaskManager()

    def build(progress, check_cancel):
        progress("embedding", 42, "Embedding 42/100")
        check_cancel()
        return {"novels": ["书"], "chunk_count": 3}

    started = manager.start(build)
    finished = _wait_terminal(manager)

    assert started["id"] == finished["id"]
    assert finished["status"] == "completed"
    assert finished["progress"] == 100
    assert finished["result"]["chunk_count"] == 3


def test_task_can_be_safely_cancelled():
    manager = IndexTaskManager()
    entered = threading.Event()

    def build(progress, check_cancel):
        entered.set()
        while True:
            progress("embedding", 30, "正在处理")
            check_cancel()
            time.sleep(0.01)

    task = manager.start(build)
    assert entered.wait(timeout=1)
    cancelling = manager.cancel(task["id"])
    finished = _wait_terminal(manager)

    assert cancelling["status"] == "cancelling"
    assert finished["status"] == "cancelled"
    assert "没有写入半套索引" in finished["message"]


def test_failure_keeps_reason_and_allows_a_later_task():
    manager = IndexTaskManager()

    def fail(progress, check_cancel):
        raise RuntimeError("PostgreSQL 暂时不可用")

    first = manager.start(fail)
    failed = _wait_terminal(manager)
    assert failed["id"] == first["id"]
    assert failed["status"] == "failed"
    assert failed["error"] == "PostgreSQL 暂时不可用"

    second = manager.start(
        lambda progress, check_cancel: {"novels": [], "chunk_count": 0},
        retry_of=first["id"],
    )
    completed = _wait_terminal(manager)
    assert second["retry_of"] == first["id"]
    assert completed["status"] == "completed"


def test_second_active_task_is_rejected():
    manager = IndexTaskManager()
    entered = threading.Event()
    release = threading.Event()

    def blocked(progress, check_cancel):
        entered.set()
        release.wait(timeout=1)
        return {"novels": [], "chunk_count": 0}

    manager.start(blocked)
    assert entered.wait(timeout=1)
    try:
        manager.start(blocked)
    except TaskAlreadyRunning as exc:
        assert exc.task["status"] == "running"
    else:
        raise AssertionError("expected TaskAlreadyRunning")
    finally:
        release.set()
        _wait_terminal(manager)


def test_task_transitions_are_persisted():
    """任务从排队到完成的每次状态变化都应写进 store。"""
    store = FakeStore()
    manager = IndexTaskManager(store=store)

    def build(progress, check_cancel):
        progress("embedding", 40, "Embedding 40/100")
        return {"novels": ["书"], "chunk_count": 1}

    started = manager.start(build)
    finished = _wait_terminal(manager)

    saved = store.rows[finished["id"]]
    assert saved["status"] == "completed"
    assert saved["result"]["chunk_count"] == 1
    assert saved["finished_at"] is not None
    assert store.rows[started["id"]] is saved


def test_restart_restores_latest_task_from_store():
    """重启后（新 manager 实例）current()/get() 应能回退读库拿到最后状态。"""
    store = FakeStore()
    first = IndexTaskManager(store=store)
    task = first.start(lambda progress, check: {"novels": [], "chunk_count": 0})
    finished = _wait_terminal(first)

    restarted = IndexTaskManager(store=store)  # 模拟进程重启：内存全空
    assert restarted.current()["id"] == finished["id"]
    assert restarted.current()["status"] == "completed"
    assert restarted.get(task["id"])["progress"] == 100

    # 没有 store 或库里没有记录时，行为和纯内存版一致
    assert IndexTaskManager().current() is None


def test_set_store_marks_stale_active_tasks_interrupted():
    """启动接上 store 时，上次遗留的 active 任务应被如实标记为 failed。"""
    store = FakeStore()
    stale = {"id": "legacy-1", "status": "running", "created_at": "2026-01-01T00:00:00+00:00"}
    store.save(stale)

    manager = IndexTaskManager()
    interrupted = manager.set_store(store)

    assert interrupted == 1
    assert store.rows["legacy-1"]["status"] == "failed"
    # 遗留任务恢复显示后可被读取，供用户点重试
    assert manager.current()["id"] == "legacy-1"


def test_store_failure_never_breaks_tasks():
    """落库抛异常只允许记警告，任务的运行与结果必须不受影响。"""

    class BrokenStore(FakeStore):
        def save(self, snapshot):
            raise RuntimeError("数据库不可用")

    manager = IndexTaskManager(store=BrokenStore())

    def build(progress, check_cancel):
        progress("embedding", 10, "x")
        return {"novels": [], "chunk_count": 0}

    started = manager.start(build)
    done = _wait_terminal(manager)
    assert done["status"] == "completed"
    assert done["id"] == started["id"]
    # current() 在内存命中时也不受坏 store 影响；内存未命中时读库失败返回 None
    assert manager.current()["id"] == started["id"]

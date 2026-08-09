import threading
import time

from backend.index_tasks import IndexTaskManager, TaskAlreadyRunning


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

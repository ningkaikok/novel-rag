"""Authenticated internal operations, excluded from the public API contract."""

import os

from fastapi import APIRouter, HTTPException


def operations_router(*, connect, index_tasks):
    router = APIRouter(prefix="/internal", include_in_schema=False)

    def require_worker():
        if os.environ.get("PRODUCTION_WORKER") != "1":
            raise HTTPException(404)

    @router.get("/status")
    def status():
        require_worker()
        database_ok = False
        size = 0
        try:
            with connect() as conn:
                size = conn.execute(
                    "SELECT pg_database_size(current_database()) AS size"
                ).fetchone()["size"]
                database_ok = True
        except Exception:
            pass
        task = index_tasks.current()
        return {
            "database_ok": database_ok,
            "database_bytes": size,
            "index_task_failed": bool(task and task["status"] == "failed"),
            "index_task_active": bool(
                task and task["status"] in {"queued", "running", "cancelling"}
            ),
        }

    return router

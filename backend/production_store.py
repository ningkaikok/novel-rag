"""Durable, transactional quotas shared by gateway processes. Stores no content or keys."""

from __future__ import annotations

import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path


class QuotaExceeded(Exception):
    def __init__(self, code: str, status: int = 429):
        self.code = code
        self.status = status


class ProductionStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self.transaction() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS usage (
                    tenant TEXT, day TEXT, calls INTEGER NOT NULL DEFAULT 0,
                    minute INTEGER NOT NULL DEFAULT 0, requests INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (tenant, day)
                );
                CREATE TABLE IF NOT EXISTS leases (
                    id TEXT PRIMARY KEY, tenant TEXT, expires REAL, bytes INTEGER
                );
                CREATE INDEX IF NOT EXISTS leases_tenant ON leases(tenant);
                CREATE TABLE IF NOT EXISTS maintenance (tenant TEXT PRIMARY KEY);
                CREATE TABLE IF NOT EXISTS metrics (
                    tenant TEXT, name TEXT, value REAL NOT NULL DEFAULT 0,
                    PRIMARY KEY (tenant, name)
                );
                CREATE TABLE IF NOT EXISTS alerts (
                    tenant TEXT, name TEXT, status TEXT, severity TEXT, updated REAL,
                    PRIMARY KEY(tenant,name)
                );
            """)

    @contextmanager
    def transaction(self):
        db = sqlite3.connect(self.path, timeout=15, isolation_level=None)
        try:
            db.execute("PRAGMA busy_timeout=15000")
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def acquire(self, tenant, *, call: bool, upload_bytes: int, stored_bytes: int) -> str:
        now = time.time()
        day = datetime.fromtimestamp(now, UTC).date().isoformat()
        minute = int(now // 60)
        lease = uuid.uuid4().hex
        with self.transaction() as db:
            db.execute("DELETE FROM leases WHERE expires < ?", (now,))
            if db.execute("SELECT 1 FROM maintenance WHERE tenant=?", (tenant.id,)).fetchone():
                raise QuotaExceeded("maintenance", 503)
            inflight, reserved = db.execute(
                "SELECT count(*), coalesce(sum(bytes),0) FROM leases WHERE tenant=?",
                (tenant.id,),
            ).fetchone()
            if inflight >= tenant.max_concurrent:
                raise QuotaExceeded("concurrency_limit")
            if upload_bytes and stored_bytes + reserved + upload_bytes > tenant.storage_bytes:
                raise QuotaExceeded("storage_quota", 413)
            db.execute(
                "INSERT OR IGNORE INTO usage(tenant,day) VALUES (?,?)", (tenant.id, day)
            )
            calls, old_minute, requests = db.execute(
                "SELECT calls,minute,requests FROM usage WHERE tenant=? AND day=?",
                (tenant.id, day),
            ).fetchone()
            requests = requests if old_minute == minute else 0
            if requests >= tenant.requests_per_minute:
                raise QuotaExceeded("rate_limit")
            if call and calls >= tenant.calls_per_day:
                raise QuotaExceeded("daily_call_quota")
            db.execute(
                "UPDATE usage SET calls=?,minute=?,requests=? WHERE tenant=? AND day=?",
                (calls + int(call), minute, requests + 1, tenant.id, day),
            )
            db.execute(
                "INSERT INTO leases VALUES (?,?,?,?)",
                (lease, tenant.id, now + tenant.request_timeout + 30, upload_bytes),
            )
        return lease

    def release(self, lease: str):
        with self.transaction() as db:
            db.execute("DELETE FROM leases WHERE id=?", (lease,))

    def metric(self, tenant: str, name: str, amount: float = 1):
        with self.transaction() as db:
            db.execute(
                "INSERT INTO metrics VALUES (?,?,?) ON CONFLICT(tenant,name) "
                "DO UPDATE SET value=value+excluded.value",
                (tenant, name, amount),
            )

    def metrics(self):
        with self.transaction() as db:
            return db.execute(
                "SELECT tenant,name,value FROM metrics ORDER BY tenant,name"
            ).fetchall()

    def usage(self, tenant: str):
        day = datetime.now(UTC).date().isoformat()
        with self.transaction() as db:
            row = db.execute(
                "SELECT calls FROM usage WHERE tenant=? AND day=?", (tenant, day)
            ).fetchone()
            return {"day": day, "calls": row[0] if row else 0}

    def pause(self, tenant: str):
        """Commit pause first, then drain. Stays paused until explicitly resumed."""
        with self.transaction() as db:
            db.execute("INSERT OR IGNORE INTO maintenance VALUES (?)", (tenant,))
            return db.execute(
                "SELECT count(*) FROM leases WHERE tenant=? AND expires>?",
                (tenant, time.time()),
            ).fetchone()[0]

    def resume(self, tenant: str):
        with self.transaction() as db:
            db.execute("DELETE FROM maintenance WHERE tenant=?", (tenant,))

    def prune(self, days: int = 30):
        cutoff = datetime.fromtimestamp(time.time() - days * 86400, UTC).date().isoformat()
        with self.transaction() as db:
            db.execute("DELETE FROM usage WHERE day < ?", (cutoff,))
            db.execute("DELETE FROM leases WHERE expires < ?", (time.time(),))
            db.execute(
                "DELETE FROM alerts WHERE status='resolved' AND updated < ?",
                (time.time() - days * 86400,),
            )

    def paused(self, tenant: str) -> bool:
        with self.transaction() as db:
            return (
                db.execute("SELECT 1 FROM maintenance WHERE tenant=?", (tenant,)).fetchone()
                is not None
            )

    def alerts(self, tenant: str):
        with self.transaction() as db:
            return [
                {"name": name, "severity": severity, "updated_at": updated}
                for name, severity, updated in db.execute(
                    "SELECT name,severity,updated FROM alerts WHERE tenant IN (?, 'global') AND status='firing' ORDER BY name",
                    (tenant,),
                ).fetchall()
            ]

    def record_alerts(self, alerts):
        with self.transaction() as db:
            for tenant, name, status, severity in alerts:
                db.execute(
                    "INSERT INTO alerts VALUES (?,?,?,?,?) ON CONFLICT(tenant,name) DO UPDATE SET status=excluded.status,severity=excluded.severity,updated=excluded.updated",
                    (tenant, name, status, severity, time.time()),
                )

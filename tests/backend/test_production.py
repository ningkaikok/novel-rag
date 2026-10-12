"""Production boundaries using two fake workers; no database, Docker or model calls."""

import asyncio
import base64
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import replace

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.production import (
    ManagedStream,
    Tenant,
    authenticate,
    create_app,
    load_tenants,
    stored_bytes,
)
from backend.production_store import ProductionStore, QuotaExceeded
from backend.worker_security import WorkerSecurityMiddleware


class BytesStream(httpx.AsyncByteStream):
    def __init__(self, payload):
        self.payload = payload
        self.closed = False

    async def __aiter__(self):
        yield self.payload

    async def aclose(self):
        self.closed = True


@pytest.fixture
def production(tmp_path):
    tenants = []
    for name in ("alice", "bob"):
        root = tmp_path / name
        root.mkdir()
        tenants.append(
            Tenant(
                name,
                hashlib.sha256((name + "-key").encode()).hexdigest(),
                f"http://{name}",
                root,
                name * 12,
            )
        )
    store = ProductionStore(tmp_path / "control.sqlite")
    seen = []
    streams = []

    async def worker(request):
        seen.append(request)
        if request.url.path == "/internal/status":
            return httpx.Response(
                200,
                json={"database_ok": True, "database_bytes": 1234, "index_task_failed": False},
            )
        if request.url.path == "/redirect":
            return httpx.Response(307, headers={"location": "http://bob/api/books"})
        stream = BytesStream(json.dumps({"owner": request.url.host}).encode())
        streams.append(stream)
        return httpx.Response(200, stream=stream, headers={"content-type": "application/json"})

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(worker))
    app = create_app(
        tenants=tuple(tenants), store=store, client=upstream, admin_token="admin" * 12
    )
    with TestClient(app) as client:
        yield client, tenants, store, seen, streams
    asyncio.run(upstream.aclose())


def auth(name="alice"):
    return {"Authorization": "Bearer " + name + "-key"}


def test_every_resource_is_routed_by_verified_identity(production):
    client, tenants, _, seen, _ = production
    for path in (
        "/api/books",
        "/api/sessions/same-id",
        "/api/runs/same-id/events",
        "/api/graph/edges",
        "/api/index-tasks/same-id",
        "/api/knowledge/documents",
    ):
        assert client.get(path).status_code == 401
        result = client.get(
            path, headers={**auth(), "x-tenant-id": "bob", "x-worker-secret": "forged"}
        )
        assert result.json()["owner"] == "alice"
        assert seen[-1].headers["x-worker-secret"] == tenants[0].worker_secret
        assert seen[-1].headers["x-tenant-id"] == "alice"
        assert "authorization" not in seen[-1].headers
        assert client.get(path, headers=auth("bob")).json()["owner"] == "bob"


def test_browser_basic_auth_and_disabled_identity(production):
    client, tenants, _, _, _ = production
    token = base64.b64encode(b"alice:alice-key").decode()
    assert client.get("/", headers={"Authorization": "Basic " + token}).status_code == 200
    assert authenticate("Basic !!!", tuple(tenants)) is None
    assert authenticate("Bearer alice-key", (replace(tenants[0], enabled=False),)) is None
    assert (
        authenticate("Basic " + base64.b64encode(b"bob:alice-key").decode(), tuple(tenants))
        is None
    )


@pytest.mark.parametrize(
    "path",
    [
        "/internal/status",
        "/internal/retention",
        "/%252finternal/status",
        "/%2finternal/status",
    ],
)
def test_internal_routes_cannot_be_reached(production, path):
    client, _, _, seen, _ = production
    assert client.get(path, headers=auth()).status_code == 403
    assert not seen


def test_cross_site_writes_rejected(production):
    client, _, _, seen, _ = production
    assert (
        client.post(
            "/api/books", headers={**auth(), "Origin": "https://evil.example"}, content=b"file"
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/api/model", headers={**auth(), "Sec-Fetch-Site": "cross-site"}
        ).status_code
        == 403
    )
    assert not seen


def test_stream_closed_and_reservation_released(production):
    client, _, store, _, streams = production
    response = client.post("/api/ask", headers=auth(), json={"question": "private"})
    assert response.status_code == 200
    assert streams[-1].closed
    with store.transaction() as db:
        assert db.execute("SELECT count(*) FROM leases").fetchone()[0] == 0
    assert store.usage("alice")["calls"] == 1


def test_redirects_never_follow_other_worker(production):
    client, _, store, seen, _ = production
    assert client.get("/redirect", headers=auth()).status_code == 502
    assert len(seen) == 1
    assert store.pause("alice") == 0


def test_admin_metrics_and_maintenance(production):
    client, _, store, seen, _ = production
    admin = {"Authorization": "Bearer " + "admin" * 12}
    assert client.get("/metrics", headers=auth()).status_code == 401
    response = client.get("/metrics", headers=admin)
    assert 'novel_rag_database_up{tenant="alice"} 1' in response.text
    assert "alice-key" not in response.text
    assert client.post("/admin/tenants/alice/pause", headers=auth()).status_code == 401
    assert client.post("/admin/tenants/alice/pause", headers=admin).status_code == 200
    assert client.get("/api/books", headers=auth()).status_code == 503
    assert client.get("/api/books", headers=auth("bob")).status_code == 200
    assert client.post("/admin/tenants/alice/resume", headers=admin).status_code == 200
    assert client.get("/api/books", headers=auth()).status_code == 200
    assert store.usage("alice")["calls"] == 0


def test_quota_is_atomic_durable_and_tenant_local(tmp_path):
    tenant = Tenant(
        "alice",
        "a" * 64,
        "http://alice",
        tmp_path,
        "a" * 40,
        calls_per_day=1,
        max_concurrent=20,
        requests_per_minute=100,
    )
    store = ProductionStore(tmp_path / "quota.sqlite")

    def attempt(_):
        try:
            return store.acquire(tenant, call=True, upload_bytes=0, stored_bytes=0)
        except QuotaExceeded:
            return None

    with ThreadPoolExecutor(max_workers=8) as workers:
        leases = list(workers.map(attempt, range(12)))
    assert sum(lease is not None for lease in leases) == 1
    assert ProductionStore(store.path).usage("alice")["calls"] == 1
    assert store.acquire(replace(tenant, id="bob"), call=True, upload_bytes=0, stored_bytes=0)


def test_storage_reservations_prevent_concurrent_overrun(tmp_path):
    tenant = Tenant("alice", "a" * 64, "http://alice", tmp_path, "a" * 40, storage_bytes=100)
    store = ProductionStore(tmp_path / "quota.sqlite")
    lease = store.acquire(tenant, call=False, upload_bytes=60, stored_bytes=10)
    with pytest.raises(QuotaExceeded, match="storage_quota"):
        store.acquire(tenant, call=False, upload_bytes=40, stored_bytes=10)
    store.release(lease)
    assert store.acquire(tenant, call=False, upload_bytes=40, stored_bytes=10)


def test_concurrent_and_rate_limits(tmp_path):
    tenant = Tenant(
        "alice",
        "a" * 64,
        "http://alice",
        tmp_path,
        "a" * 40,
        max_concurrent=1,
        requests_per_minute=1,
    )
    store = ProductionStore(tmp_path / "quota.sqlite")
    lease = store.acquire(tenant, call=False, upload_bytes=0, stored_bytes=0)
    with pytest.raises(QuotaExceeded, match="concurrency_limit"):
        store.acquire(tenant, call=False, upload_bytes=0, stored_bytes=0)
    store.release(lease)
    with pytest.raises(QuotaExceeded, match="rate_limit"):
        store.acquire(tenant, call=False, upload_bytes=0, stored_bytes=0)


def test_expired_leases_and_pause_survive_restart(tmp_path):
    tenant = Tenant("alice", "a" * 64, "http://alice", tmp_path, "a" * 40)
    store = ProductionStore(tmp_path / "quota.sqlite")
    store.acquire(tenant, call=False, upload_bytes=0, stored_bytes=0)
    with store.transaction() as db:
        db.execute("UPDATE leases SET expires=0")
    assert store.pause("alice") == 0
    with pytest.raises(QuotaExceeded, match="maintenance"):
        ProductionStore(store.path).acquire(tenant, call=False, upload_bytes=0, stored_bytes=0)
    store.resume("alice")
    assert store.acquire(tenant, call=False, upload_bytes=0, stored_bytes=0)


def test_source_storage_count(tmp_path):
    (tmp_path / "file").write_bytes(b"abc")
    (tmp_path / ".upload").write_bytes(b"temp")
    assert stored_bytes(tmp_path) == 7


def test_config_refuses_overlapping_volumes(tmp_path, monkeypatch):
    root = tmp_path / "alice"
    root.mkdir()
    nested = root / "bob"
    nested.mkdir()
    monkeypatch.setenv("ALICE_SECRET", "a" * 40)
    monkeypatch.setenv("BOB_SECRET", "b" * 40)
    tenants = [
        {
            "id": "alice",
            "token_sha256": "a" * 64,
            "upstream": "http://alice",
            "data_dir": str(root),
            "worker_secret_env": "ALICE_SECRET",
        },
        {
            "id": "bob",
            "token_sha256": "b" * 64,
            "upstream": "http://bob",
            "data_dir": str(nested),
            "worker_secret_env": "BOB_SECRET",
        },
    ]
    config = tmp_path / "tenants.json"
    config.write_text(json.dumps({"tenants": tenants}))
    with pytest.raises(ValueError, match="overlap"):
        load_tenants(config)


def test_worker_rejects_missing_or_wrong_gateway_identity(monkeypatch):
    monkeypatch.setenv("PRODUCTION_WORKER", "1")
    monkeypatch.setenv("WORKER_SECRET", "a" * 40)
    monkeypatch.setenv("TENANT_ID", "alice")
    from fastapi import FastAPI

    worker = FastAPI()
    worker.add_middleware(WorkerSecurityMiddleware)

    @worker.get("/api/books")
    def books():
        return {"books": ["private"]}

    with TestClient(worker) as client:
        assert client.get("/api/books").status_code == 401
        assert (
            client.get(
                "/api/books", headers={"x-worker-secret": "a" * 40, "x-tenant-id": "bob"}
            ).status_code
            == 401
        )
        assert (
            client.get(
                "/api/books", headers={"x-worker-secret": "a" * 40, "x-tenant-id": "alice"}
            ).status_code
            == 200
        )


def test_worker_fails_closed_without_secret(monkeypatch):
    monkeypatch.setenv("PRODUCTION_WORKER", "1")
    monkeypatch.delenv("WORKER_SECRET", raising=False)
    with pytest.raises(ValueError):
        WorkerSecurityMiddleware(None)


def test_stream_cleanup_when_headers_cannot_be_sent():
    seen = []

    async def chunks():
        yield b"unused"

    async def cleanup():
        seen.append("released")

    async def send(_):
        raise OSError("client disconnected")

    async def receive():
        return {"type": "http.disconnect"}

    async def scenario():
        response = ManagedStream(chunks(), cleanup=cleanup)
        with suppress(Exception):
            await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)

    asyncio.run(scenario())
    assert seen == ["released"]


def test_alerts_are_authenticated_persistent_and_tenant_scoped(production):
    client, _, store, _, _ = production
    admin = {"Authorization": "Bearer " + "admin" * 12}
    payload = {
        "alerts": [
            {
                "status": "firing",
                "labels": {
                    "alertname": "KnowledgeIndexFailed",
                    "tenant": "alice",
                    "severity": "warning",
                },
                "annotations": {"private": "must not persist"},
            }
        ]
    }
    assert client.post("/admin/alerts", headers=auth(), json=payload).status_code == 401
    assert client.post("/admin/alerts", headers=admin, json=payload).json()["accepted"] == 1
    assert (
        client.get("/api/alerts", headers=auth()).json()["alerts"][0]["name"]
        == "KnowledgeIndexFailed"
    )
    assert client.get("/api/alerts", headers=auth("bob")).json()["alerts"] == []
    assert ProductionStore(store.path).alerts("alice")
    payload["alerts"][0]["status"] = "resolved"
    assert client.post("/admin/alerts", headers=admin, json=payload).status_code == 200
    assert not store.alerts("alice")


def test_encoded_question_mark_cannot_change_upstream_route(production):
    client, _, _, seen, _ = production
    client.post("/api/books%3Fx=1", headers=auth(), content=b"not-an-upload")
    assert seen[-1].url.raw_path.startswith(b"/api/books%3F")


def test_failed_worker_releases_quota(tmp_path):
    tenant = Tenant(
        "alice", hashlib.sha256(b"alice-key").hexdigest(), "http://alice", tmp_path, "a" * 40
    )
    store = ProductionStore(tmp_path / "quota.sqlite")

    async def broken(_):
        raise httpx.ConnectError("private connection details")

    client = httpx.AsyncClient(transport=httpx.MockTransport(broken))
    with TestClient(
        create_app(tenants=(tenant,), store=store, client=client, admin_token="a" * 40)
    ) as gateway:
        result = gateway.post("/api/ask", headers=auth(), json={"question": "private"})
        assert result.status_code == 503
        assert "private" not in result.text
        assert store.pause("alice") == 0
    asyncio.run(client.aclose())

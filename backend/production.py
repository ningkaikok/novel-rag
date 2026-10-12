"""Authenticated production entry point. Each user owns a dedicated worker/database/volume.

Run with uvicorn backend.production:app. Config is loaded at lifespan startup, not import.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlsplit

import anyio
import httpx
from fastapi import FastAPI, Request
from starlette.responses import JSONResponse, PlainTextResponse, StreamingResponse

from backend.production_store import ProductionStore, QuotaExceeded

SAFE_ID = re.compile(r"[a-z][a-z0-9-]{0,31}\Z")
HOP_HEADERS = {
    "host",
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "authorization",
    "cookie",
    "x-worker-secret",
    "x-tenant-id",
    "x-user-id",
    "x-request-id",
    "forwarded",
    "x-forwarded-for",
    "x-forwarded-host",
    "x-forwarded-proto",
}
MAX_BODY_BYTES = 22 * 1024 * 1024
ALERT_NAMES = {
    "KnowledgeGatewayDown",
    "KnowledgeWorkerDown",
    "KnowledgeDatabaseDown",
    "KnowledgeStorageNearLimit",
    "KnowledgeCallsNearLimit",
    "KnowledgeUpstreamErrors",
    "KnowledgeQuotaRejections",
    "KnowledgeIndexFailed",
    "KnowledgeBackupStale",
    "KnowledgeMaintenanceLong",
    "KnowledgeLatencyHigh",
}


class ManagedStream(StreamingResponse):
    """Release leases even if disconnect happens while sending response headers."""

    def __init__(self, *args, cleanup, **kwargs):
        super().__init__(*args, **kwargs)
        self.cleanup = cleanup

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):
                await self.cleanup()


class PrivateResponseHeaders:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def private_send(message):
            if message["type"] == "http.response.start":
                headers = [
                    (key, value)
                    for key, value in message.get("headers", [])
                    if key.lower() != b"cache-control"
                ]
                message["headers"] = headers + [
                    (b"cache-control", b"no-store"),
                    (b"x-content-type-options", b"nosniff"),
                    (b"referrer-policy", b"same-origin"),
                ]
            await send(message)

        await self.app(scope, receive, private_send)


@dataclass(frozen=True)
class Tenant:
    id: str
    token_sha256: str
    upstream: str
    data_dir: Path
    worker_secret: str
    calls_per_day: int = 200
    requests_per_minute: int = 120
    max_concurrent: int = 2
    storage_bytes: int = 1024 * 1024 * 1024
    request_timeout: int = 1800
    enabled: bool = True


def load_tenants(path: Path) -> tuple[Tenant, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tenants: list[Tenant] = []
    for entry in payload["tenants"]:
        entry = dict(entry)
        secret = os.environ.get(entry.pop("worker_secret_env"), "")
        entry["worker_secret"] = secret
        entry["data_dir"] = Path(entry["data_dir"]).resolve()
        tenant = Tenant(**entry)
        url = urlsplit(tenant.upstream)
        if not SAFE_ID.fullmatch(tenant.id) or not re.fullmatch(
            "[0-9a-f]{64}", tenant.token_sha256
        ):
            raise ValueError("Invalid tenant identity or token digest")
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username
            or url.password
            or url.path
            or url.query
            or url.fragment
        ):
            raise ValueError("Worker must be a fixed origin without credentials/path/query")
        if (
            len(secret) < 32
            or min(
                tenant.calls_per_day,
                tenant.requests_per_minute,
                tenant.max_concurrent,
                tenant.storage_bytes,
                tenant.request_timeout,
            )
            <= 0
        ):
            raise ValueError(
                "Tenant requires positive quotas and a 32+ character worker secret"
            )
        if not tenant.data_dir.is_dir():
            raise ValueError("Tenant source volume must exist")
        for other in tenants:
            if (
                tenant.id == other.id
                or tenant.token_sha256 == other.token_sha256
                or tenant.upstream == other.upstream
                or tenant.worker_secret == other.worker_secret
            ):
                raise ValueError("Each user requires a unique tenant, credential and worker")
            if (
                tenant.data_dir == other.data_dir
                or tenant.data_dir in other.data_dir.parents
                or other.data_dir in tenant.data_dir.parents
            ):
                raise ValueError("Tenant source volumes must not overlap")
        tenants.append(tenant)
    if not tenants:
        raise ValueError("At least one tenant is required")
    return tuple(tenants)


def stored_bytes(root: Path) -> int:
    """Source storage quota; includes hidden/staged files, refuses symlink escapes."""
    total = 0
    for item in root.rglob("*"):
        if item.is_symlink() or root.resolve() not in item.resolve().parents:
            raise ValueError("Symlinks are not permitted in tenant source volumes")
        if item.is_file():
            total += item.stat().st_size
    return total


def authenticate(header: str, tenants: tuple[Tenant, ...]) -> Tenant | None:
    identity = None
    if header.startswith("Bearer "):
        token = header[7:]
    elif header.startswith("Basic "):
        try:
            identity, token = (
                base64.b64decode(header[6:], validate=True).decode().split(":", 1)
            )
        except (ValueError, UnicodeError):
            return None
    else:
        return None
    digest = hashlib.sha256(token.encode()).hexdigest()
    for tenant in tenants:
        if (
            hmac.compare_digest(digest, tenant.token_sha256)
            and tenant.enabled
            and (identity is None or identity == tenant.id)
        ):
            return tenant
    return None


def _error(code: str, status: int):
    return JSONResponse(
        {"error": {"code": code}},
        status_code=status,
        headers={"WWW-Authenticate": 'Basic realm="Knowledge", charset="UTF-8"'}
        if status == 401
        else {},
    )


def create_app(*, tenants=None, store=None, client=None, admin_token=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application):
        audit = logging.getLogger("production.access")
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        audit.handlers = [handler]
        audit.setLevel(logging.INFO)
        audit.propagate = False
        application.state.tenants = tenants or load_tenants(Path(os.environ["TENANTS_CONFIG"]))
        application.state.store = store or ProductionStore(
            Path(os.environ.get("QUOTA_DB", "/control/quotas.sqlite"))
        )
        application.state.admin_token = admin_token or os.environ.get(
            "PRODUCTION_ADMIN_TOKEN", ""
        )
        if len(application.state.admin_token) < 32:
            raise ValueError("Production management requires a 32+ character admin token")
        application.state.client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(1800, connect=5),
            trust_env=False,
            follow_redirects=False,
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=30),
        )
        application.state.store.prune()
        for tenant in application.state.tenants:
            if not tenant.enabled:
                continue
            for name in (
                "requests_total",
                "upstream_errors_total",
                "quota_rejections_total",
                "request_duration_seconds_sum",
                "backup_timestamp_seconds",
            ):
                application.state.store.metric(tenant.id, name, 0)

        async def prune_control():
            while True:
                await asyncio.sleep(3600)
                application.state.store.prune()

        pruning = asyncio.create_task(prune_control())
        try:
            yield
        finally:
            pruning.cancel()
            await asyncio.gather(pruning, return_exceptions=True)
            if client is None:
                await application.state.client.aclose()

    application = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    application.add_middleware(PrivateResponseHeaders)

    def admin(request):
        return hmac.compare_digest(
            request.headers.get("authorization", "").encode(),
            ("Bearer " + application.state.admin_token).encode(),
        )

    @application.get("/healthz")
    def health():
        return {"ok": True}

    @application.get("/metrics")
    async def metrics(request: Request):
        if not admin(request):
            return _error("unauthorized", 401)
        rows = application.state.store.metrics()
        lines = [
            f'novel_rag_{name}{{tenant="{tenant}"}} {value}' for tenant, name, value in rows
        ]
        for tenant in application.state.tenants:
            if not tenant.enabled:
                continue
            up = 0
            db_up = 0
            try:
                response = await application.state.client.get(
                    tenant.upstream + "/internal/status",
                    headers={
                        "x-worker-secret": tenant.worker_secret,
                        "x-tenant-id": tenant.id,
                    },
                    timeout=5,
                )
                if response.status_code == 200:
                    up = 1
                    status = response.json()
                    db_up = int(status["database_ok"])
                    lines.append(
                        f'novel_rag_database_bytes{{tenant="{tenant.id}"}} {status["database_bytes"]}'
                    )
                    lines.append(
                        f'novel_rag_index_task_failed{{tenant="{tenant.id}"}} {int(status["index_task_failed"])}'
                    )
            except (httpx.HTTPError, ValueError, KeyError):
                pass
            lines.append(f'novel_rag_worker_up{{tenant="{tenant.id}"}} {up}')
            lines.append(f'novel_rag_database_up{{tenant="{tenant.id}"}} {db_up}')
            lines.append(
                f'novel_rag_maintenance{{tenant="{tenant.id}"}} {int(application.state.store.paused(tenant.id))}'
            )
            size = stored_bytes(tenant.data_dir)
            lines.append(f'novel_rag_storage_bytes{{tenant="{tenant.id}"}} {size}')
            lines.append(
                f'novel_rag_storage_limit_bytes{{tenant="{tenant.id}"}} {tenant.storage_bytes}'
            )
            lines.append(
                f'novel_rag_calls_today{{tenant="{tenant.id}"}} {application.state.store.usage(tenant.id)["calls"]}'
            )
            lines.append(
                f'novel_rag_calls_limit{{tenant="{tenant.id}"}} {tenant.calls_per_day}'
            )
        return PlainTextResponse(
            "\n".join(lines) + "\n", media_type="text/plain; version=0.0.4"
        )

    @application.post("/admin/tenants/{tenant_id}/{action}")
    async def maintenance(tenant_id: str, action: str, request: Request):
        if not admin(request):
            return _error("unauthorized", 401)
        tenant = next((t for t in application.state.tenants if t.id == tenant_id), None)
        if tenant is None:
            return _error("tenant_not_found", 404)
        control = application.state.store
        if action == "pause":
            active = control.pause(tenant_id)
            return JSONResponse(
                {"paused": True, "active_requests": active}, status_code=409 if active else 200
            )
        if action == "resume":
            control.resume(tenant_id)
            return {"paused": False}
        return _error("action_not_found", 404)

    @application.post("/admin/alerts")
    async def alert_webhook(request: Request):
        if not admin(request):
            return _error("unauthorized", 401)
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 1024 * 1024:
                return _error("body_too_large", 413)
        try:
            payload = json.loads(body)
            alerts = payload["alerts"]
            if not isinstance(alerts, list) or len(alerts) > 1000:
                raise ValueError()
            known_tenants = {t.id for t in application.state.tenants} | {"global"}
            records = []
            for alert in alerts:
                labels = alert["labels"]
                tenant_id = labels.get("tenant", "global")
                name = labels["alertname"]
                status = alert["status"]
                severity = labels.get("severity", "warning")
                if (
                    tenant_id not in known_tenants
                    or name not in ALERT_NAMES
                    or status not in {"firing", "resolved"}
                    or severity not in {"warning", "critical"}
                ):
                    raise ValueError()
                records.append((tenant_id, name, status, severity))
            application.state.store.record_alerts(records)
        except (ValueError, KeyError, TypeError):
            return _error("invalid_alert", 400)
        return {"accepted": len(records)}

    @application.api_route(
        "/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]
    )
    async def proxy(path: str, request: Request):
        tenant = authenticate(
            request.headers.get("authorization", ""), application.state.tenants
        )
        if tenant is None:
            return _error("unauthorized", 401)
        # Decode/canonicalize routing once. No raw paths, redirects, internal routes or spoofed identity.
        if (
            path.startswith("/")
            or "%" in path
            or path.split("/", 1)[0] in {"internal", "admin"}
            or "\\" in path
            or any(p in {".", ".."} for p in path.split("/"))
        ):
            return _error("forbidden", 403)
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            if request.headers.get("sec-fetch-site") == "cross-site" or (
                origin and urlsplit(origin).netloc != request.headers.get("host")
            ):
                return _error("cross_site_write_refused", 403)
        if path == "api/usage" and request.method == "GET":
            return {
                **application.state.store.usage(tenant.id),
                "storage_bytes": stored_bytes(tenant.data_dir),
                "storage_limit_bytes": tenant.storage_bytes,
                "calls_limit": tenant.calls_per_day,
            }
        if path == "api/alerts" and request.method == "GET":
            return {"alerts": application.state.store.alerts(tenant.id)}
        upload = path in {"api/books", "api/knowledge/documents"} and request.method == "POST"
        try:
            length = int(request.headers.get("content-length", "0"))
        except ValueError:
            return _error("invalid_content_length", 400)
        if length < 0 or length > MAX_BODY_BYTES:
            return _error("body_too_large", 413)
        if upload and not length:
            return _error("content_length_required", 411)
        control = application.state.store
        lease = None
        response = None
        started = time.monotonic()
        request_id = uuid.uuid4().hex
        try:
            lease = control.acquire(
                tenant,
                call=request.method == "POST",
                upload_bytes=length * 2 if upload else 0,
                stored_bytes=stored_bytes(tenant.data_dir),
            )
        except QuotaExceeded as exc:
            control.metric(tenant.id, "quota_rejections_total")
            return _error(exc.code, exc.status)
        try:
            body = bytearray()
            async with asyncio.timeout(30):
                async for chunk in request.stream():
                    body.extend(chunk)
                    if len(body) > MAX_BODY_BYTES or (upload and len(body) > length):
                        control.release(lease)
                        return _error("body_too_large", 413)
            headers = {
                k: v for k, v in request.headers.items() if k.lower() not in HOP_HEADERS
            }
            headers.update(
                {
                    "x-worker-secret": tenant.worker_secret,
                    "x-tenant-id": tenant.id,
                    "x-request-id": request_id,
                }
            )
            upstream = application.state.client.build_request(
                request.method,
                tenant.upstream + "/" + quote(path, safe="/"),
                params=request.query_params,
                headers=headers,
                content=bytes(body),
            )
            async with asyncio.timeout(
                max(0.01, tenant.request_timeout - (time.monotonic() - started))
            ):
                response = await application.state.client.send(upstream, stream=True)
            if 300 <= response.status_code < 400:
                await response.aclose()
                control.release(lease)
                return _error("upstream_redirect_refused", 502)
        except (httpx.HTTPError, TimeoutError):
            control.metric(tenant.id, "upstream_errors_total")
            control.release(lease)
            return _error("worker_unavailable", 503)
        except BaseException:
            control.release(lease)
            raise

        async def chunks():
            try:
                remaining = max(0.01, tenant.request_timeout - (time.monotonic() - started))
                async with asyncio.timeout(remaining):
                    async for chunk in response.aiter_raw():
                        yield chunk
            except (httpx.HTTPError, TimeoutError):
                control.metric(tenant.id, "upstream_errors_total")
                raise

        async def cleanup():
            try:
                await response.aclose()
            finally:
                control.release(lease)
                control.metric(tenant.id, "requests_total")
                control.metric(
                    tenant.id, "request_duration_seconds_sum", time.monotonic() - started
                )
                if response.status_code >= 500:
                    control.metric(tenant.id, "upstream_errors_total")
                # No path, query, body, credentials, filenames or exception text in audit logs.
                logging.getLogger("production.access").info(
                    json.dumps(
                        {
                            "event": "request_finished",
                            "request_id": request_id,
                            "tenant_id": tenant.id,
                            "method": request.method,
                            "status": response.status_code,
                            "duration_ms": round((time.monotonic() - started) * 1000),
                        }
                    )
                )

        return ManagedStream(
            chunks(),
            cleanup=cleanup,
            status_code=response.status_code,
            headers={
                **{
                    k: v
                    for k, v in response.headers.items()
                    if k.lower() not in HOP_HEADERS
                    and k.lower() not in {"set-cookie", "x-request-id"}
                },
                "x-request-id": request_id,
                "cache-control": "no-store",
            },
        )

    return application


app = create_app()

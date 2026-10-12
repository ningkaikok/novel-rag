"""Internal workers accept only their gateway, including static files and health routes."""

import hmac
import os

from starlette.responses import JSONResponse


class WorkerSecurityMiddleware:
    def __init__(self, app):
        self.app = app
        self.enabled = os.environ.get("PRODUCTION_WORKER", "0") == "1"
        self.secret = os.environ.get("WORKER_SECRET", "")
        self.tenant = os.environ.get("TENANT_ID", "")
        if self.enabled and (len(self.secret) < 32 or not self.tenant):
            raise ValueError("Production worker requires TENANT_ID and a 32+ character secret")

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not self.enabled:
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        provided = headers.get(b"x-worker-secret", b"")
        tenant = headers.get(b"x-tenant-id", b"")
        if not (
            hmac.compare_digest(provided, self.secret.encode())
            and hmac.compare_digest(tenant, self.tenant.encode())
        ):
            await JSONResponse({"error": {"code": "unauthorized"}}, status_code=401)(
                scope, receive, send
            )
            return
        await self.app(scope, receive, send)

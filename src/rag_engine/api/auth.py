from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = structlog.get_logger(__name__)

_raw = os.environ.get("RAG_TENANT_TOKENS", "{}")
TENANT_MAP: dict[str, str] = json.loads(_raw)


def resolve_tenant(token: str) -> str | None:
    return TENANT_MAP.get(token)


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if request.url.path in {"/health", "/ready"}:
            return await call_next(request)

        header = request.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            return Response(
                content='{"detail":"Missing bearer token"}',
                status_code=401,
                media_type="application/json",
            )

        token = header.removeprefix("Bearer ").strip()
        tenant_id = resolve_tenant(token)
        if tenant_id is None:
            return Response(
                content='{"detail":"Invalid token"}',
                status_code=401,
                media_type="application/json",
            )

        structlog.contextvars.bind_contextvars(tenant_id=tenant_id)
        request.state.tenant_id = tenant_id
        response = await call_next(request)
        structlog.contextvars.unbind_contextvars("tenant_id")
        return response

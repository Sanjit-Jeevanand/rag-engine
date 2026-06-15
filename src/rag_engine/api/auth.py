from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

if TYPE_CHECKING:
    import asyncpg

logger = structlog.get_logger(__name__)

_raw = os.environ.get("RAG_TENANT_TOKENS", "{}")
TENANT_MAP: dict[str, str] = json.loads(_raw)

# Set by app lifespan when RAG_DB_URL is configured.
_pool: asyncpg.Pool | None = None


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


async def resolve_tenant(token: str) -> str | None:
    if _pool is not None:
        row = await _pool.fetchrow(
            "SELECT tenant_id FROM api_keys WHERE key_hash = $1 AND active = TRUE",
            _hash(token),
        )
        if row is not None:
            return str(row["tenant_id"])
    return TENANT_MAP.get(token)


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        path = request.url.path
        if path in {"/health", "/ready"} or path.startswith("/ui"):
            return await call_next(request)

        header = request.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            return Response(
                content='{"detail":"Missing bearer token"}',
                status_code=401,
                media_type="application/json",
            )

        token = header.removeprefix("Bearer ").strip()
        tenant_id = await resolve_tenant(token)
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

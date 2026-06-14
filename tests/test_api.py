from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from rag_engine.api.app import _state, app
from rag_engine.api.models import Citation, QueryResult

# ── fixtures ──────────────────────────────────────────────────────────────────


def _make_redis(*, ping_ok: bool = True) -> AsyncMock:
    r = AsyncMock()
    r.ping = AsyncMock(return_value=True)
    r.get = AsyncMock(return_value=None)
    r.set = AsyncMock(return_value=True)
    r.incr = AsyncMock(return_value=1)
    r.hgetall = AsyncMock(return_value={})
    r.hset = AsyncMock(return_value=1)
    r.eval = AsyncMock(return_value=1)
    r.aclose = AsyncMock()
    if not ping_ok:
        r.ping = AsyncMock(side_effect=Exception("down"))
    return r


def _make_cache(*, hit: QueryResult | None = None) -> AsyncMock:
    c = AsyncMock()
    c.get = AsyncMock(return_value=hit)
    c.set = AsyncMock()
    c.hit_rate = AsyncMock(return_value=0.25)
    return c


@pytest_asyncio.fixture
async def client() -> AsyncIterator[AsyncClient]:
    _state["redis"] = _make_redis()
    _state["cache"] = _make_cache()
    _state["index_ready"] = True
    _state["retriever"] = None

    with patch("rag_engine.api.auth.TENANT_MAP", {"test-token": "test-tenant"}):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            yield ac


AUTH = {"Authorization": "Bearer test-token"}

# ── auth tests ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_missing_token(client: AsyncClient) -> None:
    r = await client.post("/query", json={"query": "hello"})
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_invalid_token(client: AsyncClient) -> None:
    r = await client.post(
        "/query",
        json={"query": "hello"},
        headers={"Authorization": "Bearer bad-token"},
    )
    assert r.status_code == 401


# ── rate limit ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rate_limit_exceeded(client: AsyncClient) -> None:
    _state["redis"].eval = AsyncMock(return_value=0)
    r = await client.post("/query", json={"query": "hello"}, headers=AUTH)
    assert r.status_code == 429
    assert "Retry-After" in r.headers


# ── cache hit ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cache_hit_returns_sse(client: AsyncClient) -> None:
    cached = QueryResult(
        query_id="cached-id",
        answer="Paris",
        citations=[],
        cache_hit=True,
        tenant_id="test-tenant",
    )
    _state["cache"] = _make_cache(hit=cached)
    r = await client.post("/query", json={"query": "capital of France"}, headers=AUTH)
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    assert "done" in r.text


# ── retrieval timeout ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_retrieval_timeout_returns_partial(client: AsyncClient) -> None:
    _state["cache"] = _make_cache(hit=None)

    async def _slow(*_: Any, **__: Any) -> list[Citation]:
        raise TimeoutError

    with patch("rag_engine.api.app._retrieve", _slow):
        r = await client.post("/query", json={"query": "slow"}, headers=AUTH)
    assert r.status_code == 200
    assert "generation_unavailable" in r.text


# ── LLM timeout ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_llm_timeout_returns_gen_unavailable(client: AsyncClient) -> None:
    import asyncio

    _state["cache"] = _make_cache(hit=None)

    async def _fast(*_: Any, **__: Any) -> list[Citation]:
        return [Citation(passage_id="p1", title="T", text="text", score=0.9)]

    async def _slow_stream(*_: Any, **__: Any) -> AsyncIterator[str]:
        await asyncio.sleep(10)
        yield "never"

    with (
        patch("rag_engine.api.app._retrieve", _fast),
        patch("rag_engine.api.app._build_answer_stream", _slow_stream),
    ):
        r = await client.post("/query", json={"query": "slow llm"}, headers=AUTH)
    assert r.status_code == 200
    assert "generation_unavailable" in r.text


# ── GET /query/{id} ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_query_result(client: AsyncClient) -> None:
    stored = QueryResult(
        query_id="abc", answer="42", citations=[], tenant_id="test-tenant"
    )
    _state["redis"].get = AsyncMock(return_value=stored.model_dump_json().encode())
    r = await client.get("/query/abc", headers=AUTH)
    assert r.status_code == 200
    assert r.json()["answer"] == "42"


@pytest.mark.asyncio
async def test_get_query_not_found(client: AsyncClient) -> None:
    _state["redis"].get = AsyncMock(return_value=None)
    r = await client.get("/query/missing", headers=AUTH)
    assert r.status_code == 404


# ── health / ready ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_health(client: AsyncClient) -> None:
    r = await client.get("/health")
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_ready_ok(client: AsyncClient) -> None:
    r = await client.get("/ready")
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_ready_redis_down(client: AsyncClient) -> None:
    _state["redis"] = _make_redis(ping_ok=False)
    r = await client.get("/ready")
    assert r.status_code == 503


@pytest.mark.asyncio
async def test_ready_index_not_loaded(client: AsyncClient) -> None:
    _state["index_ready"] = False
    r = await client.get("/ready")
    assert r.status_code == 503


# ── cache metrics ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cache_metrics(client: AsyncClient) -> None:
    r = await client.get("/metrics/cache", headers=AUTH)
    assert r.status_code == 200
    assert "hit_rate" in r.json()

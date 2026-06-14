from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast

import structlog
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from redis.asyncio import Redis
from sse_starlette.sse import EventSourceResponse

from rag_engine.api import models, ratelimit
from rag_engine.api import stream as ev
from rag_engine.api.auth import AuthMiddleware
from rag_engine.api.cache import SemanticCache
from rag_engine.config import Settings

logger = structlog.get_logger(__name__)

_RETRIEVAL_TIMEOUT = 0.2
_LLM_TIMEOUT = 5.0

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    cfg = Settings()
    redis_url = getattr(cfg, "redis_url", "redis://localhost:6379")
    redis: Redis = Redis.from_url(redis_url, decode_responses=False)
    _state["redis"] = redis
    _state["embedder"] = None
    _state["retriever"] = None
    _state["cache"] = SemanticCache(redis, _state["embedder"])
    _state["index_ready"] = False

    try:
        from sentence_transformers import SentenceTransformer

        _state["embedder"] = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
        _state["cache"] = SemanticCache(redis, _state["embedder"])
        _state["index_ready"] = True
        logger.info("api_startup_complete")
    except Exception:
        logger.warning("api_startup_partial_no_index")

    yield

    await redis.aclose()


app = FastAPI(title="RAG Engine API", lifespan=lifespan)
app.add_middleware(AuthMiddleware)


def _get_redis() -> Redis:
    return cast(Redis, _state["redis"])


def _get_cache() -> SemanticCache:
    return cast(SemanticCache, _state["cache"])


async def _retrieve(query: str, top_k: int) -> list[models.Citation]:
    retriever = _state.get("retriever")
    if retriever is None:
        return []
    passages = retriever.retrieve(query, top_k=top_k)
    return [
        models.Citation(
            passage_id=str(p.doc_id),
            title=p.title or "",
            text=p.text,
            score=float(p.score),
        )
        for p in passages
    ]


async def _build_answer_stream(
    passages: list[models.Citation],
    query: str,
) -> AsyncIterator[str]:
    from rag_engine.agent.llm import stream_complete

    context = "\n\n".join(f"[{p.passage_id}] {p.title}\n{p.text}" for p in passages)
    messages = [
        {
            "role": "user",
            "content": (
                f"Answer using only the passages below. "
                f"Cite passage IDs.\n\n{context}\n\nQuestion: {query}"
            ),
        }
    ]
    async for token in stream_complete(messages):
        yield token


async def _sse_generator(
    query: str,
    top_k: int,
    tenant_id: str,
    cache: SemanticCache,
    redis: Redis,
) -> AsyncIterator[str]:
    query_id = str(uuid.uuid4())
    log = logger.bind(query_id=query_id, tenant_id=tenant_id)

    # 1. semantic cache check
    cached = await cache.get(query)
    if cached is not None:
        cached.cache_hit = True
        cached.query_id = query_id
        for word in cached.answer.split(" "):
            yield ev.token_event(word + " ")
        await redis.set(f"result:{query_id}", cached.model_dump_json(), ex=3600)
        yield ev.done_event(query_id)
        log.info("query_served_from_cache")
        return

    # 2. retrieval with timeout
    passages: list[models.Citation] = []
    timed_out_retrieval = False
    try:
        passages = await asyncio.wait_for(
            _retrieve(query, top_k), timeout=_RETRIEVAL_TIMEOUT
        )
    except TimeoutError:
        timed_out_retrieval = True
        log.warning("retrieval_timeout")

    if timed_out_retrieval:
        yield ev.partial_event(passages)
        result = models.QueryResult(
            query_id=query_id,
            answer="",
            citations=passages,
            partial=True,
            tenant_id=tenant_id,
        )
        await redis.set(f"result:{query_id}", result.model_dump_json(), ex=3600)
        yield ev.done_event(query_id)
        return

    # 3. LLM streaming with timeout
    answer_parts: list[str] = []
    timed_out_llm = False

    try:
        gen = _build_answer_stream(passages, query)
        while True:
            try:
                token = await asyncio.wait_for(gen.__anext__(), timeout=_LLM_TIMEOUT)
                answer_parts.append(token)
                yield ev.token_event(token)
            except StopAsyncIteration:
                break
            except TimeoutError:
                timed_out_llm = True
                log.warning("llm_timeout")
                break
    except Exception as exc:
        log.error("llm_error", error=str(exc))
        yield ev.error_event(str(exc))
        return

    if timed_out_llm:
        yield ev.gen_unavailable_event(passages)
        result = models.QueryResult(
            query_id=query_id,
            answer="",
            citations=passages,
            generation_unavailable=True,
            tenant_id=tenant_id,
        )
    else:
        answer = "".join(answer_parts)
        result = models.QueryResult(
            query_id=query_id,
            answer=answer,
            citations=passages,
            tenant_id=tenant_id,
        )
        await cache.set(query, result)

    await redis.set(f"result:{query_id}", result.model_dump_json(), ex=3600)
    yield ev.done_event(query_id)
    log.info("query_complete", n_passages=len(passages))


@app.post("/query")
async def post_query(request: Request, body: models.QueryRequest) -> Response:
    tenant_id: str = getattr(request.state, "tenant_id", "unknown")
    redis = _get_redis()

    allowed = await ratelimit.check(redis, tenant_id)
    if not allowed:
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded"},
            headers={"Retry-After": "60"},
        )

    cache = _get_cache()
    return EventSourceResponse(
        _sse_generator(body.query, body.top_k, tenant_id, cache, redis)
    )


@app.get("/query/{query_id}")
async def get_query(query_id: str) -> models.QueryResult:
    redis = _get_redis()
    raw = await redis.get(f"result:{query_id}")
    if raw is None:
        raise HTTPException(status_code=404, detail="Query ID not found")
    return models.QueryResult.model_validate_json(raw)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
async def ready() -> JSONResponse:
    redis = _get_redis()
    try:
        await redis.ping()
    except Exception:
        return JSONResponse(status_code=503, content={"status": "redis unavailable"})
    if not _state.get("index_ready"):
        return JSONResponse(status_code=503, content={"status": "index not loaded"})
    return JSONResponse(content={"status": "ready"})


@app.get("/metrics/cache")
async def cache_metrics() -> dict[str, float | int]:
    cache = _get_cache()
    return {"hit_rate": await cache.hit_rate()}

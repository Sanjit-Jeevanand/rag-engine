from __future__ import annotations

import asyncio
import concurrent.futures
import sqlite3
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

import faiss
import numpy as np
import structlog
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from redis.asyncio import Redis
from starlette.responses import StreamingResponse

from rag_engine.api import auth, models, ratelimit
from rag_engine.api import stream as ev
from rag_engine.api.auth import AuthMiddleware
from rag_engine.api.cache import SemanticCache
from rag_engine.api.metrics import (
    CACHE_HIT_RATE,
    HOP_COUNT,
    QUERY_ERRORS,
    QUERY_TOTAL,
    STAGE_LATENCY,
)
from rag_engine.config import Settings
from rag_engine.retrieval import (
    BM25Retriever,
    CrossEncoderReranker,
    reciprocal_rank_fusion,
)

logger = structlog.get_logger(__name__)

_RETRIEVAL_TIMEOUT = 30.0
_LLM_TIMEOUT = 5.0

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    cfg = Settings()
    redis_url = cfg.redis_url
    redis: Redis = Redis.from_url(redis_url, decode_responses=False)
    _state["redis"] = redis
    _state["embedder"] = None
    _state["retriever"] = None
    _state["cache"] = SemanticCache(redis, _state["embedder"])
    _state["index_ready"] = False

    try:
        from sentence_transformers import SentenceTransformer

        embedder = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
        _state["embedder"] = embedder
        _state["cache"] = SemanticCache(redis, embedder)
        logger.info("embedder_loaded")
    except Exception as exc:
        logger.warning("embedder_load_failed", error=str(exc))

    try:
        hnsw_path = Path(cfg.hnsw_path)
        bm25_path = Path("data/bm25_index")
        db_path = Path("data/docs.db")

        if hnsw_path.exists() and bm25_path.exists() and db_path.exists():
            logger.info("loading_hnsw_index", path=str(hnsw_path))
            hnsw_index = faiss.read_index(str(hnsw_path))
            hnsw_index.hnsw.efSearch = cfg.hnsw_ef_search  # type: ignore[attr-defined]
            faiss.omp_set_num_threads(cfg.faiss_omp_threads)

            conn = sqlite3.connect(db_path)
            offset_rows = conn.execute(
                "SELECT vector_offset, title FROM documents"
                " WHERE status='embedded' AND chunk_index=0"
            ).fetchall()
            text_rows = conn.execute(
                "SELECT title, chunk_text FROM documents"
                " WHERE status='embedded' AND chunk_index=0"
            ).fetchall()
            conn.close()

            _state["hnsw"] = hnsw_index
            _state["first_chunk_offsets"] = {
                int(off): title for off, title in offset_rows if off is not None
            }
            _state["title_to_text"] = {title: text for title, text in text_rows}
            _state["bm25"] = BM25Retriever.load(bm25_path)
            _state["reranker"] = CrossEncoderReranker()
            _state["index_ready"] = True
            logger.info("api_startup_complete")
        else:
            logger.warning("api_startup_partial_no_index")
    except Exception as exc:
        logger.warning("api_startup_partial_no_index", error=str(exc), exc_info=True)

    if cfg.db_url:
        import asyncpg

        auth._pool = await asyncpg.create_pool(cfg.db_url, min_size=1, max_size=5)
        logger.info("postgres_pool_created")

    yield

    if auth._pool is not None:
        await auth._pool.close()
    await redis.aclose()


app = FastAPI(title="RAG Engine API", lifespan=lifespan)
app.add_middleware(AuthMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

_web = Path(__file__).parent.parent.parent.parent / "web"
if _web.exists():
    app.mount("/ui", StaticFiles(directory=_web, html=True), name="static")


def _get_redis() -> Redis:
    return cast(Redis, _state["redis"])


def _get_cache() -> SemanticCache:
    return cast(SemanticCache, _state["cache"])


def _retrieve_sync(query: str, top_k: int) -> list[models.Citation]:
    hnsw = _state.get("hnsw")
    if hnsw is None:
        return []

    embedder = _state["embedder"]
    first_chunk_offsets: dict[int, str] = _state["first_chunk_offsets"]
    title_to_text: dict[str, str] = _state["title_to_text"]
    bm25: BM25Retriever = _state["bm25"]
    reranker: CrossEncoderReranker = _state["reranker"]

    _CANDIDATE = 100
    _RERANK = 10
    _DENSE_K = 64

    cosine_sim: dict[str, float] = {}

    def _dense() -> list[str]:
        qvec = embedder.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)
        distances, indices = hnsw.search(qvec, _DENSE_K)
        seen: set[str] = set()
        ids: list[str] = []
        for dist, raw_idx in zip(distances[0], indices[0], strict=False):
            title = first_chunk_offsets.get(int(raw_idx), "")
            if title and title not in seen:
                seen.add(title)
                ids.append(title)
                cosine_sim[title] = float(dist)
            if len(ids) == _CANDIDATE:
                break
        return ids

    def _sparse() -> list[str]:
        return bm25.retrieve(query, _CANDIDATE)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        dense_fut = pool.submit(_dense)
        sparse_fut = pool.submit(_sparse)
        dense_ids = dense_fut.result()
        sparse_ids = sparse_fut.result()

    _RRF_K = 60
    _RRF_MAX = 1.0 / (_RRF_K + 1)
    sparse_rrf = {doc: 1.0 / (_RRF_K + rank + 1) for rank, doc in enumerate(sparse_ids)}

    fused = reciprocal_rank_fusion([dense_ids, sparse_ids], k=_RERANK)
    scores = reranker.scores(query, fused, title_to_text)
    ranked = np.argsort(scores)[::-1][:top_k]

    return [
        models.Citation(
            passage_id=str(i + 1),
            title=fused[int(idx)],
            text=title_to_text.get(fused[int(idx)], "")[:800],
            score=float(scores[int(idx)]),
            dense_score=cosine_sim.get(fused[int(idx)], 0.0),
            bm25_score=sparse_rrf.get(fused[int(idx)], 0.0) / _RRF_MAX,
        )
        for i, idx in enumerate(ranked)
    ]


async def _retrieve(query: str, top_k: int) -> list[models.Citation]:
    return await asyncio.to_thread(_retrieve_sync, query, top_k)


def _rerank_sync(
    query: str, passages: list[models.Citation], top_k: int
) -> list[models.Citation]:
    reranker: CrossEncoderReranker = _state["reranker"]
    title_to_text: dict[str, str] = _state["title_to_text"]
    titles = [p.title for p in passages]
    scores = reranker.scores(query, titles, title_to_text)
    ranked = np.argsort(scores)[::-1][:top_k]
    result = []
    for new_i, idx in enumerate(ranked):
        p = passages[int(idx)]
        result.append(
            models.Citation(
                passage_id=str(new_i + 1),
                title=p.title,
                text=p.text,
                score=float(scores[int(idx)]),
                dense_score=p.dense_score,
                bm25_score=p.bm25_score,
            )
        )
    return result


async def _extract_bridge(query: str, passages: list[models.Citation]) -> str | None:
    from rag_engine.agent.llm import stream_complete

    context = "\n\n".join(
        f"Passage {i} (title: {p.title}):\n{p.text[:300]}"
        for i, p in enumerate(passages[:3], 1)
    )
    messages = [
        {
            "role": "user",
            "content": (
                f"Question: {query}\n\n"
                f"Retrieved passages:\n{context}\n\n"
                "Based on these passages, what single entity or concept must be "
                "looked up next to fully answer the question? "
                "Reply with ONLY the search term (a name, place, or concept), "
                "nothing else. "
                "If the passages already contain the full answer, reply with NONE."
            ),
        }
    ]
    parts: list[str] = []
    async for token in stream_complete(messages):
        parts.append(token)
    bridge = "".join(parts).strip()
    return None if not bridge or bridge.upper() == "NONE" else bridge


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
    max_hops: int,
    tenant_id: str,
    cache: SemanticCache,
    redis: Redis,
) -> AsyncIterator[str]:
    query_id = str(uuid.uuid4())
    log = logger.bind(query_id=query_id, tenant_id=tenant_id)
    t0 = time.monotonic()

    yield ev.query_id_event(query_id)

    # 1. semantic cache check (embedding happens inside cache.get)
    t_embed = time.monotonic()
    cached = await cache.get(query)
    embed_ms = int((time.monotonic() - t_embed) * 1000)

    if cached is not None:
        yield ev.cache_hit_event(hit=True)
        for i, p in enumerate(cached.citations, 1):
            yield ev.passage_event(p, i)
        cached.cache_hit = True
        cached.query_id = query_id
        for word in cached.answer.split(" "):
            yield ev.token_event(word + " ")
        await redis.set(f"result:{query_id}", cached.model_dump_json(), ex=3600)
        total_ms = int((time.monotonic() - t0) * 1000)
        yield ev.done_event(query_id, cached=True, total_ms=total_ms)
        QUERY_TOTAL.labels(tenant=tenant_id, cache_hit="true").inc()
        CACHE_HIT_RATE.set(await cache.hit_rate())
        log.info("query_served_from_cache", total_ms=total_ms)
        return

    STAGE_LATENCY.labels(stage="embed").observe(embed_ms / 1000)
    yield ev.cache_hit_event(hit=False)
    yield ev.trace_step_event("embed", "Embed query", "bge-small-en-v1.5", embed_ms)

    # 2. retrieval with timeout
    passages: list[models.Citation] = []
    timed_out_retrieval = False
    t_retrieve = time.monotonic()
    try:
        passages = await asyncio.wait_for(
            _retrieve(query, top_k), timeout=_RETRIEVAL_TIMEOUT
        )
    except TimeoutError:
        timed_out_retrieval = True
        log.warning("retrieval_timeout")
    retrieve_ms = int((time.monotonic() - t_retrieve) * 1000)

    STAGE_LATENCY.labels(stage="retrieve_hop1").observe(retrieve_ms / 1000)
    if timed_out_retrieval:
        QUERY_ERRORS.labels(tenant=tenant_id, stage="retrieve_hop1").inc()
    yield ev.trace_step_event(
        "retrieve",
        "Hop 1 — retrieve" if max_hops > 1 else "Retrieve passages",
        f"hybrid · top-{top_k}",
        retrieve_ms,
    )
    for i, p in enumerate(passages, 1):
        yield ev.passage_event(p, i, hop=1 if max_hops > 1 else None)

    # 2b. multi-hop: extract bridge entity and retrieve again
    if max_hops > 1 and passages and not timed_out_retrieval:
        t_bridge = time.monotonic()
        try:
            bridge = await asyncio.wait_for(
                _extract_bridge(query, passages), timeout=8.0
            )
        except Exception:
            bridge = None
        bridge_ms = int((time.monotonic() - t_bridge) * 1000)
        STAGE_LATENCY.labels(stage="bridge").observe(bridge_ms / 1000)

        if bridge:
            yield ev.trace_step_event(
                "hop1",
                f"Bridge → {bridge[:50]}",
                f"from top-{min(3, len(passages))} passages",
                bridge_ms,
                reflect=True,
            )

            t_hop2 = time.monotonic()
            try:
                hop2_passages = await asyncio.wait_for(
                    _retrieve(bridge, top_k), timeout=_RETRIEVAL_TIMEOUT
                )
            except TimeoutError:
                hop2_passages = []
            hop2_ms = int((time.monotonic() - t_hop2) * 1000)
            STAGE_LATENCY.labels(stage="retrieve_hop2").observe(hop2_ms / 1000)

            yield ev.trace_step_event(
                "hop2", "Hop 2 — retrieve", f"query: {bridge[:40]}", hop2_ms
            )

            seen_titles = {p.title for p in passages}
            new_passages = [p for p in hop2_passages if p.title not in seen_titles]
            if new_passages:
                offset = len(passages)
                for i, p in enumerate(new_passages, 1):
                    yield ev.passage_event(p, offset + i, hop=2)

            merged = passages + new_passages
            if merged:
                passages = await asyncio.to_thread(_rerank_sync, query, merged, top_k)

    if timed_out_retrieval:
        total_ms = int((time.monotonic() - t0) * 1000)
        result = models.QueryResult(
            query_id=query_id,
            answer="",
            citations=passages,
            partial=True,
            tenant_id=tenant_id,
        )
        await redis.set(f"result:{query_id}", result.model_dump_json(), ex=3600)
        yield ev.done_event(
            query_id,
            generation_unavailable=True,
            lat={
                "embed": embed_ms,
                "retrieve": retrieve_ms,
                "rerank": 0,
                "generate": 0,
            },  # noqa: E501
            total_ms=total_ms,
        )
        return

    # 3. LLM streaming with timeout
    yield ev.generation_start_event()
    answer_parts: list[str] = []
    timed_out_llm = False
    t_gen = time.monotonic()

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

    gen_ms = int((time.monotonic() - t_gen) * 1000)
    total_ms = int((time.monotonic() - t0) * 1000)
    lat = {"embed": embed_ms, "retrieve": retrieve_ms, "rerank": 0, "generate": gen_ms}
    STAGE_LATENCY.labels(stage="generate").observe(gen_ms / 1000)
    if timed_out_llm:
        QUERY_ERRORS.labels(tenant=tenant_id, stage="generate").inc()

    yield ev.trace_step_event(
        "generate",
        "Generate answer",
        "gpt-4o-mini",
        gen_ms,
        skipped=timed_out_llm,
    )

    if timed_out_llm:
        result = models.QueryResult(
            query_id=query_id,
            answer="",
            citations=passages,
            generation_unavailable=True,
            tenant_id=tenant_id,
        )
        yield ev.done_event(
            query_id, generation_unavailable=True, lat=lat, total_ms=total_ms
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
        yield ev.done_event(query_id, mode="grounded", lat=lat, total_ms=total_ms)

    await redis.set(f"result:{query_id}", result.model_dump_json(), ex=3600)
    hop_count = 1 + (1 if max_hops > 1 and "bridge" in locals() and bridge else 0)
    QUERY_TOTAL.labels(tenant=tenant_id, cache_hit="false").inc()
    HOP_COUNT.observe(hop_count)
    CACHE_HIT_RATE.set(await cache.hit_rate())
    log.info(
        "query_complete",
        n_passages=len(passages),
        hop_count=hop_count,
        embed_ms=embed_ms,
        retrieve_ms=retrieve_ms,
        gen_ms=gen_ms,
        total_ms=total_ms,
    )


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
    return StreamingResponse(
        _sse_generator(body.query, body.top_k, body.max_hops, tenant_id, cache, redis),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
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


@app.get("/metrics")
async def prometheus_metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

from __future__ import annotations

import json
import pickle
from typing import Any, cast

import numpy as np
import structlog
from redis.asyncio import Redis

from rag_engine.api.models import QueryResult

logger = structlog.get_logger(__name__)

_SIM_THRESHOLD = 0.97
_TTL_SECONDS = 3600
_EMBED_KEY = "cache:embeddings"
_HIT_KEY = "cache:hits"
_TOTAL_KEY = "cache:total"


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class SemanticCache:
    def __init__(self, redis: Redis, embedder: Any) -> None:
        self._redis = redis
        self._embedder = embedder

    def _embed(self, query: str) -> np.ndarray:
        vecs = self._embedder.encode([query], normalize_embeddings=True)
        return np.array(vecs[0], dtype=np.float32)

    async def get(self, query: str) -> QueryResult | None:
        await self._redis.incr(_TOTAL_KEY)
        q_vec = self._embed(query)

        stored = cast(dict[bytes, bytes], await self._redis.hgetall(_EMBED_KEY))
        best_key: str | None = None
        best_sim = 0.0

        for cache_key_b, emb_b in stored.items():
            cached_vec: np.ndarray = pickle.loads(emb_b)  # noqa: S301
            sim = _cosine(q_vec, cached_vec)
            if sim > best_sim:
                best_sim = sim
                best_key = cache_key_b.decode()

        if best_key is not None and best_sim >= _SIM_THRESHOLD:
            raw = await self._redis.get(f"cache:result:{best_key}")
            if raw:
                await self._redis.incr(_HIT_KEY)
                logger.info("semantic_cache_hit", sim=round(best_sim, 4))
                return QueryResult.model_validate(json.loads(raw))

        logger.info("semantic_cache_miss")
        return None

    async def set(self, query: str, result: QueryResult) -> None:
        q_vec = self._embed(query)
        cache_key = result.query_id
        await self._redis.hset(_EMBED_KEY, cache_key, pickle.dumps(q_vec))
        await self._redis.set(
            f"cache:result:{cache_key}",
            result.model_dump_json(),
            ex=_TTL_SECONDS,
        )

    async def hit_rate(self) -> float:
        hits = int(await self._redis.get(_HIT_KEY) or 0)
        total = int(await self._redis.get(_TOTAL_KEY) or 0)
        return hits / total if total else 0.0

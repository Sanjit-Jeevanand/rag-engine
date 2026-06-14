from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str
    max_hops: int = Field(default=2, ge=1, le=3)
    top_k: int = Field(default=5, ge=1, le=20)


class Citation(BaseModel):
    passage_id: str
    title: str
    text: str
    score: float
    dense_score: float = 0.0
    bm25_score: float = 0.0


class QueryResult(BaseModel):
    query_id: str
    answer: str
    citations: list[Citation]
    cache_hit: bool = False
    partial: bool = False
    generation_unavailable: bool = False
    tenant_id: str = ""
    cost_usd: float | None = None

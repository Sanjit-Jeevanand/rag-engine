from __future__ import annotations

import json
from typing import Any

from rag_engine.api.models import Citation


def _ev(name: str, payload: dict[str, Any]) -> str:
    return f"event: {name}\ndata: {json.dumps(payload)}\n\n"


def query_id_event(query_id: str) -> str:
    return _ev("query_id", {"id": query_id})


def cache_hit_event(hit: bool, sim: float | None = None) -> str:
    payload: dict[str, Any] = {"hit": hit}
    if sim is not None:
        payload["sim"] = round(sim, 4)
    return _ev("cache_hit", payload)


def trace_step_event(
    step: str,
    label: str,
    sub: str,
    ms: int,
    *,
    reflect: bool = False,
    skipped: bool = False,
) -> str:
    return _ev(
        "trace_step",
        {
            "step": step,
            "label": label,
            "sub": sub,
            "ms": ms,
            "reflect": reflect,
            "skipped": skipped,
        },
    )


def passage_event(p: Citation, num: int, hop: int | None = None) -> str:
    payload: dict[str, Any] = {
        "num": num,
        "title": p.title,
        "snippet": p.text[:220],
        "dense": round(p.dense_score, 4),
        "bm25": round(p.bm25_score, 4),
        "rerank": round(p.score, 4),
    }
    if hop is not None:
        payload["hop"] = hop
    return _ev("passage", payload)


def generation_start_event() -> str:
    return _ev("generation_start", {})


def token_event(text: str) -> str:
    return _ev("token", {"text": text})


def done_event(
    query_id: str,
    *,
    mode: str = "grounded",
    abstained: bool = False,
    cached: bool = False,
    generation_unavailable: bool = False,
    lat: dict[str, int] | None = None,
    total_ms: int = 0,
) -> str:
    payload: dict[str, Any] = {
        "query_id": query_id,
        "mode": mode,
        "abstained": abstained,
        "cached": cached,
        "generation_unavailable": generation_unavailable,
        "lat": lat or {"embed": 0, "retrieve": 0, "rerank": 0, "generate": 0},
        "total_ms": total_ms,
    }
    return _ev("done", payload)


def error_event(message: str) -> str:
    return _ev("error", {"message": message})

from __future__ import annotations

import json

from rag_engine.api.models import Citation


def token_event(text: str) -> str:
    return f"data: {json.dumps({'type': 'token', 'text': text})}\n\n"


def done_event(query_id: str) -> str:
    return f"data: {json.dumps({'type': 'done', 'query_id': query_id})}\n\n"


def partial_event(passages: list[Citation]) -> str:
    payload = {
        "type": "partial",
        "passages": [p.model_dump() for p in passages],
    }
    return f"data: {json.dumps(payload)}\n\n"


def gen_unavailable_event(passages: list[Citation]) -> str:
    payload = {
        "type": "generation_unavailable",
        "passages": [p.model_dump() for p in passages],
    }
    return f"data: {json.dumps(payload)}\n\n"


def error_event(message: str) -> str:
    return f"data: {json.dumps({'type': 'error', 'message': message})}\n\n"

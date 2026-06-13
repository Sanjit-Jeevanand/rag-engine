from typing import Any

import numpy as np
from sentence_transformers import CrossEncoder

from rag_engine.cost import cost_tracker


class CrossEncoderReranker:
    _model: Any

    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        self._model = CrossEncoder(model_name)

    def scores(
        self,
        query: str,
        candidates: list[str],
        doc_texts: dict[str, str],
    ) -> np.ndarray:
        pairs = [(query, doc_texts.get(doc_id, "")) for doc_id in candidates]
        raw: Any = self._model.predict(pairs)
        cost_tracker.add_reranker(len(candidates))
        return np.asarray(raw, dtype=np.float32)

    def rerank(
        self,
        query: str,
        candidates: list[str],
        doc_texts: dict[str, str],
        k: int,
    ) -> list[str]:
        pairs = [(query, doc_texts.get(doc_id, "")) for doc_id in candidates]
        raw: Any = self._model.predict(pairs)
        scores: np.ndarray = np.asarray(raw, dtype=np.float32)
        cost_tracker.add_reranker(len(candidates))
        ranked_idx = np.argsort(scores)[::-1]
        return [candidates[int(i)] for i in ranked_idx[:k]]

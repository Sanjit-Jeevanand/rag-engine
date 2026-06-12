from typing import Any

import faiss
import numpy as np


class DenseRetriever:
    _index: Any

    def __init__(self, doc_ids: list[str], corpus_vecs: np.ndarray) -> None:
        self._doc_ids = doc_ids
        dim = int(corpus_vecs.shape[1])
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(corpus_vecs)

    def retrieve(self, query_vec: np.ndarray, k: int) -> list[str]:
        q = query_vec.reshape(1, -1).astype(np.float32)
        _, indices = self._index.search(q, k)
        return [self._doc_ids[int(i)] for i in indices[0] if int(i) >= 0]

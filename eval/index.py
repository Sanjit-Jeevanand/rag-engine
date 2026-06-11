import sqlite3
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
VECTOR_DIM = 384
DEDUP_FACTOR = 5  # fetch k*5 chunks then dedup to k unique articles


class VectorIndex:
    def __init__(
        self, db_path: Path, vectors_path: Path = Path("data/vectors.bin")
    ) -> None:
        vectors = np.fromfile(vectors_path, dtype=np.float32).reshape(-1, VECTOR_DIM)
        n = len(vectors)

        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT vector_offset, title FROM documents"
            " WHERE status = 'embedded'"
            " ORDER BY vector_offset"
        ).fetchall()
        conn.close()

        self._titles: list[str] = [""] * n
        for offset, title in rows:
            if offset < n:
                self._titles[offset] = title

        self._index = faiss.IndexFlatIP(VECTOR_DIM)
        self._index.add(vectors)

        self._model = SentenceTransformer(MODEL_NAME, device="mps")

    def search(self, query: str, k: int = 10) -> list[str]:
        qvec = self._model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        _, indices = self._index.search(qvec, k * DEDUP_FACTOR)

        seen: set[str] = set()
        results: list[str] = []
        for idx in indices[0]:
            title = self._titles[idx]
            if title and title not in seen:
                seen.add(title)
                results.append(title)
            if len(results) == k:
                break

        return results

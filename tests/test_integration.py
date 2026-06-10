"""
End-to-end integration test: parse → chunk → embed → verify.
Uses a synthetic fixture (100 articles) so no real download is needed.
"""

import gzip
import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from rag_engine.ingest.embedder import VECTOR_DIM, run_embedder
from rag_engine.ingest.pipeline import run_pipeline

PATCH_TARGET = "rag_engine.ingest.embedder.SentenceTransformer"
N_ARTICLES = 100


def _make_dump(path: Path, n: int) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for i in range(n):
            index = {"index": {"_type": "page", "_id": str(i), "_index": "enwiki"}}
            content = {
                "namespace": 0,
                "title": f"Article {i}",
                "text": f"This is the full text of article {i}. " * 40,
                "categories": ["Category A", "Category B"],
                "timestamp": "2024-01-01T00:00:00Z",
            }
            f.write(json.dumps(index) + "\n")
            f.write(json.dumps(content) + "\n")


def _fake_model() -> MagicMock:
    model = MagicMock()
    model.encode.side_effect = lambda texts, **kw: np.zeros(
        (len(texts), VECTOR_DIM), dtype=np.float32
    )
    return model


def test_end_to_end() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        dump_path = Path(tmp) / "wiki.json.gz"
        db_path = Path(tmp) / "docs.db"
        vectors_path = Path(tmp) / "vectors.bin"

        _make_dump(dump_path, N_ARTICLES)
        run_pipeline(dump_path, db_path)

        conn = sqlite3.connect(db_path)
        total_chunks = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        assert total_chunks > 0

        with patch(PATCH_TARGET, return_value=_fake_model()):
            run_embedder(db_path, vectors_path)

        # all chunks embedded
        pending = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE status = 'pending'"
        ).fetchone()[0]
        assert pending == 0

        # vector file has correct shape
        mmap = np.memmap(
            vectors_path, dtype="float32", mode="r", shape=(total_chunks, VECTOR_DIM)
        )
        assert mmap.shape == (total_chunks, VECTOR_DIM)

        # offsets are unique and sequential
        offsets = sorted(
            r[0] for r in conn.execute("SELECT vector_offset FROM documents").fetchall()
        )
        assert offsets == list(range(total_chunks))

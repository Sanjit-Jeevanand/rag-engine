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
N_ARTICLES = 10


def _make_dump(path: Path, n: int) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for i in range(n):
            f.write(
                json.dumps(
                    {"index": {"_type": "page", "_id": str(i), "_index": "enwiki"}}
                )
                + "\n"
            )
            f.write(
                json.dumps(
                    {
                        "namespace": 0,
                        "title": f"Article {i}",
                        "text": f"Text of article {i}. " * 20,
                        "categories": ["Test"],
                        "timestamp": "2024-01-01T00:00:00Z",
                    }
                )
                + "\n"
            )


def _fake_model() -> MagicMock:
    model = MagicMock()
    model.encode.side_effect = lambda texts, **kw: np.zeros(
        (len(texts), VECTOR_DIM), dtype=np.float32
    )
    return model


def test_only_failed_chunk_reembeds_after_crash() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        dump_path = Path(tmp) / "wiki.json.gz"
        db_path = Path(tmp) / "docs.db"
        vectors_path = Path(tmp) / "vectors.bin"

        _make_dump(dump_path, N_ARTICLES)
        run_pipeline(dump_path, db_path)

        # first run: embed everything
        with patch(PATCH_TARGET, return_value=_fake_model()):
            run_embedder(db_path, vectors_path)

        conn = sqlite3.connect(db_path)
        total_chunks = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        size_after_full = vectors_path.stat().st_size
        assert size_after_full == total_chunks * VECTOR_DIM * 4

        # simulate crash: flip one row back to 'pending'
        conn.execute(
            "UPDATE documents SET status = 'pending', vector_offset = NULL,"
            " embedded_at = NULL, checksum = NULL WHERE rowid = 1"
        )
        conn.commit()

        # restart: only the one pending row should be re-embedded
        with patch(PATCH_TARGET, return_value=_fake_model()):
            run_embedder(db_path, vectors_path)

        # file truncated to match DB then re-embedded — same size as before
        size_after_restart = vectors_path.stat().st_size
        assert size_after_restart == size_after_full

        # that row is now embedded with a new offset at the end
        row = conn.execute(
            "SELECT status, vector_offset FROM documents WHERE rowid = 1"
        ).fetchone()
        assert row[0] == "embedded"
        assert row[1] == total_chunks - 1  # appended at position N-1 (9 still embedded)

        # all other rows are unchanged
        still_embedded = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE status = 'embedded'"
        ).fetchone()[0]
        assert still_embedded == total_chunks

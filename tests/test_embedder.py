import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from rag_engine.ingest.embedder import VECTOR_DIM, run_embedder
from rag_engine.ingest.pipeline import run_pipeline

FIXTURE = Path(__file__).parent / "fixtures" / "sample_snapshot.jsonl"
PATCH_TARGET = "rag_engine.ingest.embedder.SentenceTransformer"


def _fake_model() -> MagicMock:
    model = MagicMock()
    model.encode.side_effect = lambda texts, **kw: np.zeros(
        (len(texts), VECTOR_DIM), dtype=np.float32
    )
    return model


def _setup(tmp: str) -> tuple[Path, Path]:
    db_path = Path(tmp) / "test.db"
    vectors_path = Path(tmp) / "vectors.bin"
    run_pipeline(FIXTURE, db_path)
    return db_path, vectors_path


def test_creates_vector_file() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path, vectors_path = _setup(tmp)
        with patch(PATCH_TARGET, return_value=_fake_model()):
            run_embedder(db_path, vectors_path)
        assert vectors_path.exists()
        expected = 3 * VECTOR_DIM * 4  # 3 chunks × 1024 floats × 4 bytes
        assert vectors_path.stat().st_size == expected


def test_vector_file_loadable_as_memmap() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path, vectors_path = _setup(tmp)
        with patch(PATCH_TARGET, return_value=_fake_model()):
            run_embedder(db_path, vectors_path)
        mmap = np.memmap(vectors_path, dtype="float32", mode="r", shape=(3, VECTOR_DIM))
        assert mmap.shape == (3, VECTOR_DIM)


def test_updates_status_to_embedded() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path, vectors_path = _setup(tmp)
        with patch(PATCH_TARGET, return_value=_fake_model()):
            run_embedder(db_path, vectors_path)
        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT status FROM documents").fetchall()
        assert all(r[0] == "embedded" for r in rows)


def test_vector_offsets_are_sequential() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path, vectors_path = _setup(tmp)
        with patch(PATCH_TARGET, return_value=_fake_model()):
            run_embedder(db_path, vectors_path)
        conn = sqlite3.connect(db_path)
        offsets = sorted(
            r[0] for r in conn.execute("SELECT vector_offset FROM documents").fetchall()
        )
        assert offsets == [0, 1, 2]


def test_idempotent() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path, vectors_path = _setup(tmp)
        with patch(PATCH_TARGET, return_value=_fake_model()):
            run_embedder(db_path, vectors_path)
            size_after_first = vectors_path.stat().st_size
            run_embedder(db_path, vectors_path)
            assert vectors_path.stat().st_size == size_after_first

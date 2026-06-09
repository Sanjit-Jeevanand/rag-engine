import sqlite3
import tempfile
from pathlib import Path

from rag_engine.ingest.pipeline import (
    CHUNK_CHARS,
    OVERLAP_CHARS,
    run_pipeline,
    split_text,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sample_snapshot.jsonl"


def test_split_text_single_chunk() -> None:
    chunks = split_text("short text")
    assert len(chunks) == 1
    assert chunks[0] == "short text"


def test_split_text_multiple_chunks() -> None:
    text = "x" * 3000
    chunks = split_text(text)
    # start=0 → 1300 → 2600 → 3900 (stop): 3 chunks
    assert len(chunks) == 3


def test_split_text_overlap() -> None:
    text = "a" * CHUNK_CHARS + "b" * CHUNK_CHARS
    chunks = split_text(text)
    # last OVERLAP_CHARS of chunk[0] == first OVERLAP_CHARS of chunk[1]
    assert chunks[0][-OVERLAP_CHARS:] == chunks[1][:OVERLAP_CHARS]


def test_run_pipeline_inserts_rows() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        run_pipeline(FIXTURE, db_path)

        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT article_id, status FROM documents").fetchall()
        assert len(rows) == 3  # 3 fixture articles, each fits in 1 chunk
        assert all(status == "pending" for _, status in rows)


def test_run_pipeline_idempotent() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        run_pipeline(FIXTURE, db_path)
        run_pipeline(FIXTURE, db_path)  # second run must not duplicate rows

        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        assert count == 3

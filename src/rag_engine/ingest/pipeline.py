import sqlite3
from pathlib import Path

from rag_engine.ingest.parser import parse_snapshot
from rag_engine.ingest.schema import init_db

CHUNK_CHARS = 1500
OVERLAP_CHARS = 200
BATCH_SIZE = 1000

Row = tuple[str, int, str, str, str, str, int]


def split_text(
    text: str,
    chunk_chars: int = CHUNK_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + chunk_chars])
        start += chunk_chars - overlap_chars
    return chunks


def _insert_batch(conn: sqlite3.Connection, batch: list[Row]) -> None:
    conn.executemany(
        """
        INSERT OR IGNORE INTO documents
            (article_id, chunk_index, title, categories, timestamp,
             chunk_text, chunk_count)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        batch,
    )
    conn.commit()


def run_pipeline(
    snapshot_path: Path,
    db_path: Path,
    max_articles: int | None = None,
    chunk_chars: int = CHUNK_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
) -> None:
    conn = init_db(db_path)
    batch: list[Row] = []
    for article_count, article in enumerate(parse_snapshot(snapshot_path)):
        if max_articles is not None and article_count >= max_articles:
            break
        chunks = split_text(article.text, chunk_chars, overlap_chars)
        chunk_count = len(chunks)
        for i, text in enumerate(chunks):
            batch.append(
                (
                    article.article_id,
                    i,
                    article.title,
                    article.categories,
                    article.timestamp,
                    text,
                    chunk_count,
                )
            )
            if len(batch) >= BATCH_SIZE:
                _insert_batch(conn, batch)
                batch.clear()

    if batch:
        _insert_batch(conn, batch)

    conn.close()

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    article_id    TEXT    NOT NULL,
    chunk_index   INTEGER NOT NULL,
    title         TEXT    NOT NULL,
    categories    TEXT    NOT NULL,
    timestamp     TEXT    NOT NULL,
    chunk_text    TEXT    NOT NULL,
    chunk_count   INTEGER NOT NULL,
    vector_offset INTEGER,
    status        TEXT    NOT NULL DEFAULT 'pending',
    embedded_at   TEXT,
    checksum      TEXT,
    PRIMARY KEY (article_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_status ON documents (status);
"""


def init_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn

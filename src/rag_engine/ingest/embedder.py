import hashlib
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-large-en-v1.5"
EMBED_BATCH = 256
VECTOR_DIM = 1024


def _next_offset(conn: sqlite3.Connection) -> int:
    row: Any = conn.execute(
        "SELECT COUNT(*) FROM documents WHERE status = 'embedded'"
    ).fetchone()
    return int(row[0])


def run_embedder(
    db_path: Path, vectors_path: Path, show_progress: bool = False
) -> None:
    model = SentenceTransformer(MODEL_NAME)
    conn = sqlite3.connect(db_path)
    vectors_path.parent.mkdir(parents=True, exist_ok=True)
    offset = _next_offset(conn)

    while True:
        rows: list[Any] = conn.execute(
            """
            SELECT rowid, chunk_text FROM documents
            WHERE status = 'pending'
            LIMIT ?
            """,
            (EMBED_BATCH,),
        ).fetchall()
        if not rows:
            break

        texts = [str(row[1]) for row in rows]
        vectors = np.array(
            model.encode(
                texts,
                batch_size=EMBED_BATCH,
                normalize_embeddings=True,
                show_progress_bar=show_progress,
            ),
            dtype=np.float32,
        )

        with vectors_path.open("ab") as f:
            f.write(vectors.tobytes())

        now = datetime.now(UTC).isoformat()
        updates = [
            (
                offset + i,
                now,
                hashlib.sha256(str(row[1]).encode()).hexdigest(),
                int(row[0]),
            )
            for i, row in enumerate(rows)
        ]
        conn.executemany(
            """
            UPDATE documents
            SET status = 'embedded', vector_offset = ?, embedded_at = ?, checksum = ?
            WHERE rowid = ?
            """,
            updates,
        )
        conn.commit()
        offset += len(rows)

    conn.close()

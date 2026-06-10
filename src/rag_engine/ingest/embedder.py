import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_BATCH = 256  # inner encode batch — MPS sweet spot for bge-small
DB_FETCH = 4096  # rows per outer loop — amortizes SQL, commit, and file I/O
VECTOR_DIM = 384  # bge-small output dim
DEVICE = "mps"

# flush MPS allocator every outer loop (~4096 chunks) — prevents unbounded growth
_MPS_FLUSH_EVERY = 1


def _embedded_count(conn: sqlite3.Connection) -> int:
    row: Any = conn.execute(
        "SELECT COUNT(*) FROM documents WHERE status = 'embedded'"
    ).fetchone()
    return int(row[0])


def run_embedder(
    db_path: Path, vectors_path: Path, show_progress: bool = False
) -> None:
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA synchronous=NORMAL")  # WAL + NORMAL: no fsync per commit
    vectors_path.parent.mkdir(parents=True, exist_ok=True)

    offset = _embedded_count(conn)

    # crash safety: reconcile file and DB in both directions
    file_vectors = (
        vectors_path.stat().st_size // (VECTOR_DIM * 4) if vectors_path.exists() else 0
    )
    if file_vectors < offset:
        # DB claims more than file contains — reset orphaned rows to pending
        conn.execute(
            "UPDATE documents SET status='pending', vector_offset=NULL,"
            " embedded_at=NULL WHERE vector_offset >= ?",
            (file_vectors,),
        )
        conn.commit()
        offset = file_vectors
    else:
        expected = offset * VECTOR_DIM * 4
        if vectors_path.exists() and vectors_path.stat().st_size > expected:
            # file has more bytes than DB committed — truncate orphan vectors
            with vectors_path.open("r+b") as f:
                f.truncate(expected)

    loop = 0
    start = time.time()
    done_since_start = 0

    with vectors_path.open("ab") as vf:
        while True:
            rows: list[Any] = conn.execute(
                "SELECT rowid, chunk_text FROM documents"
                " WHERE status = 'pending' LIMIT ?",
                (DB_FETCH,),
            ).fetchall()
            if not rows:
                break

            texts = [row[1] for row in rows]
            vectors = model.encode(
                texts,
                batch_size=EMBED_BATCH,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            vf.write(np.asarray(vectors, dtype=np.float32).tobytes())
            vf.flush()  # file bytes must be durable before the DB says 'embedded'

            now = datetime.now(UTC).isoformat()
            conn.executemany(
                "UPDATE documents SET status='embedded', vector_offset=?,"
                " embedded_at=? WHERE rowid=?",
                [(offset + i, now, int(row[0])) for i, row in enumerate(rows)],
            )
            conn.commit()
            offset += len(rows)
            done_since_start += len(rows)
            loop += 1

            if DEVICE == "mps" and loop % _MPS_FLUSH_EVERY == 0:
                torch.mps.empty_cache()

            if show_progress:
                rate = done_since_start / (time.time() - start)
                print(
                    f"\rembedded {offset:,}  ({rate:,.0f} vec/s)",
                    end="",
                    flush=True,
                )

    conn.close()

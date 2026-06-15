"""
Prepare the 100K production dataset from the existing 1M dataset.

Steps:
  1. Rename 1M files to *_1m.* (docs.db → docs_1m.db, etc.)
  2. Build new docs.db with only 100K article rows, vector_offsets renumbered 0, 1, 2...
  3. Build new vectors.bin with only those vectors in the same order
  4. Rename hnsw_100k.index → hnsw.index

After this the default filenames (docs.db, hnsw.index, vectors.bin) are the 100K
production set. The HNSW sequential IDs 0, 1, 2... align exactly with the new
vector_offsets, so no id_map translation is needed.

Usage:
    uv run python scripts/build_100k_dataset.py
"""

import json
import shutil
import sqlite3
import time
from pathlib import Path

import numpy as np

DATA = Path("data")
VECTOR_DIM = 384


def rename_1m_files() -> None:
    renames = [
        (DATA / "docs.db", DATA / "docs_1m.db"),
        (DATA / "vectors.bin", DATA / "vectors_1m.bin"),
        (DATA / "hnsw.index", DATA / "hnsw_1m.index"),
    ]
    for src, dst in renames:
        if src.exists() and not dst.exists():
            src.rename(dst)
            print(f"  {src.name} → {dst.name}")
        elif dst.exists():
            print(f"  {dst.name} already exists, skipping rename of {src.name}")
        else:
            print(f"  {src.name} not found, skipping")


def load_title_list() -> list[str]:
    path = DATA / "title_scores_100k.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run build_production_index.py first"
        )
    titles = json.loads(path.read_text())
    print(f"  Loaded {len(titles):,} titles from {path.name}")
    return titles


def build_100k_db(titles: list[str]) -> list[int]:
    """
    Copy rows for the 100K titles from docs_1m.db into a new docs.db,
    renumbering vector_offset sequentially (0, 1, 2...).

    Returns the list of original offsets in the order they were assigned new IDs,
    so we can write vectors.bin in the same order.
    """
    src_db = DATA / "docs_1m.db"
    dst_db = DATA / "docs.db"

    src = sqlite3.connect(src_db)
    dst = sqlite3.connect(dst_db)

    dst.execute("""
        CREATE TABLE documents (
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
        )
    """)
    dst.execute("CREATE INDEX idx_status ON documents (status)")

    placeholders = ",".join("?" * len(titles))
    rows = src.execute(
        f"SELECT article_id, chunk_index, title, categories, timestamp, chunk_text,"
        f" chunk_count, vector_offset, status, embedded_at, checksum"
        f" FROM documents WHERE title IN ({placeholders})"
        f" AND vector_offset IS NOT NULL ORDER BY vector_offset",
        titles,
    ).fetchall()
    src.close()

    print(f"  {len(rows):,} chunk rows to copy")

    original_offsets = []
    new_rows = []
    new_offset = 0
    for row in rows:
        (
            article_id,
            chunk_index,
            title,
            categories,
            timestamp,
            chunk_text,
            chunk_count,
            old_offset,
            status,
            embedded_at,
            checksum,
        ) = row
        original_offsets.append(old_offset)
        new_rows.append(
            (
                article_id,
                chunk_index,
                title,
                categories,
                timestamp,
                chunk_text,
                chunk_count,
                new_offset,
                status,
                embedded_at,
                checksum,
            )
        )
        new_offset += 1

    dst.executemany("INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?,?)", new_rows)
    dst.commit()
    dst.close()

    print(f"  docs.db written with offsets 0–{new_offset - 1:,}")
    return original_offsets


def build_100k_vectors(original_offsets: list[int]) -> None:
    src_bin = DATA / "vectors_1m.bin"
    dst_bin = DATA / "vectors.bin"

    total = src_bin.stat().st_size // (VECTOR_DIM * 4)
    print(
        f"  Reading {len(original_offsets):,} vectors from {src_bin.name}"
        f" ({total:,} total)..."
    )
    t0 = time.perf_counter()

    mmap = np.memmap(src_bin, dtype=np.float32, mode="r", shape=(total, VECTOR_DIM))
    vecs = mmap[original_offsets].copy()
    del mmap

    vecs.tofile(dst_bin)
    size_gb = dst_bin.stat().st_size / 1e9
    print(f"  vectors.bin written  ({size_gb:.2f} GB, {time.perf_counter() - t0:.0f}s)")


def rename_hnsw() -> None:
    src = DATA / "hnsw_100k.index"
    dst = DATA / "hnsw.index"
    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found — run build_production_index.py first"
        )
    shutil.copy2(src, dst)
    print(f"  hnsw_100k.index → hnsw.index  ({dst.stat().st_size / 1e9:.2f} GB)")


def main() -> None:
    print("\n── Step 1: rename 1M files ──")
    rename_1m_files()

    print("\n── Step 2: load 100K title list ──")
    titles = load_title_list()

    print("\n── Step 3: build 100K docs.db ──")
    original_offsets = build_100k_db(titles)

    print("\n── Step 4: build 100K vectors.bin ──")
    build_100k_vectors(original_offsets)

    print("\n── Step 5: install 100K HNSW index ──")
    rename_hnsw()

    print(
        "\nDone. Start the server with default paths"
        " (no RAG_HNSW_PATH override needed)."
    )


if __name__ == "__main__":
    main()

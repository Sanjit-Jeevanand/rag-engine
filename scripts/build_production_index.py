"""
Build a production HNSW index from the top N articles by incoming_links.

Steps:
  1. Scan the Wikipedia dump to collect incoming_links for every title in the DB
  2. Select the top N titles by score
  3. Load their chunk vectors from vectors.bin via memmap (no full file load)
  4. Build and save the HNSW index

No re-embedding required — all vectors already exist in vectors.bin.

Usage:
    uv run python scripts/build_production_index.py --limit 100000
"""

import argparse
import json
import sqlite3
import time
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm

from rag_engine.ingest.parser import parse_snapshot

SNAPSHOT = Path("data/wiki-dump.json.gz")
DB = Path("data/docs.db")
VECTORS_PATH = Path("data/vectors.bin")
VECTOR_DIM = 384
HNSW_M = 32
EF_CONSTRUCTION = 200
EF_SEARCH = 64


def collect_scores(limit: int) -> list[str]:
    """Pass 1: scan dump, collect incoming_links for DB titles, return top-N."""
    conn = sqlite3.connect(DB)
    in_db: set[str] = {
        row[0] for row in conn.execute("SELECT DISTINCT title FROM documents")
    }
    conn.close()
    print(f"  {len(in_db):,} distinct titles in DB")

    scores: dict[str, int] = {}
    for article in tqdm(
        parse_snapshot(SNAPSHOT), unit=" articles", desc="Scanning dump"
    ):
        if article.title in in_db:
            scores[article.title] = article.incoming_links

    print(f"  Matched {len(scores):,} titles with incoming_links scores")
    top = sorted(scores, key=lambda t: scores[t], reverse=True)[:limit]
    cutoff = scores[top[-1]]
    print(f"  Top {limit:,} cutoff: incoming_links >= {cutoff:,}")
    return top


def load_vectors(titles: list[str]) -> np.ndarray:
    """Load vectors for the given titles from vectors.bin using memmap."""
    conn = sqlite3.connect(DB)
    placeholders = ",".join("?" * len(titles))
    rows = conn.execute(
        f"SELECT vector_offset FROM documents WHERE title IN ({placeholders})"
        " AND vector_offset IS NOT NULL ORDER BY vector_offset",
        titles,
    ).fetchall()
    conn.close()

    offsets = [row[0] for row in rows]
    print(f"  {len(offsets):,} chunk vectors to load")

    total_vectors = VECTORS_PATH.stat().st_size // (VECTOR_DIM * 4)
    mmap = np.memmap(
        VECTORS_PATH, dtype=np.float32, mode="r", shape=(total_vectors, VECTOR_DIM)
    )
    vecs = mmap[offsets].copy()
    del mmap
    return vecs


def build_index(vecs: np.ndarray, out_path: Path) -> None:
    print(f"  Building IndexHNSWFlat over {len(vecs):,} vectors...")
    t0 = time.perf_counter()
    index = faiss.IndexHNSWFlat(VECTOR_DIM, HNSW_M)
    index.hnsw.efConstruction = EF_CONSTRUCTION
    index.hnsw.efSearch = EF_SEARCH
    index.add(vecs)
    print(f"  Built in {time.perf_counter() - t0:.0f}s")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_path))
    print(f"  Saved to {out_path}  ({out_path.stat().st_size / 1e9:.2f} GB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=100_000,
        help="Number of top articles by incoming_links (default: 100000)",
    )
    args = parser.parse_args()

    out_path = Path(f"data/hnsw_{args.limit // 1000}k.index")

    print(f"\n── Step 1: collect top {args.limit:,} titles from dump ──")
    titles = collect_scores(args.limit)

    scores_path = Path(f"data/title_scores_{args.limit // 1000}k.json")
    scores_path.write_text(json.dumps(titles))
    print(f"  Title list saved to {scores_path}")

    print(f"\n── Step 2: load vectors from {VECTORS_PATH} ──")
    vecs = load_vectors(titles)

    print("\n── Step 3: build HNSW index ──")
    build_index(vecs, out_path)

    print("\nDone. Run build_100k_dataset.py to install as production files.")


if __name__ == "__main__":
    main()

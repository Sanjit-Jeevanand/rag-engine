"""
Build and persist the HNSW index to disk.

Reads vectors.bin once, builds IndexHNSWFlat with the parameters chosen
from the Phase 3 benchmark (M=32, efSearch=64 — 98.6% recall, 0.39ms p50),
and writes the result to data/hnsw.index.

After this runs once, eval/index.py loads from the saved file instead of
rebuilding from 13.5 GB of vectors every startup (~30s -> ~5s).

Run with:
    uv run python scripts/build_hnsw_index.py
"""

import time
from pathlib import Path

import faiss
import numpy as np

VECTORS_PATH = Path("data/vectors.bin")
INDEX_PATH = Path("data/hnsw.index")
VECTOR_DIM = 384
HNSW_M = 32  # neighbours per node — chosen in Phase 3 benchmark
EF_CONSTRUCTION = 200  # graph build quality — higher = better graph, slower build
EF_SEARCH = 64  # query-time exploration — 98.6% recall at 0.39ms p50


def main() -> None:
    print(f"Loading vectors from {VECTORS_PATH}...")
    vecs = np.fromfile(VECTORS_PATH, dtype=np.float32).reshape(-1, VECTOR_DIM)
    print(f"  {len(vecs):,} vectors  ({VECTORS_PATH.stat().st_size / 1e9:.1f} GB)")

    print(f"\nBuilding IndexHNSWFlat (M={HNSW_M}, efConstruction={EF_CONSTRUCTION})...")
    t0 = time.perf_counter()
    index = faiss.IndexHNSWFlat(VECTOR_DIM, HNSW_M)
    index.hnsw.efConstruction = EF_CONSTRUCTION
    index.hnsw.efSearch = EF_SEARCH
    index.add(vecs)
    build_s = time.perf_counter() - t0
    print(f"  built in {build_s:.0f}s  — {index.ntotal:,} vectors")

    print(f"\nSaving to {INDEX_PATH}...")
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    size_gb = INDEX_PATH.stat().st_size / 1e9
    print(f"  saved  ({size_gb:.1f} GB on disk)")

    print("\nDone. eval/index.py will now load from this file instead of rebuilding.")


if __name__ == "__main__":
    main()

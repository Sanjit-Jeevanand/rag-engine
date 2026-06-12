"""
Phase 4 — py-spy profiling target.

Runs a sustained single-vector query loop so py-spy can sample the call
stack and produce a flamegraph.  No timing, no tqdm — just the raw hot
path that we want to see in the flamegraph.

Run under py-spy:
    py-spy record --output flamegraph.svg -- \
        uv run python scripts/profile_target.py

Open flamegraph.svg in any browser.
"""

from pathlib import Path

import faiss
import numpy as np

VECTORS_PATH = Path("data/vectors.bin")
HNSW_PATH = Path("data/hnsw.index")
VECTOR_DIM = 384
N_QUERIES = 500  # pool size to cycle through
N_ITERATIONS = 5_000  # total search() calls — enough for py-spy to sample
K = 10
SEED = 42


def main() -> None:
    faiss.omp_set_num_threads(1)

    print("Loading vectors...")
    vecs = np.fromfile(VECTORS_PATH, dtype=np.float32).reshape(-1, VECTOR_DIM)

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(vecs), size=N_QUERIES, replace=False)
    queries = vecs[idx]
    del vecs

    print("Loading HNSW index...")
    index = faiss.read_index(str(HNSW_PATH))
    index.hnsw.efSearch = 64  # type: ignore[attr-defined]

    print(f"Profiling: {N_ITERATIONS} single-vector search() calls...")
    for i in range(N_ITERATIONS):
        q = queries[i % N_QUERIES : i % N_QUERIES + 1]  # shape (1, 384)
        index.search(q, K)

    print("Done.")


if __name__ == "__main__":
    main()

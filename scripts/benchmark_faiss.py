"""
Phase 3 — FAISS index benchmark harness.

Compares index types on recall@10 vs exact search and P99 latency.
Each index is benchmarked with the same 1,000 query vectors sampled
from the corpus itself (standard FAISS benchmarking practice).

Run with:
    PYTHONPATH=src:. uv run python scripts/benchmark_faiss.py
"""

import time
from pathlib import Path

import faiss
import numpy as np

VECTORS_PATH = Path("data/vectors.bin")
VECTOR_DIM = 384
N_QUERIES = 1_000  # queries to benchmark with
K = 10  # top-K results per query
SEED = 42


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_vectors(path: Path) -> np.ndarray:
    """Load the full flat binary vector file into a float32 matrix."""
    vecs = np.fromfile(path, dtype=np.float32).reshape(-1, VECTOR_DIM)
    print(f"Loaded {len(vecs):,} vectors  ({path.stat().st_size / 1e9:.1f} GB)")
    return vecs


def sample_queries(vectors: np.ndarray, n: int, seed: int) -> np.ndarray:
    """
    Pick n random rows from the corpus as benchmark queries.

    Using corpus vectors as queries is standard practice:
    we already know the exact ground-truth neighbours for them
    (because we compute it with IndexFlatL2), so we can measure
    recall precisely.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(vectors), size=n, replace=False)
    return vectors[idx]


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------


def build_flat_l2(vectors: np.ndarray) -> faiss.IndexFlatL2:
    """
    Exact brute-force index.  Computes true L2 nearest neighbours by
    scanning every vector — no approximation, no compression.
    Used as ground truth: every other index's recall is measured
    against what THIS index returns.
    """
    index = faiss.IndexFlatL2(VECTOR_DIM)
    index.add(vectors)
    print(f"IndexFlatL2 built  — {index.ntotal:,} vectors")
    return index


def compute_ground_truth(
    index: faiss.IndexFlatL2,
    queries: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Run all queries against the exact index.
    Returns shape (N_QUERIES, k) — the row indices of the true top-k
    neighbours for each query.
    """
    _, indices = index.search(queries, k)
    return indices  # shape (N_QUERIES, K)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def benchmark(
    index: faiss.Index,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    k: int,
    label: str,
) -> dict[str, float]:
    """
    Run all queries through `index`, record latency per query,
    compare results against `ground_truth` to compute recall@k.

    recall@10 here means: of the 10 true nearest neighbours,
    how many did the approximate index find?
    This is ANNS recall — different from retrieval recall in eval/.
    """
    n = len(queries)
    latencies: list[float] = []
    recall_scores: list[float] = []

    for i in range(n):
        q = queries[i : i + 1]  # shape (1, 384) — single query

        t0 = time.perf_counter()
        _, retrieved = index.search(q, k)
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

        # recall: fraction of ground-truth neighbours found
        true_set = set(ground_truth[i].tolist())
        found = len(true_set & set(retrieved[0].tolist()))
        recall_scores.append(found / k)

    latencies_arr = np.array(latencies)
    result = {
        "label": label,
        "recall_at_10": round(float(np.mean(recall_scores)), 4),
        "p50_ms": round(float(np.percentile(latencies_arr, 50)), 3),
        "p99_ms": round(float(np.percentile(latencies_arr, 99)), 3),
        "mean_ms": round(float(np.mean(latencies_arr)), 3),
    }

    print(
        f"\n{label}\n"
        f"  recall@10 : {result['recall_at_10']:.4f}\n"
        f"  mean_ms   : {result['mean_ms']:.3f}\n"
        f"  p50_ms    : {result['p50_ms']:.3f}\n"
        f"  p99_ms    : {result['p99_ms']:.3f}"
    )
    return result


# ---------------------------------------------------------------------------
# IndexHNSWFlat
# ---------------------------------------------------------------------------


def build_hnsw(vectors: np.ndarray, ef_construction: int = 200) -> faiss.IndexHNSWFlat:
    """
    Graph-based ANN index.

    M=32: each node connects to 32 neighbours in the graph.
    Higher M = better recall + more memory.  32 is the standard default.

    ef_construction: how many candidates to explore while BUILDING the graph.
    Higher = better graph quality, slower build.  200 is the standard default.

    ef_search (set separately): how many nodes to explore at QUERY time.
    This is the recall-speed knob we sweep below.
    """
    index = faiss.IndexHNSWFlat(VECTOR_DIM, 32)  # 32 neighbours per node
    index.hnsw.efConstruction = ef_construction
    index.add(vectors)
    print(
        f"IndexHNSWFlat built  — {index.ntotal:,} vectors  "
        f"(M=32, efC={ef_construction})"
    )
    return index


# ---------------------------------------------------------------------------
# IndexIVFPQ
# ---------------------------------------------------------------------------


def build_ivfpq(
    vectors: np.ndarray,
    nlist: int = 4096,
    m_pq: int = 48,
    nbits: int = 8,
) -> faiss.IndexIVFPQ:
    """
    Inverted-file index with product quantization compression.

    nlist=4096: number of Voronoi cells (clusters).  Rule of thumb: sqrt(N).
                For 8.8M vectors, sqrt ≈ 2966; 4096 is the next power-of-two up.

    m_pq=48:    split each 384-dim vector into 48 sub-vectors of 8 dims each.
                Each sub-vector is compressed to `nbits` bits.
                Memory per vector = m_pq * nbits / 8 bytes = 48 bytes  (vs 1536 raw).
                384 must be divisible by m_pq — 384 / 48 = 8. ✓

    nbits=8:    256 centroids per sub-quantizer (standard).

    Must be trained on a representative sample before adding vectors.
    """
    quantizer = faiss.IndexFlatL2(VECTOR_DIM)  # coarse quantizer for cluster centres
    index = faiss.IndexIVFPQ(quantizer, VECTOR_DIM, nlist, m_pq, nbits)

    print(f"Training IndexIVFPQ (nlist={nlist}, M={m_pq}, nbits={nbits})...")
    # train on a random 500K sample — enough for k-means to converge
    rng = np.random.default_rng(SEED)
    train_idx = rng.choice(len(vectors), size=min(500_000, len(vectors)), replace=False)
    index.train(vectors[train_idx])

    index.add(vectors)
    mem_gb = index.ntotal * m_pq * nbits / 8 / 1e9
    raw_gb = index.ntotal * VECTOR_DIM * 4 / 1e9
    print(
        f"IndexIVFPQ built  — {index.ntotal:,} vectors  "
        f"(~{mem_gb:.1f} GB compressed vs {raw_gb:.1f} GB raw)"
    )
    return index


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 55)
    print("Phase 3 — FAISS Benchmark")
    print("=" * 55)

    vectors = load_vectors(VECTORS_PATH)
    queries = sample_queries(vectors, N_QUERIES, SEED)
    print(f"Query set: {len(queries):,} vectors sampled from corpus\n")

    # Ground truth — exact search
    flat = build_flat_l2(vectors)
    print("\nComputing ground truth (exact search)...")
    ground_truth = compute_ground_truth(flat, queries, K)

    results = []
    results.append(benchmark(flat, queries, ground_truth, K, "IndexFlatL2 (exact)"))

    # ── HNSW: sweep ef_search values ────────────────────────────────────────
    print("\n\nBuilding HNSW index (this takes a few minutes)...")
    hnsw = build_hnsw(vectors)

    for ef in [32, 64, 128, 256]:
        hnsw.hnsw.efSearch = ef  # set the search-time exploration budget
        results.append(
            benchmark(hnsw, queries, ground_truth, K, f"IndexHNSWFlat ef={ef}")
        )

    # ── IVFPQ: sweep nprobe values ───────────────────────────────────────────
    print("\n\nBuilding IVFPQ index (training + adding vectors)...")
    ivfpq = build_ivfpq(vectors)

    for nprobe in [8, 32, 64, 128]:
        ivfpq.nprobe = nprobe  # how many clusters to search per query
        results.append(
            benchmark(ivfpq, queries, ground_truth, K, f"IndexIVFPQ nprobe={nprobe}")
        )

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n\n" + "=" * 55)
    print(f"{'Index':<30} {'recall@10':>10} {'p50 ms':>8} {'p99 ms':>8}")
    print("-" * 55)
    for r in results:
        print(
            f"{r['label']:<30} {r['recall_at_10']:>10.4f}"
            f" {r['p50_ms']:>8.3f} {r['p99_ms']:>8.3f}"
        )


if __name__ == "__main__":
    main()

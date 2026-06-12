import time
from pathlib import Path

import faiss
import numpy as np

VECTORS_PATH = Path("data/vectors.bin")
VECTOR_DIM = 384
N_QUERIES = 1_000
K = 10
SEED = 42


def load_vectors(path: Path) -> np.ndarray:
    vecs = np.fromfile(path, dtype=np.float32).reshape(-1, VECTOR_DIM)
    print(f"Loaded {len(vecs):,} vectors  ({path.stat().st_size / 1e9:.1f} GB)")
    return vecs


def sample_queries(vectors: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(vectors), size=n, replace=False)
    return vectors[idx]


def build_flat_l2(vectors: np.ndarray) -> faiss.IndexFlatL2:
    index = faiss.IndexFlatL2(VECTOR_DIM)
    index.add(vectors)
    print(f"IndexFlatL2 built  — {index.ntotal:,} vectors")
    return index


def benchmark(
    index: faiss.Index,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    k: int,
    label: str,
) -> dict[str, float]:
    latencies: list[float] = []
    recall_scores: list[float] = []

    for i in range(len(queries)):
        q = queries[i : i + 1]  # shape (1, 384) — FAISS requires 2-D input
        t0 = time.perf_counter()
        _, retrieved = index.search(q, k)
        latencies.append((time.perf_counter() - t0) * 1000)
        true_set = set(ground_truth[i].tolist())
        recall_scores.append(len(true_set & set(retrieved[0].tolist())) / k)

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


def build_hnsw(vectors: np.ndarray, ef_construction: int = 200) -> faiss.IndexHNSWFlat:
    index = faiss.IndexHNSWFlat(VECTOR_DIM, 32)
    index.hnsw.efConstruction = ef_construction
    index.add(vectors)
    print(
        f"IndexHNSWFlat built  — {index.ntotal:,} vectors  "
        f"(M=32, efC={ef_construction})"
    )
    return index


def build_ivfpq(
    vectors: np.ndarray,
    nlist: int = 4096,
    m_pq: int = 48,
    nbits: int = 8,
) -> faiss.IndexIVFPQ:
    quantizer = faiss.IndexFlatL2(VECTOR_DIM)  # coarse quantizer for cluster centres
    index = faiss.IndexIVFPQ(quantizer, VECTOR_DIM, nlist, m_pq, nbits)

    print(f"Training IndexIVFPQ (nlist={nlist}, M={m_pq}, nbits={nbits})...")
    rng = np.random.default_rng(SEED)
    train_idx = rng.choice(len(vectors), size=min(500_000, len(vectors)), replace=False)
    index.train(vectors[train_idx])  # 500K sample is enough for k-means to converge

    index.add(vectors)
    mem_gb = index.ntotal * m_pq * nbits / 8 / 1e9
    raw_gb = index.ntotal * VECTOR_DIM * 4 / 1e9
    print(
        f"IndexIVFPQ built  — {index.ntotal:,} vectors  "
        f"(~{mem_gb:.1f} GB compressed vs {raw_gb:.1f} GB raw)"
    )
    return index


def main() -> None:
    print("=" * 55)
    print("Phase 3 — FAISS Benchmark")
    print("=" * 55)

    vectors = load_vectors(VECTORS_PATH)
    queries = sample_queries(vectors, N_QUERIES, SEED)
    print(f"Query set: {len(queries):,} vectors sampled from corpus\n")

    flat = build_flat_l2(vectors)
    print("\nComputing ground truth (exact search)...")
    _, ground_truth = flat.search(queries, K)

    results = []
    results.append(benchmark(flat, queries, ground_truth, K, "IndexFlatL2 (exact)"))

    print("\n\nBuilding HNSW index (this takes a few minutes)...")
    hnsw = build_hnsw(vectors)
    for ef in [32, 64, 128, 256]:
        hnsw.hnsw.efSearch = ef
        results.append(
            benchmark(hnsw, queries, ground_truth, K, f"IndexHNSWFlat ef={ef}")
        )

    print("\n\nBuilding IVFPQ index (training + adding vectors)...")
    ivfpq = build_ivfpq(vectors)
    for nprobe in [8, 32, 64, 128]:
        ivfpq.nprobe = nprobe
        results.append(
            benchmark(ivfpq, queries, ground_truth, K, f"IndexIVFPQ nprobe={nprobe}")
        )

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

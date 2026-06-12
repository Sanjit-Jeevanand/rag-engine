import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm

VECTORS_PATH = Path("data/vectors.bin")
HNSW_PATH = Path("data/hnsw.index")
IVFPQ_PATH = Path("data/ivfpq.index")
VECTOR_DIM = 384
NLIST = 4096
M_PQ = 48
NBITS = 8
K = 10
SEED = 42
DURATION = 10
N_QUERY_POOL = 1_000
CONCURRENCY_LEVELS = [1, 8]

# Recall@10 values from Phase 3 benchmark (exact ground truth)
PHASE3_RECALL = {
    8: 0.6374,
    32: 0.6631,
    64: 0.6762,
    128: 0.6878,
}
HNSW_RECALL = 0.9857  # ef=64


def load_vectors(path: Path) -> np.ndarray:
    print(f"Loading vectors ({path.stat().st_size / 1e9:.1f} GB)...")
    vecs = np.fromfile(path, dtype=np.float32).reshape(-1, VECTOR_DIM)
    print(f"  {len(vecs):,} vectors")
    return vecs


def build_and_save_ivfpq(vecs: np.ndarray, path: Path) -> faiss.IndexIVFPQ:
    print(f"Building IndexIVFPQ (nlist={NLIST}, M={M_PQ}, nbits={NBITS})...")
    quantizer = faiss.IndexFlatL2(VECTOR_DIM)
    index = faiss.IndexIVFPQ(quantizer, VECTOR_DIM, NLIST, M_PQ, NBITS)

    print("  Training on 500K sample...")
    rng = np.random.default_rng(SEED)
    train_idx = rng.choice(len(vecs), size=500_000, replace=False)
    index.train(vecs[train_idx])

    print(f"  Adding {len(vecs):,} vectors...")
    index.add(vecs)

    faiss.write_index(index, str(path))
    size_gb = path.stat().st_size / 1e9
    raw_gb = len(vecs) * VECTOR_DIM * 4 / 1e9
    ratio = round(raw_gb / size_gb)
    print(f"  Saved: {size_gb:.2f} GB  (vs {raw_gb:.1f} GB raw — {ratio}× compression)")
    return index  # type: ignore[return-value]


def load_ivfpq(path: Path) -> faiss.IndexIVFPQ:
    print(f"Loading IVFPQ index from {path} ({path.stat().st_size / 1e9:.2f} GB)...")
    index = faiss.read_index(str(path))
    print(f"  {index.ntotal:,} vectors")
    return index  # type: ignore[return-value]


def load_hnsw(path: Path) -> faiss.IndexHNSWFlat:
    print(f"Loading HNSW index from {path} ({path.stat().st_size / 1e9:.1f} GB)...")
    index = faiss.read_index(str(path))
    index.hnsw.efSearch = 64  # type: ignore[attr-defined]
    print(f"  {index.ntotal:,} vectors, efSearch=64")
    return index  # type: ignore[return-value]


def sample_queries(vecs: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return vecs[rng.choice(len(vecs), size=n, replace=False)]


def _worker(
    index: faiss.Index,
    queries: np.ndarray,
    duration: float,
    worker_id: int,
    bar: tqdm,
) -> list[float]:
    latencies: list[float] = []
    n = len(queries)
    i = worker_id
    deadline = time.perf_counter() + duration

    while time.perf_counter() < deadline:
        q = queries[i % n : i % n + 1]
        t0 = time.perf_counter()
        index.search(q, K)
        latencies.append((time.perf_counter() - t0) * 1000)
        bar.update(1)
        i += 1

    return latencies


def measure(
    index: faiss.Index,
    queries: np.ndarray,
    concurrency: int,
    label: str,
) -> dict:
    all_latencies: list[float] = []
    t_start = time.perf_counter()
    with (
        tqdm(desc=f"  {label}", unit="q", dynamic_ncols=True) as bar,
        ThreadPoolExecutor(max_workers=concurrency) as pool,
    ):
        futures = [
            pool.submit(_worker, index, queries, DURATION, wid, bar)
            for wid in range(concurrency)
        ]
        for fut in as_completed(futures):
            all_latencies.extend(fut.result())
    elapsed = time.perf_counter() - t_start

    arr = np.array(all_latencies)
    return {
        "label": label,
        "concurrency": concurrency,
        "qps": round(len(arr) / elapsed, 1),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
    }


def main() -> None:
    faiss.omp_set_num_threads(1)

    print("=" * 68)
    print("Phase 4 — IVFPQ vs HNSW Throughput Comparison")
    print("=" * 68)
    print()

    vecs = load_vectors(VECTORS_PATH)
    queries = sample_queries(vecs, N_QUERY_POOL, SEED)

    if IVFPQ_PATH.exists():
        ivfpq = load_ivfpq(IVFPQ_PATH)
    else:
        ivfpq = build_and_save_ivfpq(vecs, IVFPQ_PATH)

    hnsw = load_hnsw(HNSW_PATH)
    del vecs
    print()

    results = []

    print("--- IndexIVFPQ ---")
    for nprobe in [8, 32, 64, 128]:
        ivfpq.nprobe = nprobe
        for c in CONCURRENCY_LEVELS:
            label = f"IVFPQ nprobe={nprobe:<3} c={c}"
            results.append({**measure(ivfpq, queries, c, label), "nprobe": nprobe})

    print()
    print("--- IndexHNSWFlat (ef=64) reference ---")
    for c in CONCURRENCY_LEVELS:
        label = f"HNSW ef=64          c={c}"
        results.append({**measure(hnsw, queries, c, label), "nprobe": None})

    ivfpq_gb = IVFPQ_PATH.stat().st_size / 1e9
    hnsw_gb = HNSW_PATH.stat().st_size / 1e9

    hdr = f"{'Index':<20} c  {'recall':>7}  {'mem':>5}  {'QPS':>8}  p50     p99"
    print("\n\n" + "=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        if r["nprobe"] is not None:
            recall = PHASE3_RECALL.get(r["nprobe"], 0.0)
            mem = ivfpq_gb
        else:
            recall = HNSW_RECALL
            mem = hnsw_gb
        print(
            f"{r['label']:<20} {r['concurrency']:>2}"
            f"  {recall:>7.4f}  {mem:>5.2f}"
            f"  {r['qps']:>8.1f}  {r['p50_ms']:>6.3f}  {r['p99_ms']:>6.3f}"
        )

    print()
    print("Verdict:")
    hnsw_c8 = next(r for r in results if r["nprobe"] is None and r["concurrency"] == 8)
    ivfpq_rows = [r for r in results if r["nprobe"] is not None]
    best = max(ivfpq_rows, key=lambda r: r["qps"])
    best_recall = PHASE3_RECALL.get(best["nprobe"], 0.0)
    print(
        f"  HNSW  ef=64  c=8    → {hnsw_c8['qps']:>8,.0f} QPS"
        f"  recall={HNSW_RECALL:.4f}  {hnsw_gb:.1f} GB"
    )
    print(
        f"  IVFPQ nprobe={best['nprobe']}"
        f" c={best['concurrency']}"
        f"  → {best['qps']:>8,.0f} QPS"
        f"  recall={best_recall:.4f}"
        f"  {ivfpq_gb:.2f} GB"
    )


if __name__ == "__main__":
    main()

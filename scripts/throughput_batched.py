import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm

VECTORS_PATH = Path("data/vectors.bin")
HNSW_PATH = Path("data/hnsw.index")
VECTOR_DIM = 384
N_QUERY_POOL = 1_000
K = 10
SEED = 42
DURATION = 10
BATCH_SIZES = [1, 8, 32, 64, 128, 256, 512]

BASELINE_QPS = 918.6  # single-vector c=1 from throughput_baseline.py


def load_query_pool(path: Path, n: int, seed: int) -> np.ndarray:
    vecs = np.fromfile(path, dtype=np.float32).reshape(-1, VECTOR_DIM)
    rng = np.random.default_rng(seed)
    queries = vecs[rng.choice(len(vecs), size=n, replace=False)]
    print(f"  {len(vecs):,} vectors loaded; {n} queries sampled")
    return queries


def load_hnsw(path: Path) -> faiss.IndexHNSWFlat:
    print(f"Loading HNSW index ({path.stat().st_size / 1e9:.1f} GB)...")
    index = faiss.read_index(str(path))
    index.hnsw.efSearch = 64  # type: ignore[attr-defined]
    print(f"  {index.ntotal:,} vectors, efSearch=64")
    return index  # type: ignore[return-value]


def _worker(
    index: faiss.Index,
    queries: np.ndarray,
    duration: float,
    worker_id: int,
    batch_size: int,
    bar: tqdm,
) -> list[float]:
    latencies: list[float] = []
    n = len(queries)
    i = worker_id
    deadline = time.perf_counter() + duration

    while time.perf_counter() < deadline:
        # Stack batch_size rows into one matrix — no copy if contiguous
        start = i % n
        end = start + batch_size
        if end <= n:
            batch = queries[start:end]
        else:
            # wrap around the query pool
            batch = np.concatenate([queries[start:], queries[: end - n]])

        t0 = time.perf_counter()
        index.search(batch, K)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        per_query_ms = elapsed_ms / batch_size
        latencies.extend([per_query_ms] * batch_size)
        bar.update(batch_size)
        i += batch_size

    return latencies


def measure(
    index: faiss.Index,
    queries: np.ndarray,
    concurrency: int,
    batch_size: int,
    label: str,
) -> dict:
    all_latencies: list[float] = []

    t_start = time.perf_counter()
    with (
        tqdm(desc=f"  {label}", unit="q", dynamic_ncols=True) as bar,
        ThreadPoolExecutor(max_workers=concurrency) as pool,
    ):
        futures = [
            pool.submit(_worker, index, queries, DURATION, wid, batch_size, bar)
            for wid in range(concurrency)
        ]
        for fut in as_completed(futures):
            all_latencies.extend(fut.result())
    elapsed = time.perf_counter() - t_start

    arr = np.array(all_latencies)
    qps = len(arr) / elapsed
    return {
        "label": label,
        "batch_size": batch_size,
        "concurrency": concurrency,
        "qps": round(qps, 1),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "speedup": round(qps / BASELINE_QPS, 1),
    }


def main() -> None:
    faiss.omp_set_num_threads(1)

    print("=" * 68)
    print("Phase 4 — Batched Search Throughput")
    print("=" * 68)
    print(f"Baseline (single-vector c=1): {BASELINE_QPS:,.1f} QPS\n")

    print(f"Loading vectors from {VECTORS_PATH}...")
    queries = load_query_pool(VECTORS_PATH, N_QUERY_POOL, SEED)
    print()
    index = load_hnsw(HNSW_PATH)
    print()

    results = []

    print("--- Batch size sweep (concurrency=1) ---")
    for bs in BATCH_SIZES:
        results.append(measure(index, queries, 1, bs, f"batch={bs:<4} c=1"))

    c1_results = [r for r in results if r["concurrency"] == 1]
    best_bs = max(c1_results, key=lambda r: r["qps"])["batch_size"]

    print(f"\n--- Best batch size ({best_bs}) at higher concurrency ---")
    for c in [4, 8]:
        results.append(measure(index, queries, c, best_bs, f"batch={best_bs:<4} c={c}"))

    sep = "=" * 65
    print(f"\n\n{sep}")
    cols = f"{'Label':<20} {'batch':>5} {'c':>2}  {'QPS':>8}  {'p50':>7}  {'x':>7}"
    print(cols)
    print("-" * 65)
    for r in results:
        best_speedup = max(x["speedup"] for x in results)
        marker = " ←" if r["speedup"] == best_speedup else ""
        print(
            f"{r['label']:<20} {r['batch_size']:>5} {r['concurrency']:>2}"
            f"  {r['qps']:>8.1f}  {r['p50_ms']:>7.3f}"
            f"  {r['speedup']:>6.1f}×{marker}"
        )

    best = max(results, key=lambda r: r["qps"])
    print(
        f"\nBest: {best['label'].strip()}  →  {best['qps']:,.0f} QPS"
        f"  ({best['speedup']}× over single-vector baseline)"
    )


if __name__ == "__main__":
    main()

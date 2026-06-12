"""
Phase 4 — Concurrency baseline (before any batching optimisation).

Measures raw FAISS search throughput at concurrency 1, 4, and 8 using
ThreadPoolExecutor.  No model encoder — queries are pre-sampled corpus
vectors so we isolate the FAISS bottleneck from encoding overhead.

Each (index, concurrency) combination runs for DURATION seconds with
workers firing single-vector search() calls in a tight loop.  We collect
per-query latencies, then report QPS and P99.

This is the "before" snapshot for Phase 4.  Re-run after batching to
measure the improvement multiplier.

Run with:
    PYTHONPATH=src:. uv run python scripts/throughput_baseline.py
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm

VECTORS_PATH = Path("data/vectors.bin")
HNSW_PATH = Path("data/hnsw.index")
VECTOR_DIM = 384
N_QUERY_POOL = 1_000  # pool of pre-sampled query vectors to cycle through
K = 10
SEED = 42
DURATION = 10  # seconds per (index, concurrency) combination
CONCURRENCY_LEVELS = [1, 4, 8, 10, 12, 16]


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def load_vectors(path: Path) -> np.ndarray:
    print(f"Loading vectors from {path} ({path.stat().st_size / 1e9:.1f} GB)...")
    vecs = np.fromfile(path, dtype=np.float32).reshape(-1, VECTOR_DIM)
    print(f"  {len(vecs):,} vectors loaded")
    return vecs


def sample_query_pool(vecs: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(vecs), size=n, replace=False)
    print(f"  Sampled {n} query vectors from corpus")
    return vecs[idx]


def load_hnsw(path: Path) -> faiss.IndexHNSWFlat:
    print(f"Loading HNSW index from {path} ({path.stat().st_size / 1e9:.1f} GB)...")
    index = faiss.read_index(str(path))
    index.hnsw.efSearch = 64
    print(f"  HNSW loaded ({index.ntotal:,} vectors, efSearch=64)")
    return index  # type: ignore[return-value]


def build_flat(vecs: np.ndarray) -> faiss.IndexFlatIP:
    print("Building IndexFlatIP...")
    index = faiss.IndexFlatIP(VECTOR_DIM)
    index.add(vecs)
    print(f"  FlatIP built ({index.ntotal:,} vectors)")
    return index


# ---------------------------------------------------------------------------
# Single-worker loop: fire search() calls as fast as possible for DURATION s
# ---------------------------------------------------------------------------


def _worker(
    index: faiss.Index,
    queries: np.ndarray,
    duration: float,
    worker_id: int,
    bar: tqdm,
) -> list[float]:
    """
    Run single-vector search() calls in a loop for `duration` seconds.
    Returns a list of per-query latencies in milliseconds.
    """
    latencies: list[float] = []
    n = len(queries)
    i = worker_id  # stagger start position so workers hit different queries
    deadline = time.perf_counter() + duration

    while time.perf_counter() < deadline:
        q = queries[i % n : i % n + 1]  # shape (1, 384)
        t0 = time.perf_counter()
        index.search(q, K)
        latencies.append((time.perf_counter() - t0) * 1000)
        bar.update(1)
        i += 1

    return latencies


# ---------------------------------------------------------------------------
# Run one (index, concurrency) combination
# ---------------------------------------------------------------------------


def measure(
    index: faiss.Index,
    queries: np.ndarray,
    concurrency: int,
    label: str,
) -> dict:
    """
    Spin up `concurrency` threads, each calling search() in a loop for
    DURATION seconds.  Aggregate all latencies to compute QPS and P99.

    QPS = total queries completed / elapsed wall-clock time
    P99 = 99th percentile of individual query latencies
    """
    all_latencies: list[float] = []

    bar_desc = f"  {label} c={concurrency}"
    t_start = time.perf_counter()
    with (
        tqdm(desc=bar_desc, unit="q", dynamic_ncols=True) as bar,
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
    qps = len(arr) / elapsed
    result = {
        "label": label,
        "concurrency": concurrency,
        "qps": round(qps, 1),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "total_queries": len(arr),
    }

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    faiss.omp_set_num_threads(1)  # 1 OMP thread per search — fair baseline
    # (at concurrency>1, letting OMP use multiple threads too would cause
    #  thread oversubscription and artificially depress per-query latency)

    print("=" * 65)
    print("Phase 4 — Concurrency Baseline  (single-vector, no batching)")
    print("=" * 65)
    print(f"Duration per run: {DURATION}s  |  K={K}  |  OMP threads=1\n")

    vecs = load_vectors(VECTORS_PATH)
    queries = sample_query_pool(vecs, N_QUERY_POOL, SEED)
    print()

    hnsw = load_hnsw(HNSW_PATH)
    flat = build_flat(vecs)
    del vecs  # free RAM — indexes hold their own copy
    print()

    results = []

    print("--- IndexHNSWFlat (ef=64) ---")
    for c in CONCURRENCY_LEVELS:
        results.append(measure(hnsw, queries, c, "HNSW ef=64"))

    print()
    print("--- IndexFlatIP (exact, c=1 only — memory-bandwidth-bound) ---")
    results.append(measure(flat, queries, 1, "FlatIP exact"))

    # Summary table
    print("\n\n" + "=" * 65)
    print(f"{'Index':<22} {'c':>3}  {'QPS':>8}  {'p50 ms':>8}  {'p99 ms':>8}")
    print("-" * 65)
    for r in results:
        print(
            f"{r['label']:<22} {r['concurrency']:>3}"
            f"  {r['qps']:>8.1f}"
            f"  {r['p50_ms']:>8.3f}"
            f"  {r['p99_ms']:>8.3f}"
        )
    print()
    print("Next step: profile with py-spy, then batch queries.")


if __name__ == "__main__":
    main()

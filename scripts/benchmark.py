"""
Unified benchmark runner.

Usage:
  python scripts/benchmark.py faiss-index
  python scripts/benchmark.py throughput-concurrency
  python scripts/benchmark.py throughput-batch
  python scripts/benchmark.py throughput-ivfpq
  python scripts/benchmark.py profile
"""

import argparse
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
K = 10
SEED = 42

# ── helpers ──────────────────────────────────────────────────────────────────


def _load_vectors(path: Path) -> np.ndarray:
    print(f"Loading vectors ({path.stat().st_size / 1e9:.1f} GB)...")
    vecs = np.fromfile(path, dtype=np.float32).reshape(-1, VECTOR_DIM)
    print(f"  {len(vecs):,} vectors")
    return vecs


def _sample(vecs: np.ndarray, n: int, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return vecs[rng.choice(len(vecs), size=n, replace=False)]


def _load_hnsw(path: Path, ef: int = 64) -> faiss.IndexHNSWFlat:
    print(f"Loading HNSW ({path.stat().st_size / 1e9:.1f} GB)...")
    index = faiss.read_index(str(path))
    index.hnsw.efSearch = ef  # type: ignore[attr-defined]
    print(f"  {index.ntotal:,} vectors, efSearch={ef}")
    return index  # type: ignore[return-value]


def _worker_single(
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


def _measure_concurrency(
    index: faiss.Index,
    queries: np.ndarray,
    concurrency: int,
    label: str,
    duration: float = 10.0,
) -> dict:
    all_lat: list[float] = []
    t0 = time.perf_counter()
    with (
        tqdm(desc=f"  {label} c={concurrency}", unit="q", dynamic_ncols=True) as bar,
        ThreadPoolExecutor(max_workers=concurrency) as pool,
    ):
        futs = [
            pool.submit(_worker_single, index, queries, duration, wid, bar)
            for wid in range(concurrency)
        ]
        for fut in as_completed(futs):
            all_lat.extend(fut.result())
    elapsed = time.perf_counter() - t0
    arr = np.array(all_lat)
    return {
        "label": label,
        "concurrency": concurrency,
        "qps": round(len(arr) / elapsed, 1),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "total_queries": len(arr),
    }


# ── faiss-index ───────────────────────────────────────────────────────────────


def cmd_faiss_index() -> None:
    N_QUERIES = 1_000

    def _benchmark(
        index: faiss.Index,
        queries: np.ndarray,
        ground_truth: np.ndarray,
        label: str,
    ) -> dict:
        latencies: list[float] = []
        recall_scores: list[float] = []
        for i in range(len(queries)):
            q = queries[i : i + 1]
            t0 = time.perf_counter()
            _, retrieved = index.search(q, K)
            latencies.append((time.perf_counter() - t0) * 1000)
            true_set = set(ground_truth[i].tolist())
            recall_scores.append(len(true_set & set(retrieved[0].tolist())) / K)
        arr = np.array(latencies)
        result = {
            "label": label,
            "recall_at_10": round(float(np.mean(recall_scores)), 4),
            "p50_ms": round(float(np.percentile(arr, 50)), 3),
            "p99_ms": round(float(np.percentile(arr, 99)), 3),
            "mean_ms": round(float(np.mean(arr)), 3),
        }
        print(
            f"\n{label}\n"
            f"  recall@10 : {result['recall_at_10']:.4f}\n"
            f"  mean_ms   : {result['mean_ms']:.3f}\n"
            f"  p50_ms    : {result['p50_ms']:.3f}\n"
            f"  p99_ms    : {result['p99_ms']:.3f}"
        )
        return result

    print("=" * 55)
    print("FAISS Index Benchmark")
    print("=" * 55)

    vectors = _load_vectors(VECTORS_PATH)
    queries = _sample(vectors, N_QUERIES)
    print(f"Query set: {N_QUERIES:,} vectors\n")

    flat = faiss.IndexFlatL2(VECTOR_DIM)
    flat.add(vectors)
    print(f"IndexFlatL2 built — {flat.ntotal:,} vectors")

    print("\nComputing ground truth (exact search)...")
    _, ground_truth = flat.search(queries, K)

    results = [_benchmark(flat, queries, ground_truth, "IndexFlatL2 (exact)")]

    print("\n\nBuilding HNSW (this takes a few minutes)...")
    hnsw = faiss.IndexHNSWFlat(VECTOR_DIM, 32)
    hnsw.hnsw.efConstruction = 200
    hnsw.add(vectors)
    print(f"IndexHNSWFlat built — {hnsw.ntotal:,} vectors (M=32, efC=200)")
    for ef in [32, 64, 128, 256]:
        hnsw.hnsw.efSearch = ef
        results.append(
            _benchmark(hnsw, queries, ground_truth, f"IndexHNSWFlat ef={ef}")
        )

    print("\n\nBuilding IVFPQ...")
    quantizer = faiss.IndexFlatL2(VECTOR_DIM)
    ivfpq = faiss.IndexIVFPQ(quantizer, VECTOR_DIM, 4096, 48, 8)
    train_idx = np.random.default_rng(SEED).choice(
        len(vectors), size=500_000, replace=False
    )
    ivfpq.train(vectors[train_idx])
    ivfpq.add(vectors)
    for nprobe in [8, 32, 64, 128]:
        ivfpq.nprobe = nprobe
        results.append(
            _benchmark(ivfpq, queries, ground_truth, f"IndexIVFPQ nprobe={nprobe}")
        )

    print("\n\n" + "=" * 55)
    print(f"{'Index':<30} {'recall@10':>10} {'p50 ms':>8} {'p99 ms':>8}")
    print("-" * 55)
    for r in results:
        print(
            f"{r['label']:<30} {r['recall_at_10']:>10.4f}"
            f" {r['p50_ms']:>8.3f} {r['p99_ms']:>8.3f}"
        )


# ── throughput-concurrency ────────────────────────────────────────────────────


def cmd_throughput_concurrency() -> None:
    CONCURRENCY_LEVELS = [1, 4, 8, 10, 12, 16]

    faiss.omp_set_num_threads(1)
    print("=" * 65)
    print("Concurrency Baseline  (single-vector, no batching)")
    print("=" * 65)
    print(f"Duration: 10s/run  |  K={K}  |  OMP=1\n")

    vecs = _load_vectors(VECTORS_PATH)
    queries = _sample(vecs, 1_000)

    hnsw = _load_hnsw(HNSW_PATH)
    flat = faiss.IndexFlatIP(VECTOR_DIM)
    flat.add(vecs)
    print(f"  FlatIP built ({flat.ntotal:,} vectors)")
    del vecs

    results = []
    print("\n--- IndexHNSWFlat (ef=64) ---")
    for c in CONCURRENCY_LEVELS:
        results.append(_measure_concurrency(hnsw, queries, c, "HNSW ef=64"))

    print("\n--- IndexFlatIP (c=1 only) ---")
    results.append(_measure_concurrency(flat, queries, 1, "FlatIP exact"))

    print("\n\n" + "=" * 65)
    print(f"{'Index':<22} {'c':>3}  {'QPS':>8}  {'p50 ms':>8}  {'p99 ms':>8}")
    print("-" * 65)
    for r in results:
        print(
            f"{r['label']:<22} {r['concurrency']:>3}"
            f"  {r['qps']:>8.1f}  {r['p50_ms']:>8.3f}  {r['p99_ms']:>8.3f}"
        )


# ── throughput-batch ──────────────────────────────────────────────────────────


def cmd_throughput_batch() -> None:
    BATCH_SIZES = [1, 8, 32, 64, 128, 256, 512]
    BASELINE_QPS = 918.6
    DURATION = 10.0

    def _worker_batch(
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
            start = i % n
            end = start + batch_size
            batch = (
                queries[start:end]
                if end <= n
                else np.concatenate([queries[start:], queries[: end - n]])
            )
            t0 = time.perf_counter()
            index.search(batch, K)
            ms = (time.perf_counter() - t0) * 1000 / batch_size
            latencies.extend([ms] * batch_size)
            bar.update(batch_size)
            i += batch_size
        return latencies

    def _measure_batch(
        index: faiss.Index,
        queries: np.ndarray,
        concurrency: int,
        batch_size: int,
        label: str,
    ) -> dict:
        all_lat: list[float] = []
        t0 = time.perf_counter()
        with (
            tqdm(desc=f"  {label}", unit="q", dynamic_ncols=True) as bar,
            ThreadPoolExecutor(max_workers=concurrency) as pool,
        ):
            futs = [
                pool.submit(
                    _worker_batch, index, queries, DURATION, wid, batch_size, bar
                )
                for wid in range(concurrency)
            ]
            for fut in as_completed(futs):
                all_lat.extend(fut.result())
        elapsed = time.perf_counter() - t0
        arr = np.array(all_lat)
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

    faiss.omp_set_num_threads(1)
    print("=" * 68)
    print("Batched Search Throughput")
    print("=" * 68)
    print(f"Baseline (single-vector c=1): {BASELINE_QPS:,.1f} QPS\n")

    queries = _sample(_load_vectors(VECTORS_PATH), 1_000)
    index = _load_hnsw(HNSW_PATH)

    results = []
    print("--- Batch size sweep (c=1) ---")
    for bs in BATCH_SIZES:
        results.append(_measure_batch(index, queries, 1, bs, f"batch={bs:<4} c=1"))

    best_bs = max(
        (r for r in results if r["concurrency"] == 1), key=lambda r: r["qps"]
    )["batch_size"]
    print(f"\n--- Best batch size ({best_bs}) at higher concurrency ---")
    for c in [4, 8]:
        results.append(
            _measure_batch(index, queries, c, best_bs, f"batch={best_bs:<4} c={c}")
        )

    print("\n\n" + "=" * 65)
    print(f"{'Label':<20} {'batch':>5} {'c':>2}  {'QPS':>8}  {'p50':>7}  {'x':>7}")
    print("-" * 65)
    best_speedup = max(r["speedup"] for r in results)
    for r in results:
        marker = " ←" if r["speedup"] == best_speedup else ""
        print(
            f"{r['label']:<20} {r['batch_size']:>5} {r['concurrency']:>2}"
            f"  {r['qps']:>8.1f}  {r['p50_ms']:>7.3f}"
            f"  {r['speedup']:>6.1f}×{marker}"
        )
    best = max(results, key=lambda r: r["qps"])
    print(
        f"\nBest: {best['label'].strip()}"
        f"  →  {best['qps']:,.0f} QPS  ({best['speedup']}× over baseline)"
    )


# ── throughput-ivfpq ──────────────────────────────────────────────────────────


def cmd_throughput_ivfpq() -> None:
    PHASE3_RECALL = {8: 0.6374, 32: 0.6631, 64: 0.6762, 128: 0.6878}
    HNSW_RECALL = 0.9857

    def _build_and_save_ivfpq(vecs: np.ndarray) -> faiss.IndexIVFPQ:
        print("Building IndexIVFPQ (nlist=4096, M=48, nbits=8)...")
        quantizer = faiss.IndexFlatL2(VECTOR_DIM)
        index = faiss.IndexIVFPQ(quantizer, VECTOR_DIM, 4096, 48, 8)
        train_idx = np.random.default_rng(SEED).choice(
            len(vecs), size=500_000, replace=False
        )
        index.train(vecs[train_idx])
        index.add(vecs)
        faiss.write_index(index, str(IVFPQ_PATH))
        size_gb = IVFPQ_PATH.stat().st_size / 1e9
        raw_gb = len(vecs) * VECTOR_DIM * 4 / 1e9
        print(
            f"  Saved: {size_gb:.2f} GB"
            f"  (vs {raw_gb:.1f} GB raw — {round(raw_gb / size_gb)}×)"
        )
        return index  # type: ignore[return-value]

    faiss.omp_set_num_threads(1)
    print("=" * 68)
    print("IVFPQ vs HNSW Throughput Comparison")
    print("=" * 68)

    vecs = _load_vectors(VECTORS_PATH)
    queries = _sample(vecs, 1_000)

    if IVFPQ_PATH.exists():
        print(f"Loading IVFPQ ({IVFPQ_PATH.stat().st_size / 1e9:.2f} GB)...")
        ivfpq = faiss.read_index(str(IVFPQ_PATH))
        print(f"  {ivfpq.ntotal:,} vectors")
    else:
        ivfpq = _build_and_save_ivfpq(vecs)

    hnsw = _load_hnsw(HNSW_PATH)
    del vecs

    results = []
    print("\n--- IndexIVFPQ ---")
    for nprobe in [8, 32, 64, 128]:
        ivfpq.nprobe = nprobe
        for c in [1, 8]:
            label = f"IVFPQ nprobe={nprobe:<3} c={c}"
            results.append(
                {**_measure_concurrency(ivfpq, queries, c, label), "nprobe": nprobe}
            )

    print("\n--- IndexHNSWFlat (ef=64) ---")
    for c in [1, 8]:
        label = f"HNSW ef=64          c={c}"
        results.append(
            {**_measure_concurrency(hnsw, queries, c, label), "nprobe": None}
        )

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

    hnsw_c8 = next(r for r in results if r["nprobe"] is None and r["concurrency"] == 8)
    best_ivfpq = max(
        (r for r in results if r["nprobe"] is not None), key=lambda r: r["qps"]
    )
    print(
        f"\nHNSW ef=64 c=8  → {hnsw_c8['qps']:,.0f} QPS"
        f"  recall={HNSW_RECALL:.4f}  {hnsw_gb:.1f} GB\n"
        f"IVFPQ nprobe={best_ivfpq['nprobe']} c={best_ivfpq['concurrency']}"
        f"  → {best_ivfpq['qps']:,.0f} QPS"
        f"  recall={PHASE3_RECALL.get(best_ivfpq['nprobe'], 0.0):.4f}"
        f"  {ivfpq_gb:.2f} GB"
    )


# ── profile ───────────────────────────────────────────────────────────────────


def cmd_profile() -> None:
    N_ITERATIONS = 5_000

    faiss.omp_set_num_threads(1)
    print("Loading vectors...")
    vecs = np.fromfile(VECTORS_PATH, dtype=np.float32).reshape(-1, VECTOR_DIM)
    queries = _sample(vecs, 500)
    del vecs

    print("Loading HNSW index...")
    index = faiss.read_index(str(HNSW_PATH))
    index.hnsw.efSearch = 64  # type: ignore[attr-defined]

    print(f"Profiling: {N_ITERATIONS:,} single-vector search() calls...")
    for i in range(N_ITERATIONS):
        q = queries[i % len(queries) : i % len(queries) + 1]
        index.search(q, K)
    print("Done.")


# ── entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified FAISS benchmark runner")
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("faiss-index", help="Compare FlatL2 / HNSW / IVFPQ index types")
    sub.add_parser(
        "throughput-concurrency", help="Concurrency sweep on HNSW (single-vector)"
    )
    sub.add_parser("throughput-batch", help="Batch size sweep on HNSW")
    sub.add_parser("throughput-ivfpq", help="IVFPQ vs HNSW throughput comparison")
    sub.add_parser("profile", help="5,000 single-vector searches for py-spy profiling")

    args = parser.parse_args()
    {
        "faiss-index": cmd_faiss_index,
        "throughput-concurrency": cmd_throughput_concurrency,
        "throughput-batch": cmd_throughput_batch,
        "throughput-ivfpq": cmd_throughput_ivfpq,
        "profile": cmd_profile,
    }[args.mode]()


if __name__ == "__main__":
    main()

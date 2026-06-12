"""
Phase 4 — Throughput regression gate.

Measures HNSW search QPS at the production serving config (c=8, ef=64)
and compares against a stored baseline.  Exits 1 if QPS drops more than
TOLERANCE below the baseline, or if p99 exceeds the P99_BUDGET.

Usage:
    # Record a new baseline (run after any intentional perf improvement):
    PYTHONPATH=src:. uv run python scripts/perf_check.py --record

    # Check current throughput against baseline (run before pushing):
    PYTHONPATH=src:. uv run python scripts/perf_check.py

Added to Makefile as 'make perf'.  NOT wired into 'make ci' — the HNSW
index (15.9 GB) is not available on CI runners and hardware varies too
much for a meaningful absolute threshold.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import faiss
import numpy as np

VECTORS_PATH = Path("data/vectors.bin")
HNSW_PATH = Path("data/hnsw.index")
BASELINE_PATH = Path("eval/results/perf_baseline.json")
VECTOR_DIM = 384
K = 10
SEED = 42
N_QUERY_POOL = 1_000
CONCURRENCY = 8
EF_SEARCH = 64
WARMUP_S = 2  # discarded — lets index warm up in page cache
MEASURE_S = 5  # measurement window
TOLERANCE = 0.10  # fail if QPS drops > 10% below baseline
P99_BUDGET = 5.0  # ms — absolute ceiling from Phase 4 spec


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def load_queries(vectors_path: Path, n: int, seed: int) -> np.ndarray:
    vecs = np.fromfile(vectors_path, dtype=np.float32).reshape(-1, VECTOR_DIM)
    rng = np.random.default_rng(seed)
    return vecs[rng.choice(len(vecs), size=n, replace=False)]


def load_index(path: Path) -> faiss.IndexHNSWFlat:
    index = faiss.read_index(str(path))
    index.hnsw.efSearch = EF_SEARCH  # type: ignore[attr-defined]
    return index  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def _worker(
    index: faiss.Index,
    queries: np.ndarray,
    duration: float,
    worker_id: int,
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
        i += 1
    return latencies


def measure(index: faiss.Index, queries: np.ndarray, duration: float) -> dict:
    all_latencies: list[float] = []
    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = [
            pool.submit(_worker, index, queries, duration, wid)
            for wid in range(CONCURRENCY)
        ]
        for fut in as_completed(futures):
            all_latencies.extend(fut.result())
    elapsed = time.perf_counter() - t_start
    arr = np.array(all_latencies)
    return {
        "qps": round(len(arr) / elapsed, 1),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "total_queries": len(arr),
    }


# ---------------------------------------------------------------------------
# Record / check
# ---------------------------------------------------------------------------


def record(index: faiss.Index, queries: np.ndarray) -> None:
    print(f"Warming up ({WARMUP_S}s)...")
    measure(index, queries, WARMUP_S)

    print(f"Measuring ({MEASURE_S}s)...")
    result = measure(index, queries, MEASURE_S)

    baseline = {
        "qps": result["qps"],
        "p99_ms": result["p99_ms"],
        "config": {
            "concurrency": CONCURRENCY,
            "ef_search": EF_SEARCH,
            "measure_s": MEASURE_S,
        },
    }
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_PATH.write_text(json.dumps(baseline, indent=2) + "\n")

    print(f"\nBaseline recorded → {BASELINE_PATH}")
    print(f"  QPS  : {result['qps']:,.1f}")
    print(f"  p50  : {result['p50_ms']:.3f} ms")
    print(f"  p99  : {result['p99_ms']:.3f} ms")


def check(index: faiss.Index, queries: np.ndarray) -> None:
    if not BASELINE_PATH.exists():
        print(f"No baseline found at {BASELINE_PATH}.")
        print("Run with --record first.")
        sys.exit(1)

    baseline = json.loads(BASELINE_PATH.read_text())

    print(f"Warming up ({WARMUP_S}s)...")
    measure(index, queries, WARMUP_S)

    print(f"Measuring ({MEASURE_S}s)...")
    result = measure(index, queries, MEASURE_S)

    baseline_qps = baseline["qps"]
    floor_qps = baseline_qps * (1 - TOLERANCE)
    qps_ok = result["qps"] >= floor_qps
    p99_ok = result["p99_ms"] <= P99_BUDGET

    delta_pct = (result["qps"] - baseline_qps) / baseline_qps * 100

    print("\nPerf check results:")
    print(
        f"  QPS   : {result['qps']:>8,.1f}  (baseline {baseline_qps:,.1f}"
        f", floor {floor_qps:,.1f})  {'✓' if qps_ok else '✗ FAIL'}"
    )
    print(
        f"  p99   : {result['p99_ms']:>8.3f} ms  (budget {P99_BUDGET:.1f} ms)"
        f"  {'✓' if p99_ok else '✗ FAIL'}"
    )
    print(f"  delta : {delta_pct:>+.1f}%")

    if qps_ok and p99_ok:
        print("\nperf gate passed")
    else:
        if not qps_ok:
            print(
                f"\n✗ QPS regression: {result['qps']:,.1f} < floor {floor_qps:,.1f}"
                f" ({TOLERANCE:.0%} below baseline {baseline_qps:,.1f})"
            )
        if not p99_ok:
            print(f"\n✗ p99 over budget: {result['p99_ms']:.3f} ms > {P99_BUDGET} ms")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    faiss.omp_set_num_threads(1)

    parser = argparse.ArgumentParser(description="HNSW throughput regression gate")
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record current throughput as the new baseline",
    )
    args = parser.parse_args()

    print(f"Loading queries from {VECTORS_PATH}...")
    queries = load_queries(VECTORS_PATH, N_QUERY_POOL, SEED)

    print(f"Loading HNSW index from {HNSW_PATH}...")
    index = load_index(HNSW_PATH)
    print(f"  {index.ntotal:,} vectors, efSearch={EF_SEARCH}, c={CONCURRENCY}\n")

    if args.record:
        record(index, queries)
    else:
        check(index, queries)


if __name__ == "__main__":
    main()

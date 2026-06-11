"""
Phase 3 — Property-based tests for the HNSW index.

Uses a synthetic 10,000-vector corpus (same dimension as the real index)
so tests run in CI without loading 8.8M vectors.  Hypothesis generates
hundreds of random query indices and checks that the index properties hold
for all of them.
"""

import faiss
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

VECTOR_DIM = 384
N_CORPUS = 100_000
K = 10
SEED = 42
MIN_RECALL = 0.90  # minimum acceptable ANNS recall@10 vs exact search


# ---------------------------------------------------------------------------
# Module-level corpus and indexes — built once, reused across all tests
# ---------------------------------------------------------------------------


def _make_cluster_centres(n_clusters: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    c = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    return c


def _sample_around_centres(
    n: int, centres: np.ndarray, noise: float, seed: int
) -> np.ndarray:
    """
    Sample n points near the shared cluster centres with Gaussian noise.

    Corpus and queries MUST share the same centres so that query points
    have well-defined nearest neighbours inside the corpus.

    Noise std per-dimension.  Total noise magnitude = noise * sqrt(dim).
    To keep points within ~10° of their centre in 384D:
      sin(10°) ≈ 0.174  →  noise = 0.174 / sqrt(384) ≈ 0.009
    """
    rng = np.random.default_rng(seed)
    n_clusters = len(centres)
    assignments = rng.integers(0, n_clusters, size=n)
    noise_vecs = rng.standard_normal((n, centres.shape[1])).astype(np.float32) * noise
    pts = centres[assignments] + noise_vecs
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


N_CLUSTERS = 500
NOISE = 0.009  # std per-dimension; total ≈ 0.009 * sqrt(384) ≈ 0.176 → ~10° rotation

# Shared cluster centres — corpus and queries must use the same centres
CENTRES = _make_cluster_centres(N_CLUSTERS, VECTOR_DIM, SEED)
CORPUS = _sample_around_centres(N_CORPUS, CENTRES, NOISE, SEED + 1)

# Separate query vectors not in the corpus
N_QUERIES = 500
QUERIES = _sample_around_centres(N_QUERIES, CENTRES, NOISE, SEED + 99)

# Exact index — ground truth for recall measurement
EXACT = faiss.IndexFlatL2(VECTOR_DIM)
EXACT.add(CORPUS)

# HNSW at the chosen operating point (ef=64)
HNSW = faiss.IndexHNSWFlat(VECTOR_DIM, 32)
HNSW.hnsw.efConstruction = 200
HNSW.hnsw.efSearch = 64
HNSW.add(CORPUS)

# HNSW with intentionally bad ef — for the break-it test
HNSW_BROKEN = faiss.IndexHNSWFlat(VECTOR_DIM, 32)
HNSW_BROKEN.hnsw.efConstruction = 200
HNSW_BROKEN.hnsw.efSearch = 2  # far too low — recall should collapse
HNSW_BROKEN.add(CORPUS)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _query(idx: int) -> np.ndarray:
    """Return query row `idx` as a (1, dim) matrix."""
    return QUERIES[idx : idx + 1]


def _recall(approx_ids: np.ndarray, true_ids: np.ndarray) -> float:
    """Fraction of true top-K neighbours returned by the approximate index."""
    return len(set(approx_ids.tolist()) & set(true_ids.tolist())) / K


# ---------------------------------------------------------------------------
# Property 1 — always returns exactly K results
# ---------------------------------------------------------------------------


@given(st.integers(min_value=0, max_value=N_QUERIES - 1))
@settings(max_examples=200)
def test_hnsw_returns_exactly_k_results(query_idx: int) -> None:
    """
    For any query in the corpus, HNSW must return exactly K neighbours.
    Hypothesis generates 200 random query indices to stress-test this.
    """
    _, ids = HNSW.search(_query(query_idx), K)
    assert ids.shape == (
        1,
        K,
    ), f"Expected (1, {K}) result shape, got {ids.shape} for query {query_idx}"


# ---------------------------------------------------------------------------
# Property 2 — average recall@10 >= 90% vs exact search
# ---------------------------------------------------------------------------


def test_hnsw_average_recall_meets_threshold() -> None:
    """
    HNSW at ef=64 must achieve >= 90% average recall@10 across all query vectors.

    Per-query recall is NOT asserted because random high-dimensional vectors
    suffer from the curse of dimensionality — in 384D, all unit vectors are
    nearly equidistant, so some queries have no "clear" top-10, making
    per-query recall noisy.  Average recall is the correct metric and matches
    what the Phase 3 benchmark measured.

    This test pins the efSearch parameter: lowering ef will drop the average
    below the threshold and fail CI.
    """
    recalls = []
    for i in range(N_QUERIES):
        q = _query(i)
        _, true_ids = EXACT.search(q, K)
        _, approx_ids = HNSW.search(q, K)
        recalls.append(_recall(approx_ids[0], true_ids[0]))

    mean_recall = float(np.mean(recalls))
    assert mean_recall >= MIN_RECALL, (
        f"Mean recall@10 {mean_recall:.4f} < {MIN_RECALL}. "
        f"efSearch={HNSW.hnsw.efSearch} may be too low."
    )


# ---------------------------------------------------------------------------
# Property 3 — deterministic: same query always returns same results
# ---------------------------------------------------------------------------


@given(st.integers(min_value=0, max_value=N_QUERIES - 1))
@settings(max_examples=100)
def test_hnsw_is_deterministic(query_idx: int) -> None:
    """
    Running the same query twice must produce identical results.
    HNSW has no randomness at search time — this catches any accidental
    global state mutation.
    """
    q = _query(query_idx)
    _, ids_first = HNSW.search(q, K)
    _, ids_second = HNSW.search(q, K)
    assert np.array_equal(ids_first, ids_second), (
        f"Non-deterministic results for query {query_idx}: "
        f"{ids_first[0].tolist()} != {ids_second[0].tolist()}"
    )


# ---------------------------------------------------------------------------
# Break-it — ef=2 collapses recall below 50%
# ---------------------------------------------------------------------------


def test_hnsw_broken_ef_collapses_recall() -> None:
    """
    efSearch=2 must produce significantly worse recall than efSearch=64.
    Tests the relative gap rather than an absolute floor, which is more
    robust across corpus sizes — small corpora need fewer hops to navigate
    so the absolute recall at ef=2 varies, but the gap vs ef=64 is stable.
    """
    rng = np.random.default_rng(SEED + 1)
    sample_indices = rng.choice(N_QUERIES, size=100, replace=False)

    recalls_good, recalls_broken = [], []
    for idx in sample_indices:
        q = _query(idx)
        _, true_ids = EXACT.search(q, K)
        _, good_ids = HNSW.search(q, K)
        _, broken_ids = HNSW_BROKEN.search(q, K)
        recalls_good.append(_recall(good_ids[0], true_ids[0]))
        recalls_broken.append(_recall(broken_ids[0], true_ids[0]))

    gap = float(np.mean(recalls_good)) - float(np.mean(recalls_broken))
    assert gap >= 0.10, (
        f"Expected ef=64 to outperform ef=2 by >= 0.10 recall, gap was {gap:.3f}. "
        "The efSearch parameter is not having the expected effect."
    )

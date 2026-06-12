import faiss
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

VECTOR_DIM = 384
N_CORPUS = 100_000
K = 10
SEED = 42
MIN_RECALL = 0.90  # minimum acceptable ANNS recall@10 vs exact search


def _make_cluster_centres(n_clusters: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    c = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    return c


def _sample_around_centres(
    n: int, centres: np.ndarray, noise: float, seed: int
) -> np.ndarray:
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
N_QUERIES = 500
QUERIES = _sample_around_centres(N_QUERIES, CENTRES, NOISE, SEED + 99)

EXACT = faiss.IndexFlatL2(VECTOR_DIM)
EXACT.add(CORPUS)

HNSW = faiss.IndexHNSWFlat(VECTOR_DIM, 32)
HNSW.hnsw.efConstruction = 200
HNSW.hnsw.efSearch = 64
HNSW.add(CORPUS)

HNSW_BROKEN = faiss.IndexHNSWFlat(VECTOR_DIM, 32)
HNSW_BROKEN.hnsw.efConstruction = 200
HNSW_BROKEN.hnsw.efSearch = 2  # far too low — recall should collapse
HNSW_BROKEN.add(CORPUS)


def _query(idx: int) -> np.ndarray:
    return QUERIES[idx : idx + 1]


def _recall(approx_ids: np.ndarray, true_ids: np.ndarray) -> float:
    return len(set(approx_ids.tolist()) & set(true_ids.tolist())) / K


@given(st.integers(min_value=0, max_value=N_QUERIES - 1))
@settings(max_examples=200)
def test_hnsw_returns_exactly_k_results(query_idx: int) -> None:
    _, ids = HNSW.search(_query(query_idx), K)
    assert ids.shape == (
        1,
        K,
    ), f"Expected (1, {K}) result shape, got {ids.shape} for query {query_idx}"


def test_hnsw_average_recall_meets_threshold() -> None:
    # Average recall, not per-query: curse of dimensionality makes per-query noisy.
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


@given(st.integers(min_value=0, max_value=N_QUERIES - 1))
@settings(max_examples=100)
def test_hnsw_is_deterministic(query_idx: int) -> None:
    # HNSW has no randomness at search time — catches accidental global state mutation
    q = _query(query_idx)
    _, ids_first = HNSW.search(q, K)
    _, ids_second = HNSW.search(q, K)
    assert np.array_equal(ids_first, ids_second), (
        f"Non-deterministic results for query {query_idx}: "
        f"{ids_first[0].tolist()} != {ids_second[0].tolist()}"
    )


def test_hnsw_broken_ef_collapses_recall() -> None:
    # Relative gap, not absolute floor: gap vs ef=64 is stable across corpus sizes.
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

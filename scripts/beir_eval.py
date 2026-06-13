"""
BEIR evaluation: SciFact, NFCorpus, ArguAna.

Pipeline: bge-small-en-v1.5 dense + BM25 → RRF → bge-reranker-base cross-encoder.
Metrics:  nDCG@10, Recall@10, MRR  (pytrec_eval — matches BEIR leaderboard).
Cache:    corpus embeddings saved to data/beir_embeddings/<dataset>/ on first run.
Output:   eval/results/beir_YYYY-MM-DD.json

Run:
    PYTHONPATH=src:. uv run python scripts/beir_eval.py
"""

import json
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pytrec_eval
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from rag_engine.retrieval import BM25Retriever, CrossEncoderReranker
from rag_engine.retrieval.hybrid import reciprocal_rank_fusion

# ── Config ────────────────────────────────────────────────────────────────────

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
RERANK_MODEL = "BAAI/bge-reranker-base"

CANDIDATE_POOL = 100  # dense + sparse candidates each
RERANK_POOL = 20  # RRF top-N fed to cross-encoder
TOP_K = 10  # nDCG@10 / Recall@10

EMBED_CACHE_ROOT = Path("data/beir_embeddings")
RESULTS_DIR = Path("eval/results")

# (short_name, hf_corpus_dataset, hf_qrels_dataset, qrels_split)
DATASETS: list[tuple[str, str, str, str]] = [
    ("scifact", "BeIR/scifact", "BeIR/scifact-qrels", "test"),
    ("nfcorpus", "BeIR/nfcorpus", "BeIR/nfcorpus-qrels", "test"),
    ("arguana", "BeIR/arguana", "BeIR/arguana-qrels", "test"),
]


# ── Embedding cache ───────────────────────────────────────────────────────────


def _load_or_build_dense(
    name: str,
    doc_ids: list[str],
    doc_texts: list[str],
    model: SentenceTransformer,
) -> faiss.IndexFlatIP:
    cache_dir = EMBED_CACHE_ROOT / name
    emb_path = cache_dir / "embeddings.npy"
    ids_path = cache_dir / "doc_ids.json"

    if emb_path.exists() and ids_path.exists():
        cached_ids: list[str] = json.loads(ids_path.read_text())
        if cached_ids == doc_ids:
            print(f"  Loading embeddings from cache ({emb_path})")
            vectors = np.load(str(emb_path))
        else:
            print("  Cache doc_ids mismatch — rebuilding")
            vectors = _embed(doc_texts, model, name)
            _write_cache(cache_dir, emb_path, ids_path, doc_ids, vectors)
    else:
        print(f"  Embedding {len(doc_ids):,} docs...")
        vectors = _embed(doc_texts, model, name)
        _write_cache(cache_dir, emb_path, ids_path, doc_ids, vectors)

    idx = faiss.IndexFlatIP(EMBED_DIM)
    idx.add(vectors)
    return idx


def _embed(texts: list[str], model: SentenceTransformer, desc: str) -> np.ndarray:
    vecs = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=256,
        show_progress_bar=True,
    )
    return np.asarray(vecs, dtype=np.float32)


def _write_cache(
    cache_dir: Path,
    emb_path: Path,
    ids_path: Path,
    doc_ids: list[str],
    vectors: np.ndarray,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(emb_path), vectors)
    ids_path.write_text(json.dumps(doc_ids))
    print(f"  Saved → {emb_path}")


# ── Dataset loading ───────────────────────────────────────────────────────────


def _load_corpus(hf_name: str) -> tuple[list[str], list[str], dict[str, str]]:
    ds = load_dataset(hf_name, "corpus", split="corpus", trust_remote_code=False)
    doc_ids, doc_texts, corpus_map = [], [], {}
    for row in ds:  # type: ignore[union-attr]
        did = str(row["_id"])
        text = (str(row.get("title") or "") + " " + str(row.get("text") or "")).strip()
        doc_ids.append(did)
        doc_texts.append(text)
        corpus_map[did] = text
    return doc_ids, doc_texts, corpus_map


def _load_queries(hf_name: str) -> dict[str, str]:
    ds = load_dataset(hf_name, "queries", split="queries", trust_remote_code=False)
    return {str(row["_id"]): str(row["text"]) for row in ds}  # type: ignore[union-attr]


def _load_qrels(hf_name: str, split: str) -> dict[str, dict[str, int]]:
    ds = load_dataset(hf_name, split=split, trust_remote_code=False)
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for row in ds:  # type: ignore[union-attr]
        score = int(row["score"])  # type: ignore[index]
        if score > 0:
            qrels[str(row["query-id"])][str(row["corpus-id"])] = score  # type: ignore[index]
    return dict(qrels)


# ── Per-dataset run ───────────────────────────────────────────────────────────


def _run_dataset(
    name: str,
    hf_name: str,
    hf_qrels: str,
    qrels_split: str,
    embedder: SentenceTransformer,
    reranker: CrossEncoderReranker,
) -> dict[str, Any]:
    print(f"\n{'═' * 60}")
    print(f"Dataset: {name.upper()}")
    print(f"{'═' * 60}")

    doc_ids, doc_texts, corpus_map = _load_corpus(hf_name)
    print(f"Corpus: {len(doc_ids):,} docs")

    queries = _load_queries(hf_name)
    qrels = _load_qrels(hf_qrels, qrels_split)
    eval_qids = [qid for qid in qrels if qid in queries]
    print(f"Queries: {len(queries):,} total | {len(eval_qids):,} with qrels")

    print("Building BM25 index...")
    bm25 = BM25Retriever(doc_ids, doc_texts)

    print("Building dense index...")
    faiss_idx = _load_or_build_dense(name, doc_ids, doc_texts, embedder)

    def dense_search(query: str, k: int) -> list[str]:
        qvec = embedder.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)
        _, idxs = faiss_idx.search(qvec, k)
        return [doc_ids[int(i)] for i in idxs[0] if 0 <= int(i) < len(doc_ids)]

    print(f"Retrieving + reranking {len(eval_qids):,} queries...")
    run: dict[str, dict[str, float]] = {}

    for qid in tqdm(eval_qids, desc=name):
        q = queries[qid]
        dense = dense_search(q, CANDIDATE_POOL)
        sparse = bm25.retrieve(q, CANDIDATE_POOL)
        fused = reciprocal_rank_fusion([dense, sparse], k=RERANK_POOL)
        ranked = reranker.rerank(q, fused, corpus_map, TOP_K)
        run[qid] = {did: 1.0 / (rank + 1) for rank, did in enumerate(ranked)}

    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {"ndcg_cut_10", "recall_10", "recip_rank"}
    )
    per_query = evaluator.evaluate(run)

    ndcg_vals = [v["ndcg_cut_10"] for v in per_query.values()]
    recall_vals = [v["recall_10"] for v in per_query.values()]
    mrr_vals = [v["recip_rank"] for v in per_query.values()]

    metrics: dict[str, Any] = {
        "ndcg_at_10": round(float(np.mean(ndcg_vals)), 4),
        "recall_at_10": round(float(np.mean(recall_vals)), 4),
        "mrr": round(float(np.mean(mrr_vals)), 4),
        "n_queries": len(eval_qids),
        "n_docs": len(doc_ids),
    }
    print(
        f"  nDCG@10={metrics['ndcg_at_10']:.4f}  "
        f"Recall@10={metrics['recall_at_10']:.4f}  "
        f"MRR={metrics['mrr']:.4f}"
    )
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    print("Loading embedding model...")
    try:
        embedder = SentenceTransformer(EMBED_MODEL, device="mps")
    except Exception:
        embedder = SentenceTransformer(EMBED_MODEL, device="cpu")

    print(f"Loading reranker: {RERANK_MODEL}")
    reranker = CrossEncoderReranker(RERANK_MODEL)

    output: dict[str, Any] = {
        "date": str(date.today()),
        "embed_model": EMBED_MODEL,
        "rerank_model": RERANK_MODEL,
        "pipeline": "bge-small-en-v1.5 dense + bm25s → rrf → bge-reranker-base",
        "datasets": {},
    }

    for name, hf_name, hf_qrels, qrels_split in DATASETS:
        output["datasets"][name] = _run_dataset(
            name, hf_name, hf_qrels, qrels_split, embedder, reranker
        )

    # ── Summary table ─────────────────────────────────────────────────────────
    w = 52
    print(f"\n{'─' * w}")
    print(f"{'Dataset':<12} {'nDCG@10':>9}  {'Recall@10':>10}  {'MRR':>7}")
    print(f"{'─' * w}")
    for name, m in output["datasets"].items():
        print(
            f"{name:<12} {m['ndcg_at_10']:>9.4f}  "
            f"{m['recall_at_10']:>10.4f}  {m['mrr']:>7.4f}"
        )
    print(f"{'─' * w}")

    # ── Save ──────────────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"beir_{date.today()}.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()

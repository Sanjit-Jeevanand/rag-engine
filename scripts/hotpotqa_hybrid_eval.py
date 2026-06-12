import json
import sqlite3
from pathlib import Path

import numpy as np
from tqdm import tqdm

from eval.index import VectorIndex
from eval.metrics import mrr, ndcg_at_k, recall_at_k
from rag_engine.retrieval import (
    BM25Retriever,
    CrossEncoderReranker,
    reciprocal_rank_fusion,
)

DB_PATH = Path("data/docs.db")
VECTORS_PATH = Path("data/vectors.bin")
GOLD_PATH = Path("eval/hotpotqa_gold.json")
OUT = Path("eval/results/hotpotqa_staircase.json")

RERANK_MODEL = "BAAI/bge-reranker-base"
CANDIDATE_POOL = 100
RERANK_POOL = 20
K = 10


def _load_article_texts(db_path: Path) -> tuple[list[str], dict[str, str]]:
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT title, chunk_text FROM documents"
        " WHERE status='embedded' AND chunk_index=0"
    ).fetchall()
    conn.close()
    titles = [r[0] for r in rows]
    title_to_text = {r[0]: r[1] for r in rows}
    return titles, title_to_text


def main() -> None:
    print("Loading FAISS index (this reads ~13.5 GB)...")
    index = VectorIndex(DB_PATH, VECTORS_PATH)

    print("\nLoading first chunk per article for BM25...")
    titles, title_to_text = _load_article_texts(DB_PATH)
    print(f"  {len(titles):,} articles")

    print("Building BM25 index...")
    bm25 = BM25Retriever(titles, [title_to_text[t] for t in titles])

    print(f"Loading reranker: {RERANK_MODEL}")
    reranker = CrossEncoderReranker(RERANK_MODEL)

    gold = json.loads(GOLD_PATH.read_text())
    print(f"\nEvaluating {len(gold)} questions at k={K}...\n")

    ndcg_d, ndcg_h, ndcg_r = [], [], []
    rec_d, rec_h, rec_r = [], [], []
    mrr_d, mrr_h, mrr_r = [], [], []

    for item in tqdm(gold):
        q = item["question"]
        relevant = set(item["supporting_titles"])

        dense_100 = index.search(q, k=CANDIDATE_POOL)
        bm25_100 = bm25.retrieve(q, CANDIDATE_POOL)
        hybrid_10 = reciprocal_rank_fusion([dense_100, bm25_100], k=K)
        rrf_20 = reciprocal_rank_fusion([dense_100, bm25_100], k=RERANK_POOL)
        reranked_10 = reranker.rerank(q, rrf_20, title_to_text, K)

        ndcg_d.append(ndcg_at_k(dense_100[:K], relevant))
        ndcg_h.append(ndcg_at_k(hybrid_10, relevant))
        ndcg_r.append(ndcg_at_k(reranked_10, relevant))

        rec_d.append(recall_at_k(dense_100[:K], relevant))
        rec_h.append(recall_at_k(hybrid_10, relevant))
        rec_r.append(recall_at_k(reranked_10, relevant))

        mrr_d.append(mrr(dense_100[:K], relevant))
        mrr_h.append(mrr(hybrid_10, relevant))
        mrr_r.append(mrr(reranked_10, relevant))

    def avg(xs: list[float]) -> float:
        return round(float(np.mean(xs)), 4)

    results = {
        "dense": {
            "ndcg_at_10": avg(ndcg_d),
            "recall_at_10": avg(rec_d),
            "mrr": avg(mrr_d),
        },
        "hybrid": {
            "ndcg_at_10": avg(ndcg_h),
            "recall_at_10": avg(rec_h),
            "mrr": avg(mrr_h),
        },
        "hybrid_rerank": {
            "ndcg_at_10": avg(ndcg_r),
            "recall_at_10": avg(rec_r),
            "mrr": avg(mrr_r),
        },
        "n_questions": len(gold),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {OUT}")

    base_ndcg = results["dense"]["ndcg_at_10"]
    print(f"\n{'Stage':<20} {'nDCG@10':>9} {'Recall@10':>10} {'MRR':>8}  {'Δ nDCG':>8}")
    print("-" * 62)
    for stage, r in results.items():
        if stage == "n_questions":
            continue
        delta = (
            f"{r['ndcg_at_10'] - base_ndcg:+.4f}" if stage != "dense" else "baseline"
        )
        print(
            f"{stage:<20} {r['ndcg_at_10']:>9.4f}"
            f" {r['recall_at_10']:>10.4f} {r['mrr']:>8.4f}  {delta:>8}"
        )


if __name__ == "__main__":
    main()

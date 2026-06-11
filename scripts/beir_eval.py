"""
Lightweight BEIR baseline: SciFact + NFCorpus.

Embeds each corpus with bge-small-en-v1.5 (same model as Wikipedia index),
builds a per-dataset FAISS IndexFlatIP, and scores nDCG@10 / Recall@10 / MRR
on the test-split queries.

Run with:
    PYTHONPATH=src:. uv run python scripts/beir_eval.py
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 512
K = 10
OUT = Path("eval/results/beir_baseline.json")

DATASETS = [
    {
        "name": "scifact",
        "corpus_hf": ("BeIR/scifact", "corpus", "corpus"),
        "queries_hf": ("BeIR/scifact", "queries", "queries"),
        "qrels_hf": ("BeIR/scifact-qrels", None, "test"),
    },
    {
        "name": "nfcorpus",
        "corpus_hf": ("BeIR/nfcorpus", "corpus", "corpus"),
        "queries_hf": ("BeIR/nfcorpus", "queries", "queries"),
        "qrels_hf": ("BeIR/nfcorpus-qrels", None, "test"),
    },
]


# ---------------------------------------------------------------------------
# Graded nDCG — handles relevance scores > 1 (NFCorpus uses 0/1/2)
# ---------------------------------------------------------------------------


def _ndcg_at_k_graded(
    retrieved_ids: list[str],
    qrels: dict[str, int],
    k: int,
) -> float:
    dcg = sum(
        qrels.get(doc_id, 0) / math.log2(rank + 2)
        for rank, doc_id in enumerate(retrieved_ids[:k])
    )
    ideal_gains = sorted(qrels.values(), reverse=True)
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal_gains[:k]))
    return dcg / idcg if idcg > 0 else 0.0


def _recall_at_k(retrieved_ids: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for d in retrieved_ids[:k] if d in relevant) / len(relevant)


def _mrr(retrieved_ids: list[str], relevant: set[str]) -> float:
    for i, d in enumerate(retrieved_ids):
        if d in relevant:
            return 1.0 / (i + 1)
    return 0.0


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def embed_texts(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    """Encode texts in batches; returns L2-normalised float32 matrix."""
    vecs = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vecs.astype(np.float32)


# ---------------------------------------------------------------------------
# Per-dataset evaluation
# ---------------------------------------------------------------------------


def evaluate_dataset(cfg: dict, model: SentenceTransformer) -> dict:
    name = cfg["name"]
    print(f"\n{'=' * 60}")
    print(f"Dataset: {name.upper()}")
    print(f"{'=' * 60}")

    # Load corpus
    hf_id, hf_cfg, hf_split = cfg["corpus_hf"]
    corpus_ds = load_dataset(hf_id, hf_cfg, split=hf_split)
    print(f"Corpus: {len(corpus_ds):,} docs")

    # Build doc_id → text mapping (title + text, BEIR standard)
    doc_ids: list[str] = []
    doc_texts: list[str] = []
    for row in corpus_ds:
        doc_ids.append(str(row["_id"]))
        title = row.get("title", "") or ""
        text = row.get("text", "") or ""
        doc_texts.append(f"{title}. {text}".strip(". "))

    # Load qrels (test split)
    qr_id, qr_cfg, qr_split = cfg["qrels_hf"]
    qrels_ds = (
        load_dataset(qr_id, qr_cfg, split=qr_split)
        if qr_cfg
        else load_dataset(qr_id, split=qr_split)
    )
    # qrels: {query_id_str: {corpus_id_str: score}}
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for row in qrels_ds:
        qid = str(row["query-id"])
        cid = str(row["corpus-id"])
        score = int(row["score"])
        if score > 0:
            qrels[qid][cid] = score
    test_qids = set(qrels.keys())
    print(f"Test queries with qrels: {len(test_qids):,}")

    # Load queries — keep only those present in qrels
    hf_id, hf_cfg, hf_split = cfg["queries_hf"]
    queries_ds = load_dataset(hf_id, hf_cfg, split=hf_split)
    query_id_to_text: dict[str, str] = {}
    for row in queries_ds:
        qid = str(row["_id"])
        if qid in test_qids:
            query_id_to_text[qid] = row["text"]
    print(f"Matching queries loaded: {len(query_id_to_text):,}")

    # Embed corpus
    print(f"\nEmbedding {name} corpus...")
    corpus_vecs = embed_texts(doc_texts, model)

    # Build FAISS index
    dim = corpus_vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(corpus_vecs)
    print(f"Index built: {index.ntotal:,} vectors")

    # Embed queries and search
    query_ids_ordered = list(query_id_to_text.keys())
    query_texts = [query_id_to_text[qid] for qid in query_ids_ordered]

    print(f"\nEmbedding {len(query_texts)} queries...")
    query_vecs = embed_texts(query_texts, model)

    print("Searching...")
    _, indices = index.search(query_vecs, K)

    # Score
    ndcg_scores, recall_scores, mrr_scores = [], [], []
    for qidx, qid in enumerate(query_ids_ordered):
        doc_qrels = qrels[qid]
        relevant_ids = set(doc_qrels.keys())
        retrieved_ids = [doc_ids[i] for i in indices[qidx] if i >= 0]

        ndcg_scores.append(_ndcg_at_k_graded(retrieved_ids, doc_qrels, K))
        recall_scores.append(_recall_at_k(retrieved_ids, relevant_ids, K))
        mrr_scores.append(_mrr(retrieved_ids, relevant_ids))

    result = {
        "ndcg_at_10": round(np.mean(ndcg_scores), 4),
        "recall_at_10": round(np.mean(recall_scores), 4),
        "mrr": round(np.mean(mrr_scores), 4),
        "n_queries": len(ndcg_scores),
    }

    print(f"\nResults — {name}")
    print(f"  nDCG@10  : {result['ndcg_at_10']:.4f}")
    print(f"  Recall@10: {result['recall_at_10']:.4f}")
    print(f"  MRR      : {result['mrr']:.4f}")
    print(f"  Queries  : {result['n_queries']}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print(f"Loading model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    results: dict[str, dict] = {}
    for cfg in DATASETS:
        results[cfg["name"]] = evaluate_dataset(cfg, model)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {OUT}")

    # Summary table
    print(f"\n{'Dataset':<12} {'nDCG@10':>9} {'Recall@10':>10} {'MRR':>8}")
    print("-" * 42)
    for name, r in results.items():
        ndcg = r["ndcg_at_10"]
        rec = r["recall_at_10"]
        mrr_val = r["mrr"]
        print(f"{name:<12} {ndcg:>9.4f} {rec:>10.4f} {mrr_val:>8.4f}")


if __name__ == "__main__":
    main()

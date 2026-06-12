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


def _ndcg_at_k_graded(retrieved_ids: list[str], qrels: dict[str, int], k: int) -> float:
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


def embed_texts(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    vecs = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vecs.astype(np.float32)


def evaluate_dataset(cfg: dict, model: SentenceTransformer) -> dict:
    name = cfg["name"]
    print(f"\n{'=' * 60}\nDataset: {name.upper()}\n{'=' * 60}")

    hf_id, hf_cfg, hf_split = cfg["corpus_hf"]
    corpus_ds = load_dataset(hf_id, hf_cfg, split=hf_split)
    print(f"Corpus: {len(corpus_ds):,} docs")

    doc_ids: list[str] = []
    doc_texts: list[str] = []
    for row in corpus_ds:
        doc_ids.append(str(row["_id"]))
        title = row.get("title", "") or ""
        text = row.get("text", "") or ""
        doc_texts.append(f"{title}. {text}".strip(". "))

    qr_id, qr_cfg, qr_split = cfg["qrels_hf"]
    qrels_ds = (
        load_dataset(qr_id, qr_cfg, split=qr_split)
        if qr_cfg
        else load_dataset(qr_id, split=qr_split)
    )
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for row in qrels_ds:
        qid, cid, score = str(row["query-id"]), str(row["corpus-id"]), int(row["score"])
        if score > 0:
            qrels[qid][cid] = score
    test_qids = set(qrels.keys())
    print(f"Test queries with qrels: {len(test_qids):,}")

    hf_id, hf_cfg, hf_split = cfg["queries_hf"]
    queries_ds = load_dataset(hf_id, hf_cfg, split=hf_split)
    query_id_to_text: dict[str, str] = {
        str(row["_id"]): row["text"]
        for row in queries_ds
        if str(row["_id"]) in test_qids
    }
    print(f"Matching queries loaded: {len(query_id_to_text):,}")

    print(f"\nEmbedding {name} corpus...")
    corpus_vecs = embed_texts(doc_texts, model)

    index = faiss.IndexFlatIP(corpus_vecs.shape[1])
    index.add(corpus_vecs)
    print(f"Index built: {index.ntotal:,} vectors")

    query_ids_ordered = list(query_id_to_text.keys())
    query_texts = [query_id_to_text[qid] for qid in query_ids_ordered]

    print(f"\nEmbedding {len(query_texts)} queries...")
    query_vecs = embed_texts(query_texts, model)

    print("Searching...")
    _, indices = index.search(query_vecs, K)

    ndcg_scores, recall_scores, mrr_scores = [], [], []
    for qidx, qid in enumerate(query_ids_ordered):
        doc_qrels = qrels[qid]
        relevant_ids = set(doc_qrels.keys())
        retrieved_ids = [doc_ids[i] for i in indices[qidx] if i >= 0]

        ndcg_scores.append(_ndcg_at_k_graded(retrieved_ids, doc_qrels, K))
        recall_scores.append(_recall_at_k(retrieved_ids, relevant_ids, K))
        mrr_scores.append(_mrr(retrieved_ids, relevant_ids))

    result = {
        "ndcg_at_10": round(float(np.mean(ndcg_scores)), 4),
        "recall_at_10": round(float(np.mean(recall_scores)), 4),
        "mrr": round(float(np.mean(mrr_scores)), 4),
        "n_queries": len(ndcg_scores),
    }

    print(f"\nResults — {name}")
    print(f"  nDCG@10  : {result['ndcg_at_10']:.4f}")
    print(f"  Recall@10: {result['recall_at_10']:.4f}")
    print(f"  MRR      : {result['mrr']:.4f}")
    print(f"  Queries  : {result['n_queries']}")

    return result


def main() -> None:
    print(f"Loading model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    results: dict[str, dict] = {}
    for cfg in DATASETS:
        results[cfg["name"]] = evaluate_dataset(cfg, model)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {OUT}")

    print(f"\n{'Dataset':<12} {'nDCG@10':>9} {'Recall@10':>10} {'MRR':>8}")
    print("-" * 42)
    for name, r in results.items():
        print(
            f"{name:<12} {r['ndcg_at_10']:>9.4f}"
            f" {r['recall_at_10']:>10.4f} {r['mrr']:>8.4f}"
        )


if __name__ == "__main__":
    main()

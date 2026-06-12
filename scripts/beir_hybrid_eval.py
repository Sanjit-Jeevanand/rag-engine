import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from rag_engine.retrieval import (
    BM25Retriever,
    CrossEncoderReranker,
    DenseRetriever,
    reciprocal_rank_fusion,
)

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-base"
EMBED_BATCH = 512
K = 10
CANDIDATE_POOL = 100  # top-K from each retriever fed into RRF
RERANK_POOL = 20  # top-K from RRF fed into the cross-encoder

BASELINE_PATH = Path("eval/results/beir_baseline.json")
OUT = Path("eval/results/beir_staircase.json")

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


def _ndcg_at_k(retrieved: list[str], qrels: dict[str, int], k: int) -> float:
    # Graded: NFCorpus uses relevance scores 0/1/2, not binary
    dcg = sum(qrels.get(d, 0) / math.log2(r + 2) for r, d in enumerate(retrieved[:k]))
    ideal = sorted(qrels.values(), reverse=True)
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal[:k]))
    return dcg / idcg if idcg > 0 else 0.0


def load_beir_dataset(
    cfg: dict,
) -> tuple[list[str], list[str], dict[str, str], dict[str, dict[str, int]]]:
    hf_id, hf_cfg, hf_split = cfg["corpus_hf"]
    corpus_ds = load_dataset(hf_id, hf_cfg, split=hf_split)
    doc_ids: list[str] = []
    doc_texts: list[str] = []
    for row in corpus_ds:
        doc_ids.append(str(row["_id"]))
        title = row.get("title", "") or ""
        text = row.get("text", "") or ""
        doc_texts.append(f"{title}. {text}".strip(". "))

    # Qrels (ground truth relevance labels)
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

    # Queries — keep only those with qrels
    hf_id, hf_cfg, hf_split = cfg["queries_hf"]
    queries_ds = load_dataset(hf_id, hf_cfg, split=hf_split)
    query_id_to_text: dict[str, str] = {}
    for row in queries_ds:
        qid = str(row["_id"])
        if qid in test_qids:
            query_id_to_text[qid] = row["text"]

    return doc_ids, doc_texts, query_id_to_text, dict(qrels)


def embed(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    vecs = model.encode(
        texts,
        batch_size=EMBED_BATCH,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(vecs, dtype=np.float32)


def evaluate_dataset(
    cfg: dict,
    embed_model: SentenceTransformer,
    reranker: CrossEncoderReranker,
) -> dict:
    name = cfg["name"]
    print(f"\n{'=' * 60}\nDataset: {name.upper()}\n{'=' * 60}")

    doc_ids, doc_texts, query_id_to_text, qrels = load_beir_dataset(cfg)
    doc_text_map = dict(zip(doc_ids, doc_texts, strict=False))
    print(f"Corpus: {len(doc_ids):,} docs   Queries: {len(query_id_to_text):,}")

    print(f"\nEmbedding {name} corpus...")
    corpus_vecs = embed(doc_texts, embed_model)

    print(f"\nEmbedding {len(query_id_to_text)} queries...")
    query_ids = list(query_id_to_text.keys())
    query_vecs = embed([query_id_to_text[qid] for qid in query_ids], embed_model)

    print("\nBuilding BM25 index...")
    bm25 = BM25Retriever(doc_ids, doc_texts)
    dense = DenseRetriever(doc_ids, corpus_vecs)

    ndcg_dense, ndcg_hybrid, ndcg_rerank = [], [], []

    print("Running retrieval pipelines...")
    for qidx, qid in enumerate(tqdm(query_ids)):
        qtext = query_id_to_text[qid]
        qvec = query_vecs[qidx]
        doc_qrels = qrels[qid]

        dense_100 = dense.retrieve(qvec, CANDIDATE_POOL)
        ndcg_dense.append(_ndcg_at_k(dense_100[:K], doc_qrels, K))

        bm25_100 = bm25.retrieve(qtext, CANDIDATE_POOL)
        hybrid_10 = reciprocal_rank_fusion([dense_100, bm25_100], k=K)
        ndcg_hybrid.append(_ndcg_at_k(hybrid_10, doc_qrels, K))

        rrf_20 = reciprocal_rank_fusion([dense_100, bm25_100], k=RERANK_POOL)
        reranked_10 = reranker.rerank(qtext, rrf_20, doc_text_map, K)
        ndcg_rerank.append(_ndcg_at_k(reranked_10, doc_qrels, K))

    result = {
        "dense": round(float(np.mean(ndcg_dense)), 4),
        "hybrid": round(float(np.mean(ndcg_hybrid)), 4),
        "hybrid_rerank": round(float(np.mean(ndcg_rerank)), 4),
        "n_queries": len(ndcg_dense),
    }
    print(
        f"\n{name}  dense={result['dense']:.4f}  "
        f"hybrid={result['hybrid']:.4f}  "
        f"hybrid+rerank={result['hybrid_rerank']:.4f}"
    )
    return result


def main() -> None:
    print(f"Loading embed model  : {EMBED_MODEL}")
    embed_model = SentenceTransformer(EMBED_MODEL)

    print(f"Loading reranker     : {RERANK_MODEL}")
    reranker = CrossEncoderReranker(RERANK_MODEL)

    results: dict[str, dict] = {}
    for cfg in DATASETS:
        results[cfg["name"]] = evaluate_dataset(cfg, embed_model, reranker)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {OUT}")

    baseline = json.loads(BASELINE_PATH.read_text())
    print(
        f"\n{'Dataset':<12} {'Dense':>8} {'Hybrid':>8} {'+Rerank':>8}  {'Δ final':>8}"
    )
    print("-" * 52)
    for name, r in results.items():
        base = baseline.get(name, {}).get("ndcg_at_10", 0.0)
        delta = r["hybrid_rerank"] - base
        print(
            f"{name:<12} {r['dense']:>8.4f} {r['hybrid']:>8.4f}"
            f" {r['hybrid_rerank']:>8.4f}  {delta:>+8.4f}"
        )


if __name__ == "__main__":
    main()

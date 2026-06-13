"""
Unified diagnostics runner.

Usage:
  python scripts/diagnostics.py failures        # single-shot RAG on 100 questions
  python scripts/diagnostics.py links           # incoming_links distribution (50K)
  python scripts/diagnostics.py retrieval       # qualitative check on 20 questions
  python scripts/diagnostics.py embed-watch     # live embedding progress monitor
  python scripts/diagnostics.py beir-hybrid     # BEIR staircase (dense/hybrid/rerank)
  python scripts/diagnostics.py hotpotqa-hybrid # HotpotQA staircase eval
  python scripts/diagnostics.py ablation        # chunk size ablation (1500/500/300)
"""

import argparse
import json
import math
import random
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

# ── failures ──────────────────────────────────────────────────────────────────


def cmd_failures() -> None:
    from eval.index import VectorIndex
    from eval.metrics import exact_match, f1
    from rag_engine.agent.llm import complete
    from rag_engine.retrieval import (
        BM25Retriever,
        CrossEncoderReranker,
        reciprocal_rank_fusion,
    )

    DB_PATH = Path("data/docs.db")
    VECTORS_PATH = Path("data/vectors.bin")
    BM25_INDEX_DIR = Path("data/bm25_index")
    GOLD_PATH = Path("eval/hotpotqa_gold.json")
    OUT = Path("eval/results/single_shot_baseline.json")

    RERANK_MODEL = "BAAI/bge-reranker-base"
    CANDIDATE_POOL = 100
    RERANK_POOL = 20
    TOP_K = 5
    N_QUESTIONS = 100
    N_FAILURES = 10

    _SYSTEM = (
        "Answer the question using ONLY the provided passages. "
        "Give a concise factual answer — a name, date, place, or yes/no. "
        "If the answer is not in the passages, say 'I don't know.'"
    )

    def _load_article_texts(
        db_path: Path,
    ) -> tuple[list[str], list[str], dict[str, str]]:
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT title, chunk_text FROM documents"
            " WHERE status='embedded' AND chunk_index=0"
        ).fetchall()
        conn.close()
        titles = [r[0] for r in rows]
        texts = [r[1] for r in rows]
        return titles, texts, {r[0]: r[1] for r in rows}

    def _format_passages(doc_ids: list[str], doc_texts: dict[str, str]) -> str:
        parts = []
        for doc_id in doc_ids:
            text = doc_texts.get(doc_id, "")
            if text:
                parts.append(f"[{doc_id}]\n{text[:800]}")
        return "\n\n".join(parts)

    print("Loading FAISS index...")
    index = VectorIndex(DB_PATH, VECTORS_PATH)

    print("Loading article texts for BM25...")
    titles, texts, title_to_text = _load_article_texts(DB_PATH)
    print(f"  {len(titles):,} articles")

    if BM25_INDEX_DIR.exists():
        print("Loading BM25 index from cache...")
        bm25 = BM25Retriever.load(BM25_INDEX_DIR)
    else:
        print("Building BM25 index...")
        bm25 = BM25Retriever(titles, texts)
        bm25.save(BM25_INDEX_DIR)
        print(f"  Saved → {BM25_INDEX_DIR}")

    print(f"Loading reranker: {RERANK_MODEL}")
    reranker = CrossEncoderReranker(RERANK_MODEL)

    def retrieve(query: str, k: int) -> list[str]:
        dense = index.search(query, k=CANDIDATE_POOL)
        sparse = bm25.retrieve(query, CANDIDATE_POOL)
        rrf = reciprocal_rank_fusion([dense, sparse], k=RERANK_POOL)
        return reranker.rerank(query, rrf, title_to_text, k)

    gold = json.loads(GOLD_PATH.read_text())[:N_QUESTIONS]
    print(f"\nRunning single-shot RAG on {N_QUESTIONS} questions...\n")

    records = []
    for item in tqdm(gold):
        question = item["question"]
        gold_answer = item["answer"]
        supporting = item["supporting_titles"]

        top5 = retrieve(question, TOP_K)
        passages = _format_passages(top5, title_to_text)
        msg = f"Question: {question}\n\nPassages:\n{passages}"
        predicted = complete(
            [{"role": "user", "content": msg}],
            max_tokens=64,
            system=_SYSTEM,
        ).strip()

        em = exact_match(predicted, gold_answer)
        f1_score = f1(predicted, gold_answer)
        retrieved_supporting = [t for t in supporting if t in top5]

        records.append(
            {
                "question": question,
                "gold": gold_answer,
                "predicted": predicted,
                "em": em,
                "f1": f1_score,
                "retrieved": top5,
                "supporting_titles": supporting,
                "retrieved_supporting": retrieved_supporting,
                "bridge_gap": len(retrieved_supporting) < len(supporting),
            }
        )

    em_avg = float(np.mean([r["em"] for r in records]))
    f1_avg = float(np.mean([r["f1"] for r in records]))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    summary = {"em": round(em_avg, 4), "f1": round(f1_avg, 4), "n": N_QUESTIONS}
    OUT.write_text(json.dumps({"summary": summary, "records": records}, indent=2))

    print(f"\nEM: {em_avg:.4f}   F1: {f1_avg:.4f}   (n={N_QUESTIONS})")
    print(f"Saved → {OUT}\n")

    failures = sorted([r for r in records if r["em"] == 0.0], key=lambda r: r["f1"])
    bridge_gaps = sum(1 for r in failures if r["bridge_gap"])
    print(f"{'─' * 68}")
    print(
        f"10 WORST FAILURES  "
        f"({len(failures)}/{N_QUESTIONS} failed,  {bridge_gaps} bridge gaps)"
    )
    print(f"{'─' * 68}\n")

    for i, r in enumerate(failures[:N_FAILURES], 1):
        missing = [t for t in r["supporting_titles"] if t not in r["retrieved"]]
        gap_label = (
            f"✗ missing {missing}" if missing else "✓ both retrieved — LLM failure"
        )
        print(f"[{i}] F1={r['f1']:.2f}  {gap_label}")
        print(f"  Q   : {r['question']}")
        print(f"  Gold: {r['gold']}")
        print(f"  Pred: {r['predicted']}")
        print(f"  Top5: {r['retrieved']}")
        print()


# ── links ─────────────────────────────────────────────────────────────────────


def cmd_links() -> None:
    from rag_engine.ingest.parser import parse_snapshot

    SNAPSHOT = Path("data/wiki-dump.json.gz")
    SAMPLE = 50_000

    counts = []
    with tqdm(total=SAMPLE, desc="Sampling") as bar:
        for article in parse_snapshot(SNAPSHOT):
            counts.append(article.incoming_links)
            bar.update(1)
            if len(counts) >= SAMPLE:
                break

    counts.sort(reverse=True)
    n = len(counts)
    print(f"Sampled {n:,} articles")
    print(f"Max:    {counts[0]:,}")
    print(f"p99:    {counts[int(n * 0.01)]:,}")
    print(f"p90:    {counts[int(n * 0.10)]:,}")
    print(f"p75:    {counts[int(n * 0.25)]:,}")
    print(f"p50:    {counts[int(n * 0.50)]:,}")
    print(f"p25:    {counts[int(n * 0.75)]:,}")
    print(f"Min:    {counts[-1]:,}")


# ── retrieval ─────────────────────────────────────────────────────────────────


def cmd_retrieval() -> None:
    from eval.index import VectorIndex

    DB_PATH = Path("data/docs.db")
    VECTORS_PATH = Path("data/vectors.bin")
    GOLD_PATH = Path("eval/hotpotqa_gold.json")
    SAMPLE = 20
    K = 10

    def _recall(retrieved: list[str], relevant: set[str]) -> float:
        return sum(1 for t in relevant if t in retrieved) / len(relevant)

    print("loading index...")
    index = VectorIndex(DB_PATH, VECTORS_PATH)

    gold = json.loads(GOLD_PATH.read_text())
    random.seed(99)
    sample = random.sample(gold, SAMPLE)

    hits = 0
    for i, item in enumerate(sample, 1):
        retrieved = index.search(item["question"], k=K)
        relevant = set(item["supporting_titles"])
        found = {t for t in relevant if t in retrieved}
        rec = _recall(retrieved, relevant)
        full_hit = rec == 1.0
        if full_hit:
            hits += 1

        status = "✓ HIT " if full_hit else "✗ MISS"
        print(f"\n{'─' * 72}")
        print(f"[{i:02d}/{SAMPLE}] {status}   Recall={rec:.0%}")
        print(f"Q:  {item['question']}")
        print(f"A:  {item['answer']}")
        print("\nNeeded articles:")
        for t in relevant:
            mark = "  ✓" if t in found else "  ✗"
            print(f"{mark}  {t}")
        print(f"\nTop-{K} retrieved:")
        for rank, title in enumerate(retrieved, 1):
            mark = " ✓" if title in relevant else "  "
            print(f"  {rank:2}.{mark} {title}")

    print(f"\n{'═' * 72}")
    print(
        f"Full hits (both articles in top-{K}): {hits}/{SAMPLE}  ({hits / SAMPLE:.0%})"
    )


# ── embed-watch ───────────────────────────────────────────────────────────────


def cmd_embed_watch() -> None:
    DB = "data/docs.db"
    conn = sqlite3.connect(DB)
    total = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    start_count = conn.execute(
        "SELECT COUNT(*) FROM documents WHERE status='embedded'"
    ).fetchone()[0]
    start_time = time.time()

    print(f"Total chunks: {total:,}")
    print(f"{'Embedded':>12}  {'Remaining':>12}  {'Speed':>10}  {'ETA':>12}")
    print("-" * 56)

    while True:
        count = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE status='embedded'"
        ).fetchone()[0]
        elapsed = time.time() - start_time
        done = count - start_count
        speed = done / elapsed if elapsed > 1 else 0
        remaining = total - count
        if speed > 0:
            s = int(remaining / speed)
            eta_str = f"{s // 3600}h {s % 3600 // 60:02d}m"
        else:
            eta_str = "—"
        print(
            f"{count:>12,}  {remaining:>12,}  {speed:>8.0f}/s  {eta_str:>12}",
            end="\r",
            flush=True,
        )
        time.sleep(5)


# ── beir-hybrid ───────────────────────────────────────────────────────────────


def cmd_beir_hybrid() -> None:
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

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
    CANDIDATE_POOL = 100
    RERANK_POOL = 20

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
        dcg = sum(
            qrels.get(d, 0) / math.log2(r + 2) for r, d in enumerate(retrieved[:k])
        )
        ideal = sorted(qrels.values(), reverse=True)
        idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal[:k]))
        return dcg / idcg if idcg > 0 else 0.0

    def _load_beir(
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

        qr_id, qr_cfg, qr_split = cfg["qrels_hf"]
        qrels_ds = (
            load_dataset(qr_id, qr_cfg, split=qr_split)
            if qr_cfg
            else load_dataset(qr_id, split=qr_split)
        )
        qrels: dict[str, dict[str, int]] = defaultdict(dict)
        for row in qrels_ds:
            qid, cid, score = (
                str(row["query-id"]),
                str(row["corpus-id"]),
                int(row["score"]),
            )
            if score > 0:
                qrels[qid][cid] = score
        test_qids = set(qrels.keys())

        hf_id, hf_cfg, hf_split = cfg["queries_hf"]
        queries_ds = load_dataset(hf_id, hf_cfg, split=hf_split)
        query_id_to_text: dict[str, str] = {}
        for row in queries_ds:
            qid = str(row["_id"])
            if qid in test_qids:
                query_id_to_text[qid] = row["text"]

        return doc_ids, doc_texts, query_id_to_text, dict(qrels)

    def _embed(texts: list[str], model: SentenceTransformer) -> np.ndarray:
        vecs = model.encode(
            texts,
            batch_size=EMBED_BATCH,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(vecs, dtype=np.float32)

    def _evaluate(
        cfg: dict, embed_model: SentenceTransformer, reranker: CrossEncoderReranker
    ) -> dict:
        name = cfg["name"]
        print(f"\n{'=' * 60}\nDataset: {name.upper()}\n{'=' * 60}")
        doc_ids, doc_texts, query_id_to_text, qrels = _load_beir(cfg)
        doc_text_map = dict(zip(doc_ids, doc_texts, strict=False))
        print(f"Corpus: {len(doc_ids):,} docs   Queries: {len(query_id_to_text):,}")

        corpus_vecs = _embed(doc_texts, embed_model)
        query_ids = list(query_id_to_text.keys())
        query_vecs = _embed([query_id_to_text[qid] for qid in query_ids], embed_model)

        bm25 = BM25Retriever(doc_ids, doc_texts)
        dense = DenseRetriever(doc_ids, corpus_vecs)
        ndcg_dense, ndcg_hybrid, ndcg_rerank = [], [], []

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

    print(f"Loading embed model  : {EMBED_MODEL}")
    embed_model = SentenceTransformer(EMBED_MODEL)
    print(f"Loading reranker     : {RERANK_MODEL}")
    reranker = CrossEncoderReranker(RERANK_MODEL)

    results: dict[str, dict] = {}
    for cfg in DATASETS:
        results[cfg["name"]] = _evaluate(cfg, embed_model, reranker)

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


# ── hotpotqa-hybrid ───────────────────────────────────────────────────────────


def cmd_hotpotqa_hybrid() -> None:
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
        return [r[0] for r in rows], {r[0]: r[1] for r in rows}

    print("Loading FAISS index (reads ~13.5 GB)...")
    index = VectorIndex(DB_PATH, VECTORS_PATH)

    print("\nLoading first chunk per article for BM25...")
    titles, title_to_text = _load_article_texts(DB_PATH)
    print(f"  {len(titles):,} articles")

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


# ── ablation ──────────────────────────────────────────────────────────────────


def cmd_ablation() -> None:
    import faiss
    from sentence_transformers import SentenceTransformer

    from eval.metrics import mrr, ndcg_at_k, recall_at_k
    from rag_engine.ingest.pipeline import split_text

    SOURCE_DB = Path("data/docs.db")
    ABLATION_DIR = Path("data/ablation")
    VECTOR_DIM = 384
    MODEL_NAME = "BAAI/bge-small-en-v1.5"
    N_GOLD_ARTICLES = 500
    N_NOISE_ARTICLES = 1_500
    SEED = 42

    STRATEGIES: dict[str, dict[str, int]] = {
        "A-1500": {"chunk_chars": 1500, "overlap_chars": 200},
        "B-500": {"chunk_chars": 500, "overlap_chars": 50},
        "C-300": {"chunk_chars": 300, "overlap_chars": 30},
    }

    def _load_articles(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
        gold = json.loads(Path("eval/hotpotqa_gold.json").read_text())
        gold_titles = {t for item in gold for t in item["supporting_titles"]}

        placeholders = ",".join("?" * len(gold_titles))
        gold_ids = [
            r[0]
            for r in conn.execute(
                f"SELECT DISTINCT article_id FROM documents"
                f" WHERE status='embedded' AND title IN ({placeholders}) LIMIT ?",
                (*gold_titles, N_GOLD_ARTICLES),
            ).fetchall()
        ]
        extra_ids = [
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT article_id FROM documents"
                " WHERE status='embedded' ORDER BY RANDOM() LIMIT ?",
                (N_NOISE_ARTICLES,),
            ).fetchall()
        ]
        article_ids = list({*gold_ids, *extra_ids})

        result: dict[str, dict[str, Any]] = {}
        for article_id in tqdm(article_ids, desc="loading articles", unit="art"):
            rows = conn.execute(
                "SELECT chunk_index, chunk_text, title FROM documents"
                " WHERE article_id=? ORDER BY chunk_index",
                (article_id,),
            ).fetchall()
            title = rows[0][2]
            ordered = [r[1] for r in rows]
            full_text = ordered[0] + "".join(t[200:] for t in ordered[1:])
            result[article_id] = {"title": title, "text": full_text}
        return result

    def _embed_chunks(
        model: SentenceTransformer,
        articles: dict[str, dict[str, Any]],
        chunk_chars: int,
        overlap_chars: int,
    ) -> tuple[np.ndarray, list[str]]:
        all_texts: list[str] = []
        all_titles: list[str] = []
        for data in articles.values():
            chunks = split_text(data["text"], chunk_chars, overlap_chars)
            all_texts.extend(chunks)
            all_titles.extend([data["title"]] * len(chunks))

        vectors_list: list[np.ndarray] = []
        for i in tqdm(range(0, len(all_texts), 128), desc="  embedding", unit="batch"):
            batch = all_texts[i : i + 128]
            vecs = model.encode(
                batch,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            vectors_list.append(np.asarray(vecs, dtype=np.float32))
        return np.vstack(vectors_list), all_titles

    def _run_eval(
        model: SentenceTransformer,
        index: faiss.IndexFlatIP,
        titles: list[str],
        gold: list[dict[str, Any]],
        article_titles: set[str],
    ) -> dict[str, float]:
        subset_gold = [
            item
            for item in gold
            if all(t in article_titles for t in item["supporting_titles"])
        ]
        if not subset_gold:
            return {
                "ndcg_at_10": 0.0,
                "recall_at_10": 0.0,
                "mrr": 0.0,
                "n_questions": 0,
            }

        ndcgs, recalls, mrrs = [], [], []
        for item in tqdm(subset_gold, desc="  evaluating", unit="q"):
            qvec = model.encode(
                [item["question"]],
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype(np.float32)
            _, indices = index.search(qvec, 50)

            seen: set[str] = set()
            retrieved: list[str] = []
            for idx in indices[0]:
                t = titles[idx]
                if t and t not in seen:
                    seen.add(t)
                    retrieved.append(t)
                if len(retrieved) == 10:
                    break

            relevant = set(item["supporting_titles"])
            ndcgs.append(ndcg_at_k(retrieved, relevant))
            recalls.append(recall_at_k(retrieved, relevant))
            mrrs.append(mrr(retrieved, relevant))

        return {
            "ndcg_at_10": round(sum(ndcgs) / len(ndcgs), 4),
            "recall_at_10": round(sum(recalls) / len(recalls), 4),
            "mrr": round(sum(mrrs) / len(mrrs), 4),
            "n_questions": len(subset_gold),
        }

    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)

    print(f"Loading {N_GOLD_ARTICLES + N_NOISE_ARTICLES} articles from {SOURCE_DB}...")
    conn = sqlite3.connect(SOURCE_DB)
    articles = _load_articles(conn)
    conn.close()
    print(f"Loaded {len(articles)} articles\n")

    gold = json.loads(Path("eval/hotpotqa_gold.json").read_text())
    gold_titles_set = {t for item in gold for t in item["supporting_titles"]}
    gold_article_titles = {
        d["title"] for d in articles.values() if d["title"] in gold_titles_set
    }

    print("Loading model...")
    model = SentenceTransformer(MODEL_NAME, device="mps")

    results: dict[str, dict[str, Any]] = {}
    for name, params in STRATEGIES.items():
        print(f"\n{'=' * 55}")
        print(
            f"strategy {name}"
            f"  (chunk={params['chunk_chars']}, overlap={params['overlap_chars']})"
        )

        t0 = time.time()
        vectors, titles = _embed_chunks(model, articles, **params)
        elapsed = time.time() - t0
        rate = len(titles) / elapsed
        print(f"  {len(titles):,} chunks  |  {rate:.0f} vec/s  |  {elapsed:.1f}s")

        idx_flat = faiss.IndexFlatIP(VECTOR_DIM)
        idx_flat.add(vectors)
        scores = _run_eval(model, idx_flat, titles, gold, gold_article_titles)
        scores["chunks"] = len(titles)
        scores["vec_per_s"] = round(rate, 1)
        results[name] = scores

        print(
            f"  nDCG@10={scores['ndcg_at_10']}"
            f"  Recall@10={scores['recall_at_10']}"
            f"  MRR={scores['mrr']}"
            f"  (n={scores['n_questions']})"
        )

    print(f"\n{'=' * 72}")
    header = (
        f"{'Strategy':<12} {'Chunks':>8} {'vec/s':>7}"
        f" {'nDCG@10':>9} {'Recall@10':>10} {'MRR':>8} {'N':>5}"
    )
    print(header)
    print("-" * 72)
    for name, s in results.items():
        print(
            f"{name:<12} {s['chunks']:>8,} {s['vec_per_s']:>7.0f}"
            f" {s['ndcg_at_10']:>9.4f} {s['recall_at_10']:>10.4f}"
            f" {s['mrr']:>8.4f} {s['n_questions']:>5}"
        )

    out = ABLATION_DIR / "results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nsaved → {out}")


# ── entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified diagnostics runner")
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser(
        "failures", help="Single-shot RAG on 100 HotpotQA questions; show bridge gaps"
    )
    sub.add_parser("links", help="incoming_links distribution on 50K sampled articles")
    sub.add_parser(
        "retrieval", help="Qualitative retrieval check on 20 sampled questions"
    )
    sub.add_parser(
        "embed-watch", help="Live embedding progress monitor (polls SQLite every 5s)"
    )
    sub.add_parser("beir-hybrid", help="BEIR staircase eval: dense → hybrid → rerank")
    sub.add_parser(
        "hotpotqa-hybrid", help="HotpotQA staircase eval: dense → hybrid → rerank"
    )
    sub.add_parser(
        "ablation", help="Chunk size ablation on mini-corpus (1500/500/300 chars)"
    )

    args = parser.parse_args()
    {
        "failures": cmd_failures,
        "links": cmd_links,
        "retrieval": cmd_retrieval,
        "embed-watch": cmd_embed_watch,
        "beir-hybrid": cmd_beir_hybrid,
        "hotpotqa-hybrid": cmd_hotpotqa_hybrid,
        "ablation": cmd_ablation,
    }[args.mode]()


if __name__ == "__main__":
    main()

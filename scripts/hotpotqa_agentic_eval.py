import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from eval.index import VectorIndex
from eval.metrics import exact_match, f1
from rag_engine.agent import MultiHopAgent
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
RESULTS_DIR = Path("eval/results")
RESPONSES_DIR = Path("eval/responses")
DEFAULT_N = 100

RERANK_MODEL = "BAAI/bge-reranker-base"
CANDIDATE_POOL = 100
RERANK_POOL = 20
TOP_K = 5

_SS_SYSTEM = (
    "Answer the question using ONLY the provided passages. "
    "Give a concise factual answer — a name, date, place, or yes/no. "
    "If the answer is not in the passages, say 'I don't know.'"
)

_OUT_OF_CORPUS = [
    "Who won the FIFA World Cup in 2026?",
    "What AI model won the Nobel Prize in Physics in 2025?",
    "Who is the current CEO of OpenAI in 2026?",
    "Which city hosted the 2025 Summer Olympics?",
    "What is the latest version of Python released in 2026?",
]


# ── Retrieval metrics ─────────────────────────────────────────────────────────


def recall_at_k(retrieved: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved) & relevant) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / k


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, doc_id in enumerate(retrieved[:k])
        if doc_id in relevant
    )
    ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def mrr_score(retrieved: list[str], relevant: set[str]) -> float:
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


# ── Helpers ───────────────────────────────────────────────────────────────────


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
    title_to_text = {r[0]: r[1] for r in rows}
    return titles, texts, title_to_text


def _format_passages(doc_ids: list[str], doc_texts: dict[str, str]) -> str:
    parts = []
    for doc_id in doc_ids:
        text = doc_texts.get(doc_id, "")
        if text:
            parts.append(f"[{doc_id}]\n{text[:800]}")
    return "\n\n".join(parts)


def _avg(xs: list[float]) -> float:
    return round(float(np.mean(xs)), 4) if xs else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    args = parser.parse_args()
    n: int = args.n

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

    print(f"Loading reranker: {RERANK_MODEL}\n")
    reranker = CrossEncoderReranker(RERANK_MODEL)

    def retrieve(query: str, k: int) -> list[str]:
        dense = index.search(query, k=CANDIDATE_POOL)
        sparse = bm25.retrieve(query, CANDIDATE_POOL)
        rrf = reciprocal_rank_fusion([dense, sparse], k=RERANK_POOL)
        return reranker.rerank(query, rrf, title_to_text, k)

    agent = MultiHopAgent(
        retrieve=retrieve,
        doc_texts=title_to_text,
        reranker=reranker,
    )

    gold = json.loads(GOLD_PATH.read_text())[:n]

    # ── Single-shot baseline ──────────────────────────────────────────────────
    print(f"Single-shot RAG on {n} questions...")
    ss_em: list[float] = []
    ss_f1: list[float] = []
    ss_recall: list[float] = []
    ss_precision: list[float] = []
    ss_ndcg: list[float] = []
    ss_mrr: list[float] = []
    ss_records: list[dict] = []

    for item in tqdm(gold):
        supporting_set = set(item["supporting_titles"])
        top5 = retrieve(item["question"], TOP_K)
        passages = _format_passages(top5, title_to_text)
        msg = f"Question: {item['question']}\n\nPassages:\n{passages}"
        predicted = complete(
            [{"role": "user", "content": msg}], max_tokens=64, system=_SS_SYSTEM
        ).strip()

        em = exact_match(predicted, item["answer"])
        f1_score = f1(predicted, item["answer"])
        rec = recall_at_k(top5, supporting_set)
        prec = precision_at_k(top5, supporting_set, TOP_K)
        ndcg = ndcg_at_k(top5, supporting_set, TOP_K)
        mrr = mrr_score(top5, supporting_set)

        ss_em.append(em)
        ss_f1.append(f1_score)
        ss_recall.append(rec)
        ss_precision.append(prec)
        ss_ndcg.append(ndcg)
        ss_mrr.append(mrr)
        ss_records.append(
            {
                "question": item["question"],
                "gold": item["answer"],
                "supporting_titles": item["supporting_titles"],
                "predicted": predicted,
                "retrieved": top5,
                "em": em,
                "f1": f1_score,
                "recall": rec,
                "precision": prec,
                "ndcg": ndcg,
                "mrr": mrr,
            }
        )

    # ── Multi-hop agent ───────────────────────────────────────────────────────
    print(f"\nMulti-hop agent on {n} questions...")
    mh_em: list[float] = []
    mh_f1: list[float] = []
    mh_hops: list[int] = []
    mh_hop1_recall: list[float] = []
    mh_hop1_precision: list[float] = []
    mh_hop1_ndcg: list[float] = []
    mh_hop1_mrr: list[float] = []
    mh_combined_recall: list[float] = []
    mh_abstained = 0
    mh_hallucinated = 0
    mh_reflection = 0
    mh_records: list[dict] = []

    for item in tqdm(gold):
        supporting_set = set(item["supporting_titles"])
        result = agent.answer(item["question"])
        em = exact_match(result.answer, item["answer"])
        f1_score = f1(result.answer, item["answer"])

        hop1_retrieved = result.hops[0].retrieved if result.hops else []
        combined_pool = list(dict.fromkeys(d for h in result.hops for d in h.retrieved))

        rec1 = recall_at_k(hop1_retrieved, supporting_set)
        prec1 = precision_at_k(hop1_retrieved, supporting_set, TOP_K)
        ndcg1 = ndcg_at_k(hop1_retrieved, supporting_set, TOP_K)
        mrr1 = mrr_score(hop1_retrieved, supporting_set)
        rec_combined = recall_at_k(combined_pool, supporting_set)

        mh_em.append(em)
        mh_f1.append(f1_score)
        mh_hops.append(len(result.hops))
        mh_hop1_recall.append(rec1)
        mh_hop1_precision.append(prec1)
        mh_hop1_ndcg.append(ndcg1)
        mh_hop1_mrr.append(mrr1)
        mh_combined_recall.append(rec_combined)
        if result.abstained:
            mh_abstained += 1
        if result.hallucinated_ids:
            mh_hallucinated += 1
        if result.reflection_triggered:
            mh_reflection += 1

        mh_records.append(
            {
                "question": item["question"],
                "gold": item["answer"],
                "supporting_titles": item["supporting_titles"],
                "predicted": result.answer,
                "em": em,
                "f1": f1_score,
                "hops": [
                    {"query": h.query, "retrieved": h.retrieved} for h in result.hops
                ],
                "combined_pool": combined_pool,
                "hop1_recall": rec1,
                "hop1_precision": prec1,
                "hop1_ndcg": ndcg1,
                "hop1_mrr": mrr1,
                "combined_recall": rec_combined,
                "abstained": result.abstained,
                "hallucinated_ids": result.hallucinated_ids,
                "reflection_triggered": result.reflection_triggered,
                "cited_ids": result.cited_ids,
            }
        )

    # ── Abstention test ───────────────────────────────────────────────────────
    print("\nTesting abstention on out-of-corpus queries...")
    ooc_abstained = 0
    ooc_records: list[dict] = []
    for q in _OUT_OF_CORPUS:
        r = agent.answer(q)
        status = "abstained" if r.abstained else f'answered: "{r.answer[:60]}"'
        print(f"  {status}")
        if r.abstained:
            ooc_abstained += 1
        ooc_records.append(
            {"question": q, "abstained": r.abstained, "answer": r.answer}
        )

    # ── Build summary ─────────────────────────────────────────────────────────
    summary = {
        "n": n,
        "single_shot": {
            "em": _avg(ss_em),
            "f1": _avg(ss_f1),
            "recall_at_5": _avg(ss_recall),
            "precision_at_5": _avg(ss_precision),
            "ndcg_at_5": _avg(ss_ndcg),
            "mrr": _avg(ss_mrr),
        },
        "multi_hop": {
            "em": _avg(mh_em),
            "f1": _avg(mh_f1),
            "avg_hops": round(float(np.mean(mh_hops)), 2),
            "hop1_recall_at_5": _avg(mh_hop1_recall),
            "hop1_precision_at_5": _avg(mh_hop1_precision),
            "hop1_ndcg_at_5": _avg(mh_hop1_ndcg),
            "hop1_mrr": _avg(mh_hop1_mrr),
            "combined_recall": _avg(mh_combined_recall),
            "abstained": mh_abstained,
            "hallucinations": mh_hallucinated,
            "reflections": mh_reflection,
        },
        "abstention": {
            "out_of_corpus_true_positives": ooc_abstained,
            "out_of_corpus_n": len(_OUT_OF_CORPUS),
            "in_corpus_false_positives": mh_abstained,
        },
    }

    # ── Save summary (gitignored by pattern) ──────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "hotpotqa_agentic.json"
    results_path.write_text(json.dumps({"summary": summary}, indent=2))
    print(f"\nSaved summary → {results_path}")

    # ── Save detailed responses (gitignored directory) ────────────────────────
    RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    responses_path = RESPONSES_DIR / f"hotpotqa_agentic_{timestamp}.json"
    responses_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "single_shot": ss_records,
                "multi_hop": mh_records,
                "out_of_corpus": ooc_records,
            },
            indent=2,
        )
    )
    print(f"Saved responses  → {responses_path}")

    # ── Print results ─────────────────────────────────────────────────────────
    ss = summary["single_shot"]
    mh = summary["multi_hop"]
    delta_em = round(mh["em"] - ss["em"], 4)
    delta_f1 = round(mh["f1"] - ss["f1"], 4)

    w = 52
    print(f"\n{'─' * w}")
    print(f"{'':20} {'EM':>7}  {'F1':>7}  {'Hops':>5}  {'Abstain':>7}")
    print(f"{'─' * w}")
    print(
        f"{'Single-shot':<20} {ss['em']:>7.4f}  {ss['f1']:>7.4f}"
        f"  {'1.0':>5}  {'—':>7}"
    )
    print(
        f"{'Multi-hop':<20} {mh['em']:>7.4f}  {mh['f1']:>7.4f}"
        f"  {mh['avg_hops']:>5.1f}  {mh_abstained:>7}"
    )
    print(f"{'─' * w}")
    print(f"{'Δ':<20} {delta_em:>+7.4f}  {delta_f1:>+7.4f}")

    w2 = 62
    print(f"\n{'─' * w2}")
    print(
        f"{'Retrieval metrics (avg)':30}"
        f" {'Recall@5':>9}  {'Prec@5':>7}  {'nDCG@5':>7}  {'MRR':>7}"
    )
    print(f"{'─' * w2}")
    print(
        f"{'Single-shot':30}"
        f" {ss['recall_at_5']:>9.4f}  {ss['precision_at_5']:>7.4f}"
        f"  {ss['ndcg_at_5']:>7.4f}  {ss['mrr']:>7.4f}"
    )
    print(
        f"{'Multi-hop Hop 1':30}"
        f" {mh['hop1_recall_at_5']:>9.4f}  {mh['hop1_precision_at_5']:>7.4f}"
        f"  {mh['hop1_ndcg_at_5']:>7.4f}  {mh['hop1_mrr']:>7.4f}"
    )
    print(
        f"{'Multi-hop Combined pool':30}"
        f" {mh['combined_recall']:>9.4f}  {'—':>7}  {'—':>7}  {'—':>7}"
    )
    print(f"{'─' * w2}")

    print(f"\nReflections triggered : {mh_reflection}/{n}")
    print(f"Hallucinations flagged: {mh_hallucinated}/{n}")
    print(
        f"Abstention            : {ooc_abstained}/{len(_OUT_OF_CORPUS)} "
        f"out-of-corpus  |  {mh_abstained}/{n} in-corpus (false positives)"
    )


if __name__ == "__main__":
    main()

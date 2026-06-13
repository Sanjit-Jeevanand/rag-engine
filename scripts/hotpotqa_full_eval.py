"""
HotpotQA full multi-hop eval with cost tracking.

Runs the agentic pipeline on a reproducible sample of the HotpotQA dev set.
No single-shot baseline — multi-hop only.

Output: eval/results/hotpotqa_full_YYYY-MM-DD.json
        eval/responses/hotpotqa_full_<timestamp>.json  (per-question detail)

Run:
    PYTHONPATH=src:. uv run --env-file .env \
        python scripts/hotpotqa_full_eval.py --n 20
    PYTHONPATH=src:. uv run --env-file .env \
        python scripts/hotpotqa_full_eval.py --n 1000
"""

import argparse
import json
import random
import sqlite3
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from eval.index import VectorIndex
from eval.metrics import exact_match, f1
from rag_engine.agent import MultiHopAgent
from rag_engine.cost import cost_tracker
from rag_engine.retrieval import (
    BM25Retriever,
    CrossEncoderReranker,
    reciprocal_rank_fusion,
)

# ── Config ────────────────────────────────────────────────────────────────────

SEED = 42
DB_PATH = Path("data/docs.db")
VECTORS_PATH = Path("data/vectors.bin")
BM25_INDEX_DIR = Path("data/bm25_index")
GOLD_PATH = Path("eval/hotpotqa_gold.json")
RESULTS_DIR = Path("eval/results")
RESPONSES_DIR = Path("eval/responses")

RERANK_MODEL = "BAAI/bge-reranker-base"
CANDIDATE_POOL = 100
RERANK_POOL = 20
TOP_K = 5


# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_article_texts(db_path: Path) -> dict[str, str]:
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT title, chunk_text FROM documents"
        " WHERE status='embedded' AND chunk_index=0"
    ).fetchall()
    conn.close()
    return {r[0]: r[1] for r in rows}


def _avg(xs: list[float]) -> float:
    return round(float(np.mean(xs)), 4) if xs else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    args = parser.parse_args()
    n: int = args.n

    # Reproducible sample: shuffle with fixed seed, take first n
    gold_all: list[dict] = json.loads(GOLD_PATH.read_text())
    rng = random.Random(SEED)
    sample = gold_all[:]
    rng.shuffle(sample)
    gold = sample[:n]
    print(f"Sample: {n} questions (seed={SEED}, pool={len(gold_all)})")

    print("Loading FAISS index...")
    index = VectorIndex(DB_PATH, VECTORS_PATH)

    print("Loading article texts...")
    title_to_text = _load_article_texts(DB_PATH)
    print(f"  {len(title_to_text):,} articles")

    if BM25_INDEX_DIR.exists():
        print("Loading BM25 index from cache...")
        bm25 = BM25Retriever.load(BM25_INDEX_DIR)
    else:
        titles = list(title_to_text.keys())
        texts = [title_to_text[t] for t in titles]
        print("Building BM25 index...")
        bm25 = BM25Retriever(titles, texts)
        bm25.save(BM25_INDEX_DIR)

    print(f"Loading reranker: {RERANK_MODEL}\n")
    reranker = CrossEncoderReranker(RERANK_MODEL)

    def retrieve(query: str, k: int) -> list[str]:
        dense = index.search(query, k=CANDIDATE_POOL)
        sparse = bm25.retrieve(query, CANDIDATE_POOL)
        fused = reciprocal_rank_fusion([dense, sparse], k=RERANK_POOL)
        return reranker.rerank(query, fused, title_to_text, k)

    agent = MultiHopAgent(
        retrieve=retrieve,
        doc_texts=title_to_text,
        reranker=reranker,
    )

    # ── Per-question loop ─────────────────────────────────────────────────────
    em_scores: list[float] = []
    f1_scores: list[float] = []
    hop_counts: list[int] = []
    abstained_n = 0
    hallucinated_n = 0
    reflection_n = 0
    cost_snaps: list[dict] = []
    records: list[dict] = []

    print(f"Running multi-hop agent on {n} questions...")
    for item in tqdm(gold):
        cost_tracker.reset()
        result = agent.answer(item["question"])
        snap = cost_tracker.snapshot()

        em = exact_match(result.answer, item["answer"])
        f1_score = f1(result.answer, item["answer"])
        hops = len(result.hops)

        em_scores.append(em)
        f1_scores.append(f1_score)
        hop_counts.append(hops)
        if result.abstained:
            abstained_n += 1
        if result.hallucinated_ids:
            hallucinated_n += 1
        if result.reflection_triggered:
            reflection_n += 1
        cost_snaps.append(snap.as_dict())

        records.append(
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
                "abstained": result.abstained,
                "hallucinated_ids": result.hallucinated_ids,
                "reflection_triggered": result.reflection_triggered,
                "cost": snap.as_dict(),
            }
        )

    # ── Per-hop-count breakdown ───────────────────────────────────────────────
    hop_buckets: dict[int, list[float]] = defaultdict(list)
    hop_f1_buckets: dict[int, list[float]] = defaultdict(list)
    for rec in records:
        h = len(rec["hops"])
        hop_buckets[h].append(rec["em"])
        hop_f1_buckets[h].append(rec["f1"])

    hop_breakdown = {
        str(h): {
            "n": len(hop_buckets[h]),
            "em": _avg(hop_buckets[h]),
            "f1": _avg(hop_f1_buckets[h]),
        }
        for h in sorted(hop_buckets)
    }

    # ── Cost summary ──────────────────────────────────────────────────────────
    avg_input = _avg([c["llm_input_tokens"] for c in cost_snaps])
    avg_output = _avg([c["llm_output_tokens"] for c in cost_snaps])
    avg_usd = _avg([c["estimated_usd"] for c in cost_snaps])
    total_usd = round(sum(c["estimated_usd"] for c in cost_snaps), 6)
    avg_reranker = _avg([c["reranker_calls"] for c in cost_snaps])

    # ── Build summary ─────────────────────────────────────────────────────────
    summary = {
        "date": str(date.today()),
        "n": n,
        "seed": SEED,
        "model": "gpt-4o-mini",
        "metrics": {
            "em": _avg(em_scores),
            "f1": _avg(f1_scores),
            "avg_hops": _avg([float(h) for h in hop_counts]),
            "abstained": abstained_n,
            "hallucinations": hallucinated_n,
            "reflections": reflection_n,
        },
        "hop_breakdown": hop_breakdown,
        "cost": {
            "avg_llm_input_tokens": avg_input,
            "avg_llm_output_tokens": avg_output,
            "avg_reranker_calls": avg_reranker,
            "avg_usd_per_query": avg_usd,
            "total_usd": total_usd,
            "target_usd_per_query": 0.005,
            "within_budget": avg_usd <= 0.005,
        },
    }

    # ── Print results ─────────────────────────────────────────────────────────
    m = summary["metrics"]
    print(f"\n{'─' * 48}")
    print(f"  EM         : {m['em']:.4f}")
    print(f"  F1         : {m['f1']:.4f}")
    print(f"  Avg hops   : {m['avg_hops']:.2f}")
    print(f"  Abstained  : {abstained_n}/{n}")
    print(f"  Reflected  : {reflection_n}/{n}")
    print(f"  Halluc.    : {hallucinated_n}/{n}")
    print(f"{'─' * 48}")
    print("  Hop breakdown:")
    for h, stats in hop_breakdown.items():
        em_s = f"{stats['em']:.4f}"
        f1_s = f"{stats['f1']:.4f}"
        print(f"    {h} hop(s): n={stats['n']:>4}  EM={em_s}  F1={f1_s}")
    print(f"{'─' * 48}")
    print(f"  Avg input tokens : {avg_input:.1f}")
    print(f"  Avg output tokens: {avg_output:.1f}")
    print(f"  Avg reranker calls: {avg_reranker:.1f}")
    budget_flag = "✓" if avg_usd <= 0.005 else "✗ OVER BUDGET"
    print(f"  Avg cost/query   : ${avg_usd:.5f}  {budget_flag}")
    print(f"  Total cost       : ${total_usd:.4f}")
    print(f"{'─' * 48}")

    # ── Save ──────────────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"hotpotqa_full_{date.today()}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved summary  → {out_path}")

    RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    resp_path = RESPONSES_DIR / f"hotpotqa_full_{ts}.json"
    resp_path.write_text(
        json.dumps({"summary": summary, "questions": records}, indent=2)
    )
    print(f"Saved responses → {resp_path}")


if __name__ == "__main__":
    main()

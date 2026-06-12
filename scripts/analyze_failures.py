import json
import sqlite3
from pathlib import Path

import numpy as np
from tqdm import tqdm

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
    title_to_text = {r[0]: r[1] for r in rows}
    return titles, texts, title_to_text


def _format_passages(doc_ids: list[str], doc_texts: dict[str, str]) -> str:
    parts = []
    for doc_id in doc_ids:
        text = doc_texts.get(doc_id, "")
        if text:
            parts.append(f"[{doc_id}]\n{text[:800]}")
    return "\n\n".join(parts)


def main() -> None:
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
                # True when at least one supporting article was not in top-5
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
        f"({len(failures)}/{N_QUESTIONS} failed,  "
        f"{bridge_gaps} bridge gaps)"
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


if __name__ == "__main__":
    main()

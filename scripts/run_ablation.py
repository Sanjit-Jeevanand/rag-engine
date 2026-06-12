import json
import random
import sqlite3
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from eval.metrics import mrr, ndcg_at_k, recall_at_k
from rag_engine.ingest.pipeline import split_text

SOURCE_DB = Path("data/docs.db")
ABLATION_DIR = Path("data/ablation")
VECTOR_DIM = 384
MODEL_NAME = "BAAI/bge-small-en-v1.5"
N_GOLD_ARTICLES = 500  # enough to cover all ~350 unique gold titles
N_NOISE_ARTICLES = 1_500  # random distractors
N_ARTICLES = N_GOLD_ARTICLES + N_NOISE_ARTICLES
GOLD_SAMPLE = 200
SEED = 42

STRATEGIES: dict[str, dict[str, int]] = {
    "A-1500": {"chunk_chars": 1500, "overlap_chars": 200},
    "B-500": {"chunk_chars": 500, "overlap_chars": 50},
    "C-300": {"chunk_chars": 300, "overlap_chars": 30},
}


def load_articles(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    gold = json.loads(Path("eval/hotpotqa_gold.json").read_text())
    gold_titles = {t for item in gold for t in item["supporting_titles"]}

    placeholders = ",".join("?" * len(gold_titles))
    gold_ids = [
        r[0]
        for r in conn.execute(
            f"SELECT DISTINCT article_id FROM documents"
            f" WHERE status='embedded' AND title IN ({placeholders})"
            f" LIMIT ?",
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


def embed_chunks(
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
    batch_size = 128
    for i in tqdm(
        range(0, len(all_texts), batch_size), desc="  embedding", unit="batch"
    ):
        batch = all_texts[i : i + batch_size]
        vecs = model.encode(
            batch,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vectors_list.append(np.asarray(vecs, dtype=np.float32))

    return np.vstack(vectors_list), all_titles


def build_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(VECTOR_DIM)
    index.add(vectors)
    return index


def run_eval(
    model: SentenceTransformer,
    index: faiss.IndexFlatIP,
    titles: list[str],
    gold: list[dict[str, Any]],
    article_titles: set[str],
) -> dict[str, float]:
    # only score questions where both supporting articles are in our subset
    subset_gold = [
        item
        for item in gold
        if all(t in article_titles for t in item["supporting_titles"])
    ]

    if not subset_gold:
        return {"ndcg_at_10": 0.0, "recall_at_10": 0.0, "mrr": 0.0, "n_questions": 0}

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


def main() -> None:
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)

    print(
        f"loading {N_ARTICLES} articles"
        f" ({N_GOLD_ARTICLES} gold + {N_NOISE_ARTICLES} noise)"
        f" from {SOURCE_DB}..."
    )
    conn = sqlite3.connect(SOURCE_DB)
    articles = load_articles(conn)
    conn.close()
    print(f"loaded {len(articles)} articles\n")

    gold = json.loads(Path("eval/hotpotqa_gold.json").read_text())
    gold_titles_set = {t for item in gold for t in item["supporting_titles"]}
    gold_article_titles = {
        d["title"] for d in articles.values() if d["title"] in gold_titles_set
    }

    print("loading model...")
    model = SentenceTransformer(MODEL_NAME, device="mps")

    results: dict[str, dict[str, Any]] = {}

    for name, params in STRATEGIES.items():
        print(f"\n{'=' * 55}")
        chunk = params["chunk_chars"]
        overlap = params["overlap_chars"]
        print(f"strategy {name}  (chunk={chunk}, overlap={overlap})")

        t0 = time.time()
        vectors, titles = embed_chunks(model, articles, **params)
        elapsed = time.time() - t0
        rate = len(titles) / elapsed

        print(f"  {len(titles):,} chunks  |  {rate:.0f} vec/s  |  {elapsed:.1f}s")

        index = build_index(vectors)
        scores = run_eval(model, index, titles, gold, gold_article_titles)
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
    header = f"{'Strategy':<12} {'Chunks':>8} {'vec/s':>7}"
    header += f" {'nDCG@10':>9} {'Recall@10':>10} {'MRR':>8} {'N':>5}"
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


if __name__ == "__main__":
    main()

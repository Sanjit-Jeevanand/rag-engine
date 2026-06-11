"""
Show retrieval output for 20 sampled gold questions.

For each question: the correct answer, which supporting articles were needed,
which were retrieved, and the top-5 retrieved article titles.

Run with:
    PYTHONPATH=src:. uv run python scripts/sample_retrieval.py
"""

import json
import random
from pathlib import Path

from eval.index import VectorIndex

DB_PATH = Path("data/docs.db")
VECTORS_PATH = Path("data/vectors.bin")
GOLD_PATH = Path("eval/hotpotqa_gold.json")
SAMPLE = 20
SEED = 99
K = 10


def _hit(retrieved: list[str], relevant: set[str]) -> bool:
    return all(t in retrieved for t in relevant)


def _recall(retrieved: list[str], relevant: set[str]) -> float:
    return sum(1 for t in relevant if t in retrieved) / len(relevant)


def main() -> None:
    print("loading index (this takes ~30s)...")
    index = VectorIndex(DB_PATH, VECTORS_PATH)

    gold = json.loads(GOLD_PATH.read_text())
    random.seed(SEED)
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
        print(f"\n{'─'*72}")
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

    print(f"\n{'═'*72}")
    print(
        f"Full hits (both articles found in top-{K}): {hits}/{SAMPLE}  "
        f"({hits/SAMPLE:.0%})"
    )


if __name__ == "__main__":
    main()

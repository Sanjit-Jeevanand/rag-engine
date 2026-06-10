import json
from pathlib import Path

from eval.index import VectorIndex
from eval.metrics import mrr, ndcg_at_k, recall_at_k

DB_PATH = Path("data/docs.db")
VECTORS_PATH = Path("data/vectors.bin")
GOLD_PATH = Path("eval/hotpotqa_gold.json")
RESULTS_PATH = Path("eval/results/latest.json")
K = 10


def main() -> None:
    print("loading index...")
    index = VectorIndex(DB_PATH, VECTORS_PATH)

    gold = json.loads(GOLD_PATH.read_text())
    print(f"evaluating {len(gold)} questions at k={K}...")

    ndcgs, recalls, mrrs = [], [], []

    for i, item in enumerate(gold):
        retrieved = index.search(item["question"], k=K)
        relevant = set(item["supporting_titles"])

        ndcgs.append(ndcg_at_k(retrieved, relevant, k=K))
        recalls.append(recall_at_k(retrieved, relevant, k=K))
        mrrs.append(mrr(retrieved, relevant))

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(gold)}")

    results = {
        "ndcg_at_10": round(sum(ndcgs) / len(ndcgs), 4),
        "recall_at_10": round(sum(recalls) / len(recalls), 4),
        "mrr": round(sum(mrrs) / len(mrrs), 4),
        "n_questions": len(gold),
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))

    print("\n--- results ---")
    for k, v in results.items():
        print(f"  {k}: {v}")
    print(f"\nsaved → {RESULTS_PATH}")


if __name__ == "__main__":
    main()

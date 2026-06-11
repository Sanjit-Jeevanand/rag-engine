import html
import json
import random
import sqlite3
from pathlib import Path

from datasets import load_dataset

DB_PATH = Path("data/docs.db")
OUT = Path("eval/hotpotqa_gold.json")
SAMPLE = 1000
SEED = 42


def main() -> None:
    print("downloading HotpotQA dev set from HuggingFace...")
    ds = load_dataset("hotpot_qa", "distractor", split="validation")

    conn = sqlite3.connect(DB_PATH)
    titles_in_db = {
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT title FROM documents WHERE status = 'embedded'"
        ).fetchall()
    }
    conn.close()

    random.seed(SEED)
    candidates = random.sample(list(ds), len(ds))

    gold = []
    for item in candidates:
        titles = list({html.unescape(t) for t in item["supporting_facts"]["title"]})
        if all(t in titles_in_db for t in titles):
            gold.append(
                {
                    "question": item["question"],
                    "answer": item["answer"],
                    "supporting_titles": titles,
                }
            )
        if len(gold) == SAMPLE:
            break

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(gold, indent=2))
    print(f"saved {len(gold)} questions → {OUT}")


if __name__ == "__main__":
    main()

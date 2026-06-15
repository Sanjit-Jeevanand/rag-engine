import argparse
import html
import json
import random
import sqlite3
from pathlib import Path

from datasets import load_dataset

DB_PATH = Path("data/docs.db")
SAMPLE = 1000
SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--title-list",
        type=Path,
        default=None,
        help="JSON file containing a list of titles to filter against "
        "(e.g. data/title_scores_100k.json). "
        "If omitted, uses all embedded titles in the DB.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("eval/hotpotqa_gold.json"),
        help="Output path for the gold set (default: eval/hotpotqa_gold.json)",
    )
    args = parser.parse_args()

    if args.title_list:
        titles_in_scope: set[str] = set(json.loads(args.title_list.read_text()))
        print(f"Filtering to {len(titles_in_scope):,} titles from {args.title_list}")
    else:
        conn = sqlite3.connect(DB_PATH)
        titles_in_scope = {
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT title FROM documents WHERE status = 'embedded'"
            ).fetchall()
        }
        conn.close()
        print(f"Using all {len(titles_in_scope):,} embedded titles from DB")

    print("Downloading HotpotQA dev set from HuggingFace...")
    ds = load_dataset("hotpot_qa", "distractor", split="validation")

    random.seed(SEED)
    candidates = random.sample(list(ds), len(ds))

    gold = []
    for item in candidates:
        titles = list({html.unescape(t) for t in item["supporting_facts"]["title"]})
        if all(t in titles_in_scope for t in titles):
            gold.append(
                {
                    "question": item["question"],
                    "answer": item["answer"],
                    "supporting_titles": titles,
                }
            )
        if len(gold) == SAMPLE:
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(gold, indent=2))
    print(f"Saved {len(gold)} questions → {args.out}")


if __name__ == "__main__":
    main()

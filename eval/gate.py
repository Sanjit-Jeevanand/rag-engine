import json
import pathlib
import sys

from eval.comparator import compare

RESULTS = pathlib.Path("eval/results/latest.json")


def main() -> None:
    if not RESULTS.exists():
        print(f"FAIL: {RESULTS} missing — run the eval harness first", file=sys.stderr)
        sys.exit(1)

    data: dict[str, object] = json.loads(RESULTS.read_text())

    if "ndcg_at_10" not in data:
        print(f"FAIL: no eval metrics in {RESULTS}", file=sys.stderr)
        sys.exit(1)

    compare()


if __name__ == "__main__":
    main()

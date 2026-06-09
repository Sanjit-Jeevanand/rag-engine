import json
import pathlib
import sys

RESULTS = pathlib.Path("eval/results/latest.json")
SENTINEL_KEY = "sentinel"


def main() -> None:
    if not RESULTS.exists():
        print(f"FAIL: {RESULTS} missing — run the eval harness first", file=sys.stderr)
        sys.exit(1)

    data: dict[str, object] = json.loads(RESULTS.read_text())

    if SENTINEL_KEY not in data:
        print(f"FAIL: {SENTINEL_KEY!r} key missing from {RESULTS}", file=sys.stderr)
        sys.exit(1)

    print(f"OK: eval gate passed (sentinel={data[SENTINEL_KEY]!r})")


if __name__ == "__main__":
    main()

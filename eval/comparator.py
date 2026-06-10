"""
Compare latest eval results against a saved baseline.
Raises SystemExit(1) if any tracked metric regresses beyond the allowed tolerance.
"""

import json
import sys
from pathlib import Path

METRICS = ["ndcg_at_10", "recall_at_10", "mrr"]
TOLERANCE = 0.02  # allowed drop before CI fails

LATEST_PATH = Path("eval/results/latest.json")
BASELINE_PATH = Path("eval/results/baseline.json")


def compare() -> None:
    if not BASELINE_PATH.exists():
        print("no baseline found — writing current results as baseline")
        BASELINE_PATH.write_text(LATEST_PATH.read_text())
        return

    latest = json.loads(LATEST_PATH.read_text())
    baseline = json.loads(BASELINE_PATH.read_text())

    failures = []
    for metric in METRICS:
        current = latest[metric]
        base = baseline[metric]
        if current < base - TOLERANCE:
            failures.append(
                f"  {metric}: {current:.4f} < {base:.4f} (baseline)"
                f" - {TOLERANCE} tolerance"
            )

    if failures:
        print("EVAL REGRESSION DETECTED:")
        for f in failures:
            print(f)
        sys.exit(1)

    print("eval gate passed:")
    for metric in METRICS:
        delta = latest[metric] - baseline[metric]
        sign = "+" if delta >= 0 else ""
        print(f"  {metric}: {latest[metric]:.4f}  ({sign}{delta:.4f} vs baseline)")

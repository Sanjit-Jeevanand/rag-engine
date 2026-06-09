# make lint       — check code style (ruff) and formatting; fails if anything is off
# make typecheck  — run mypy --strict; fails on any type error or missing annotation
# make test       — run pytest; fails if any test fails or no tests are collected
# make audit      — scan dependencies for known CVEs via pip-audit
# make eval-gate  — check eval/results/latest.json exists with a sentinel key
# make ci         — run all of the above in order; mirrors the GitHub Actions pipeline

.PHONY: lint typecheck test audit eval-gate ci

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy src/ --strict

test:
	uv run pytest

audit:
	uv run pip-audit

eval-gate:
	uv run python eval/gate.py

ci: lint typecheck test audit eval-gate

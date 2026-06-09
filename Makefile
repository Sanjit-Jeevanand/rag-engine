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
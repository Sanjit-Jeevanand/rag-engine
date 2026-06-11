# make lint       — check code style (ruff) and formatting; fails if anything is off
# make typecheck  — run mypy --strict; fails on any type error or missing annotation
# make test       — run pytest; fails if any test fails or no tests are collected
# make audit      — scan dependencies for known CVEs via pip-audit
# make eval-gate  — check eval/results/latest.json exists with a sentinel key
# make ci         — run all of the above in order; mirrors the GitHub Actions pipeline

.PHONY: lint typecheck test test-faiss audit eval-gate ci

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy src/ --strict

test:
	uv run pytest

test-faiss:
	uv run pytest tests/test_faiss_properties.py -v

audit:
	# CVE-2026-1839 / PYSEC-2025-217: fix requires transformers>=5.0.0rc3 but
	# optimum[onnxruntime] (sentence-transformers dep) pins transformers<4.58.0 — cannot upgrade
	# CVE-2025-69872: diskcache — no fix version released yet
	# CVE-2025-3000: torch — no fix version released yet
	uv run pip-audit \
		--ignore-vuln PYSEC-2025-217 \
		--ignore-vuln CVE-2026-1839 \
		--ignore-vuln CVE-2025-69872 \
		--ignore-vuln CVE-2025-3000

eval-gate:
	PYTHONPATH=. uv run python eval/gate.py

ci: lint typecheck test audit eval-gate

# Build Progress

## Phase 0 — Engineering Foundations

### Done
- [x] Installed uv
- [x] `uv init --name rag-engine --python 3.12` → created `pyproject.toml`
- [x] Deleted `main.py` scaffold
- [x] Updated description in `pyproject.toml`
- [x] Created directory structure: `src/`, `eval/results/`, `tests/`, `infra/`, `docs/adr/`
- [x] Created `src/rag_engine/__init__.py` (marks it as a Python package)
- [ ] Add `build-system` + hatchling config to `pyproject.toml`

### In Progress
- [ ] Wire `ruff`, `mypy --strict`, `pytest`, `pip-audit` as dev dependencies
- [ ] Write `Makefile` with `lint`, `typecheck`, `test`, `audit`, `eval-gate`, `ci` targets
- [ ] Add `pydantic-settings` config (`src/rag_engine/config.py`)
- [ ] Add structured JSON logging with `request_id` (`src/rag_engine/log.py`)
- [ ] Write eval gate placeholder (`eval/gate.py`)
- [ ] Set up `pre-commit` hooks
- [ ] Set up GitHub Actions CI (`.github/workflows/ci.yml`)
- [ ] Run `uv sync` to generate pinned `uv.lock`
- [ ] Break it: open a PR with a type error — watch CI go red

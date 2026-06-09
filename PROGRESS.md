# Build Progress

## Phase 0 — Engineering Foundations
Goal: a repo where it is impossible to merge broken, untyped, unformatted, or eval-regressing code.

### Done
- [x] Installed uv (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- [x] `uv init --name rag-engine --python 3.12` → created `pyproject.toml`
- [x] Deleted `main.py` scaffold
- [x] Updated description in `pyproject.toml`
- [x] Added `build-system` + hatchling config to `pyproject.toml`
- [x] Added dev dependencies: `ruff`, `mypy`, `pytest`, `pip-audit`, `pre-commit`
- [x] Created directory structure: `src/rag_engine/`, `eval/results/`, `tests/`, `infra/`, `docs/adr/`
- [x] Created `src/rag_engine/__init__.py`
- [x] Run `uv sync` → generated pinned `uv.lock`
- [x] Written `Makefile` with `lint`, `typecheck`, `test`, `audit`, `eval-gate`, `ci` targets + descriptions
- [x] Created `src/rag_engine/config.py` — pydantic-settings, 12-factor, env-driven
- [x] Created `src/rag_engine/log.py` — structlog, JSON output, request_id via ContextVar
- [x] Created `eval/gate.py` — fails CI if eval/results/latest.json missing or lacks sentinel
- [x] Created `eval/results/latest.json` with sentinel
- [x] Created `tests/test_smoke.py` — smoke tests for config and logging
- [x] `make ci` passes end to end: lint → typecheck → test → audit → eval-gate

### In Progress
- [ ] Set up `pre-commit` hooks (`.pre-commit-config.yaml`)
- [ ] Set up GitHub Actions CI (`.github/workflows/ci.yml`)
- [ ] Break it: open a PR with a type error — watch CI go red

### Commands Run
```bash
uv init --name rag-engine --python 3.12        # 1. initialise project
rm main.py                                      # 2. remove scaffold
uv sync                                         # 3. install deps + generate uv.lock
make lint                                       # 4. ruff check + format check → passed
make typecheck                                  # 5. mypy --strict across src/ → passed
make audit                                      # 6. pip-audit security scan → no CVEs
make eval-gate                                  # 7. eval gate → FAIL (no results file)
echo '{"sentinel": true}' > eval/results/latest.json  # 8. create sentinel file
make eval-gate                                  # 9. eval gate → passed
rm eval/results/latest.json                    # 10. deliberately broke the gate → FAIL
echo '{"sentinel": true}' > eval/results/latest.json  # 11. restored it
make test                                       # 12. pytest → ModuleNotFoundError (package not installed)
conda deactivate                                # 13. deactivate conflicting conda env
rm -rf .venv && uv sync                        # 14. recreate venv clean
make test                                       # 15. 2 tests passed
uv run ruff format .                           # 16. auto-fix formatting before ci
make ci                                         # 17. full pipeline green
```

### Concepts Covered
- `uv init` creates the project skeleton (`pyproject.toml`)
- `uv sync` pins exact versions into `uv.lock` — the "works on all machines" guarantee
- `src/` layout forces the package to be installed before tests can import it — catches packaging bugs
- `__init__.py` tells Python to treat a folder as an importable package
- `build-system` tells uv which tool to use to install your own code as a package (hatchling)
- `[dependency-groups]` are dev-only tools — they don't ship with the package
- `Makefile` gives one entry point for every quality check, identical locally and in CI
- `pydantic-settings` — typed, validated config from env vars; fails loudly at startup if misconfigured
- `structlog` + `ContextVar` — structured JSON logs with request_id threaded through automatically
- Smoke tests — prove the package imports and initialises without crashing
- CI = Continuous Integration; CD = Continuous Deployment (Phase 10)

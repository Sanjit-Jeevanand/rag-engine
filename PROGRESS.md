# Build Progress

## Phase 0 — Engineering Foundations ✅
Goal: a repo where it is impossible to merge broken, untyped, unformatted, or eval-regressing code.

### Files Created
- `pyproject.toml` — project config, dependencies, tool settings
- `uv.lock` — pinned lockfile
- `.python-version` — pins Python 3.12
- `Makefile` — lint, typecheck, test, audit, eval-gate, ci targets
- `.pre-commit-config.yaml` — ruff, mypy, pre-commit-hooks
- `.github/workflows/ci.yml` — full CI pipeline
- `.gitignore` — excludes __pycache__, .venv, .env, .vscode, .convo
- `src/rag_engine/__init__.py`
- `src/rag_engine/config.py` — pydantic-settings, env prefix RAG_
- `src/rag_engine/log.py` — structlog JSON logging, request_id via ContextVar
- `eval/gate.py` — sentinel check, blocks CI if missing
- `eval/results/latest.json` — {"sentinel": true}
- `tests/test_smoke.py` — 2 smoke tests
- `PROGRESS.md`, `steps.md`

### Commands Run
```bash
uv init --name rag-engine --python 3.12        # initialise project
rm main.py                                      # remove scaffold
uv sync                                         # install deps + generate uv.lock
make lint                                       # ruff check + format → passed
make typecheck                                  # mypy --strict → passed
make audit                                      # pip-audit → no CVEs
make eval-gate                                  # → FAIL (no results file)
echo '{"sentinel": true}' > eval/results/latest.json
make eval-gate                                  # → passed
rm eval/results/latest.json                    # deliberately broke → FAIL
echo '{"sentinel": true}' > eval/results/latest.json
make test                                       # → ModuleNotFoundError (not installed)
conda deactivate                                # deactivate conflicting conda env
rm -rf .venv && uv sync                        # recreate venv clean
make test                                       # → 2 passed
uv run ruff format .                           # auto-fix formatting
make ci                                         # full pipeline green
git commit --no-verify -m "test: type error"   # bypass pre-commit → CI caught it
```

### Concepts Covered
- `uv init` / `uv sync` / `uv add` — project init, lockfile, adding deps
- `src/` layout — forces proper install before import; catches packaging bugs
- `__init__.py` — marks folder as importable package
- `build-system` + hatchling — installs your own code as a package
- `[dependency-groups]` — dev-only tools, don't ship with the package
- `Makefile` — one entry point for all quality checks, identical locally and in CI
- `pydantic-settings` — typed config from env vars, fails loudly at startup
- `structlog` + `ContextVar` — structured JSON logs with request_id
- Smoke tests — prove package imports without crashing
- pre-commit + GitHub Actions — two independent enforcement layers
- `--no-verify` bypasses pre-commit but not CI

---

## Phase 1 — Corpus Ingestion and Embedding Pipeline ⚙️
Goal: turn raw Wikipedia text into vectors you can query — and understand every step.

### Files Created
- `src/rag_engine/ingest/__init__.py`
- `src/rag_engine/ingest/schema.py` — SQLite schema, WAL mode, chunk-aware, status index
- `src/rag_engine/ingest/downloader.py` — streaming httpx download, .tmp rename, idempotent
- `src/rag_engine/ingest/parser.py` — WikiArticle dataclass, CirrusSearch paired-line parser, namespace filter, gzip support
- `src/rag_engine/ingest/pipeline.py` — split_text, run_pipeline, INSERT OR IGNORE, batch commits
- `src/rag_engine/ingest/embedder.py` — batched bge-large encoding, float32 binary file, status + offset + checksum updates
- `scripts/download_wiki_dump.py` — standalone stdlib download script, progress bar, resume support
- `tests/test_schema.py` — table creation + idempotency tests
- `tests/test_downloader.py` — skip-if-exists + mock HTTP tests
- `tests/test_parser.py` — 5 tests including namespace filter, categories join, gzip
- `tests/test_pipeline.py` — 5 tests: chunking, overlap, insert, idempotency
- `tests/test_embedder.py` — 5 tests: vector file shape, memmap load, status, offsets, idempotency
- `tests/test_integration.py` — end-to-end: 100 synthetic articles → parse → chunk → embed → assert shape + offsets + checksums
- `tests/fixtures/sample_snapshot.jsonl` — CirrusSearch paired-line format, namespace=1 entry for skip test

### Commands Run
```bash
git checkout -b phase/1-ingestion
mkdir -p src/rag_engine/ingest
touch src/rag_engine/ingest/__init__.py
uv add httpx                                    # adds httpx, updates uv.lock
uv add sentence-transformers                    # adds bge-large embedding model
uv run ruff check --fix . && uv run ruff format . # fix import ordering
make typecheck                                  # → 9 files clean
make test                                       # → 22 passed
make ci                                         # → full pipeline green

# start Wikipedia dump download in background (~20 GB)
# or use the standalone script (no uv needed):
python scripts/download_wiki_dump.py
```

### To Do
- [x] SQLite schema (`src/rag_engine/ingest/schema.py`)
- [x] Downloader (`src/rag_engine/ingest/downloader.py`)
- [x] Parser — CirrusSearch paired-line format, namespace=0 filter, gzip support (`src/rag_engine/ingest/parser.py`)
- [x] Tests for parser (`tests/test_parser.py`) — 5 tests, fixture in `tests/fixtures/`
- [x] Ingestion pipeline — parse → chunk → insert SQLite (`src/rag_engine/ingest/pipeline.py`)
- [x] Standalone download script (`scripts/download_wiki_dump.py`)
- [x] Embedding worker — batched bge-large, float32 binary file, checksum (`src/rag_engine/ingest/embedder.py`)
- [x] Integration test — 100 synthetic articles end-to-end, assert shape + offsets + checksums (`tests/test_integration.py`)
- [ ] Break it: corrupt a doc mid-ingest, restart → confirm only that doc re-embeds

### Concepts Covered
- JSONL format — one JSON object per line, parsed lazily
- Generator (`yield`) — keeps memory flat across 6M+ articles
- Streaming HTTP download — 1MB chunks, never loads full file into RAM
- `.tmp` → rename pattern — destination is always complete or absent
- `httpx` over `requests` — native async support needed for embedding worker
- `unittest.mock.patch` — replaces real HTTP calls in tests; keeps tests fast and offline
- SQLite WAL mode — concurrent reads during writes
- `vector_offset` — O(1) lookup: position of chunk's vector in binary file
- `IF NOT EXISTS` — idempotent schema init, safe to call on every pipeline restart
- Chunk-aware schema from day one — avoids painful migration when full articles added
- `split_text` — sliding window chunking, 1500 chars / 200 overlap
- `INSERT OR IGNORE` — idempotent pipeline, safe to restart after crash
- Batch commits — 1000 rows per commit, avoids per-row fsync overhead
- `chunk_count` — stored per row so completeness checks don't need a full table scan
- bge-large-en-v1.5 — 512-token context ceiling; 1500-char chunks stay safely under it
- CirrusSearch format — paired lines per article (index line + content line); namespace=0 = mainspace articles only
- `categories` as list → joined string — SQLite TEXT column; joined with space for storage
- Standalone download script — stdlib only, HTTP Range header for resume, no uv required
- gzip support in parser — `gzip.open` vs `open` based on `.gz` suffix; real dump is compressed
- `normalize_embeddings=True` — bge-large trained for cosine similarity; normalise at encode time
- Checksum — sha256 of chunk_text per row; detect stale embeddings without re-reading corpus
- Batch size — controls speed + RAM, not correctness; larger = faster matrix multiply on Neural Engine
- `vectors.tobytes()` — raw float32 bytes appended to binary file; no headers, no format overhead
- Integration test — synthetic `.json.gz` fixture; tests full pipeline without real download

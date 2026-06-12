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

## Phase 1 — Corpus Ingestion and Embedding Pipeline ✅
Goal: turn raw Wikipedia text into vectors you can query — and understand every step.

### What Actually Happened (vs plan)
- Switched model: `bge-large` → `bge-small-en-v1.5` (384-dim). bge-large OOM'd at batch=512 consuming 66 GB on a 48 GB machine. bge-small runs at ~88 vec/s on MPS — 3× faster, 62.2 MTEB vs 63.5, acceptable tradeoff.
- Reduced corpus: 7.1M articles → top **1M by `incoming_links`** (Wikipedia's organic importance signal — backlink count). 27.9M chunks was infeasible (273 h to embed). 8.8M chunks at 88 vec/s = ~28 h.
- Two-pass ingestion: pass 1 scans all 7.1M articles collecting `incoming_links` scores, pass 2 inserts only the top 1M. Cutoff was `incoming_links >= 155`.
- Fixed parser: `EOFError` + `gzip.BadGzipFile` at end of truncated/trailing-garbage gzip stream. Reads moved inside try/except.
- Embedder tuned for long-running stability: `DB_FETCH=4096` (one commit per 4096 chunks, not per 256), `vectors.bin` held open for the full run, `PRAGMA synchronous=NORMAL` removes per-commit fsyncs, crash-safety truncation on startup aligns file with DB state, `torch.mps.empty_cache()` every 200 loops prevents MPS allocator pool from growing unboundedly over millions of batches.

### Files Created
- `src/rag_engine/ingest/__init__.py`
- `src/rag_engine/ingest/schema.py` — SQLite schema, WAL mode, `idx_status` index
- `src/rag_engine/ingest/parser.py` — WikiArticle dataclass, `incoming_links` field, EOFError/BadGzipFile handling
- `src/rag_engine/ingest/pipeline.py` — `split_text`, `run_pipeline`, INSERT OR IGNORE, batch commits
- `src/rag_engine/ingest/embedder.py` — bge-small MPS, 4096-row fetch, crash-safe truncation, MPS flush
- `scripts/run_pipeline.py` — two-pass top-1M ingestion
- `scripts/run_embedder.py` — run embedder from CLI
- `scripts/watch_embed.py` — live embed progress monitor
- `scripts/sample_links.py` — incoming_links distribution sampler (found cutoff = 155)
- `scripts/dashboard.py` — stdlib live dashboard on :8765

### Numbers
- Articles ingested: **1,000,000**
- Chunks: **8,797,519**
- Vector dim: **384** (bge-small-en-v1.5)
- Embed speed: **~88 vec/s** on MPS (M4 Pro)
- Embed ETA: **~28 h** total; currently ~2.5M / 8.8M done
- Corpus size on disk: `data/docs.db` ~4 GB, `data/vectors.bin` ~13.5 GB at completion

### Commands Run
```bash
git checkout -b phase/1-ingestion
uv add sentence-transformers torch

# sample incoming_links distribution to find cutoff
PYTHONPATH=src uv run python scripts/sample_links.py
# p50=434, p75=1159, p90=2868 → cutoff 155 keeps top 1M

# two-pass ingestion
PYTHONPATH=src uv run python scripts/run_pipeline.py
# Pass 1: scan 7.1M articles → collect scores → sort → keep top 1M IDs
# Pass 2: insert only those 1M articles → 8,797,519 chunks

# start embedder (resumable — safe to kill and restart)
mkdir -p logs
nohup uv run python scripts/run_embedder.py > logs/embed.log 2>&1 &

# watch progress
PYTHONPATH=src uv run python scripts/watch_embed.py
```

### Concepts Covered
- CirrusSearch format — paired lines (index + content); namespace=0 = mainspace only
- Generator (`yield`) — keeps memory flat across 7.1M articles
- `incoming_links` — Wikipedia's organic importance signal (backlink count); used to rank and filter corpus
- Two-pass pipeline — scan all for scores, sort, whitelist top N, re-scan to insert
- `INSERT OR IGNORE` — idempotent pipeline, safe to restart after crash
- SQLite WAL mode — concurrent readers safe while embedder writes
- `vector_offset` — sequential index into binary file; byte position = `offset * 384 * 4`
- `PRAGMA synchronous=NORMAL` — safe with WAL, removes per-commit fsync stalls
- Crash-safety truncation — on restart, file is truncated to match DB committed state
- MPS allocator pool — grows unboundedly without `torch.mps.empty_cache()`; flush every N loops
- `normalize_embeddings=True` — bge-small trained for cosine; normalize at encode time, not query time
- `vf.flush()` before `conn.commit()` — file bytes must be durable before DB claims them

---

## Phase 2 — Evaluation Harness ✅
Goal: wire the eval framework before building any retrieval — so every future change is measured, not vibes.

### Files Created
- `eval/metrics.py` — pure-Python nDCG@k, Recall@k, MRR, Exact Match, F1
- `eval/index.py` — VectorIndex: loads vectors.bin into FAISS IndexFlatIP, maps offset→title, dedupes by article
- `eval/hotpotqa_eval.py` — loops 1,000 gold questions, scores all three metrics, writes latest.json
- `eval/comparator.py` — diffs latest vs baseline, fails CI if any metric drops >0.02
- `eval/__init__.py` — makes eval/ an importable package (fixes PYTHONPATH=. eval runs)
- `scripts/seed_gold_set.py` — downloads HotpotQA distractor split, filters to embedded titles, saves 1,000 questions
- `scripts/sample_retrieval.py` — qualitative view: 20 sampled questions with hit/miss breakdown
- `eval/hotpotqa_gold.json` — 1,000 gold questions + supporting titles
- `eval/results/baseline.json` — frozen reference metrics
- `eval/results/latest.json` — most recent eval run

### Numbers
- Gold questions: **1,000** (HotpotQA distractor, both supporting articles embedded)
- Final baseline (full 8.8M chunk index):
  - **nDCG@10: 0.4618** — Reasonable
  - **Recall@10: 0.478** — Poor (target for Phase 3)
  - **MRR: 0.5994** — Strong
- Regression tolerance: **0.02** — any drop fails CI

### Key Decisions
- Coverage-filtered eval: only score questions where both supporting articles are embedded — otherwise the gold set is unfair as embedding progresses
- Article-level dedup in search: one article has many chunks; score at title level to match HotpotQA labels
- 1,000 questions chosen over 200 (too low variance) and 7,400 (too slow for CI — ~30 min per run)
- FAISS IndexFlatIP: exact search, 100% ANNS recall; will upgrade to IndexHNSWFlat after Phase 3

### Commands Run
```bash
PYTHONPATH=src uv run python scripts/seed_gold_set.py     # build gold set
PYTHONPATH=src:. uv run python eval/hotpotqa_eval.py      # score 1,000 Qs
make eval-gate                                            # auto-promoted to baseline
PYTHONPATH=src:. uv run python scripts/sample_retrieval.py # qualitative check
```

### Concepts Covered
- nDCG@k — discounted cumulative gain; rewards correct docs ranked higher
- Recall@k — fraction of ground-truth articles found in top-k
- MRR — reciprocal rank of the first correct hit
- Exact Match / F1 — generation metrics (implemented now, used in Phase 3+)
- HotpotQA distractor setting — multi-hop; needs two supporting articles per question
- FAISS IndexFlatIP — inner product on L2-normalised vectors = cosine similarity
- ANNS recall vs retrieval recall — two different metrics, completely different meaning
- Regression gate — retrieval quality as a first-class CI contract

---

## BEIR Baseline (dense-only, pre-Phase 3) ✅
Goal: establish an out-of-domain retrieval baseline on standard BEIR datasets before adding any retrieval improvements.

### Files Created
- `scripts/beir_eval.py` — downloads SciFact + NFCorpus from HuggingFace, embeds with bge-small-en-v1.5,
  builds per-dataset FAISS IndexFlatIP, scores nDCG@10 / Recall@10 / MRR on test qrels
- `eval/results/beir_baseline.json` — frozen baseline numbers

### Numbers (bge-small, dense-only IndexFlatIP)
| Dataset  | nDCG@10 | Recall@10 | MRR    | Queries |
|----------|---------|-----------|--------|---------|
| SciFact  | 0.7243  | 0.8412    | 0.6924 | 300     |
| NFCorpus | 0.3409  | 0.1623    | 0.5402 | 323     |

### Interpretation
- **SciFact (scientific claims)**: nDCG 0.72 is strong — bge-small handles short biomedical claims well;
  Recall@10 0.84 means the supporting document is in the top 10 for 84% of test queries
- **NFCorpus (medical/nutrition)**: nDCG 0.34 is typical for this notoriously hard dataset (BEIR paper reports
  ~0.32 for BM25, ~0.33 for dense models); Recall@10 0.16 reflects graded relevance and a large qrels set
- These numbers are the dense-only floor; Phase 5 hybrid + reranking should improve NFCorpus most

### Commands Run
```bash
PYTHONPATH=src:. uv run python scripts/beir_eval.py
```

---

## Phase 3 — FAISS Index Comparison ✅
Goal: compare IndexFlatL2 vs IndexHNSWFlat vs IndexIVFPQ on recall-latency tradeoff; persist best index to disk.

### Files Created
- `scripts/benchmark_faiss.py` — benchmarks all three index types on 1,000 held-out queries; reports recall@10 vs exact + P50/P99 latency
- `scripts/build_hnsw_index.py` — builds IndexHNSWFlat (M=32, efC=200) over 8.8M vectors and saves to `data/hnsw.index`
- `tests/test_faiss_properties.py` — Hypothesis property tests: k results always returned, avg recall ≥ 90%, deterministic, break-it ef=2 gap test

### Benchmark Results (1,000 queries, 8.8M vectors)
| Index | recall@10 | p50 ms | p99 ms |
|-------|-----------|--------|--------|
| IndexFlatL2 (exact) | 1.0000 | 117.8 | 127.6 |
| IndexHNSWFlat ef=32 | 0.9760 | 0.229 | 0.403 |
| **IndexHNSWFlat ef=64 ✓** | **0.9857** | **0.387** | **0.682** |
| IndexHNSWFlat ef=128 | 0.9886 | 0.902 | 7.123 |
| IndexHNSWFlat ef=256 | 0.9917 | 1.298 | 2.582 |
| IndexIVFPQ nprobe=8 | 0.6374 | 0.355 | 0.507 |
| IndexIVFPQ nprobe=128 | 0.6878 | 2.920 | 3.694 |

### Key Decisions
- **IndexHNSWFlat ef=64** chosen for the serving path: 98.6% ANNS recall, 0.39ms p50, 300× faster than exact
- **IVFPQ rejected**: recall ceiling at ~69% regardless of nprobe — PQ compression (32×) is too lossy for 8.8M vectors
- **Eval gate stays on IndexFlatIP**: switching to HNSW dropped nDCG 0.46 → 0.37 due to dedup multiplier mismatch; exact search is non-negotiable for reproducible CI metrics
- **HNSW persisted to disk**: `data/hnsw.index` (15.9 GB); build takes ~21 min once, loads in seconds after
- **FAISS property tests isolated**: building 3× 100K-vector HNSW indexes segfaults with rest of suite; `make test-faiss` runs them separately

### Commands Run
```bash
PYTHONPATH=src:. uv run python scripts/benchmark_faiss.py   # full benchmark
uv run python scripts/build_hnsw_index.py                   # build + persist HNSW (~21 min)
make test-faiss                                             # property tests
make ci                                                     # full pipeline
```

---

## Phase 4 — Throughput Engineering ✅
Goal: establish a concurrency baseline, profile the bottleneck, batch queries, and pin production settings in config.

### Files Created / Modified
- `scripts/throughput_baseline.py` — measures raw FAISS QPS at concurrency 1/4/8/10/12/16; loads HNSW from disk; tqdm progress bar per run
- `src/rag_engine/config.py` — added serving settings: `hnsw_path`, `hnsw_ef_search`, `search_workers`, `faiss_omp_threads`

### Concurrency Baseline Results (HNSW ef=64, 8.8M vectors, 10s runs)
| Concurrency | QPS | p50 ms | p99 ms | Notes |
|-------------|-----|--------|--------|-------|
| 1 | 918 | 0.353 | 13.594 | GIL + Python overhead dominate; p99 spike = GC/scheduler |
| 4 | 10,192 | 0.388 | 0.666 | FAISS releases GIL → true parallelism + Python overhead hidden |
| **8 ✓** | **18,471** | **0.429** | **0.736** | **All 8 performance cores saturated — Pareto knee** |
| 12 | 20,694 | 0.533 | 1.370 | Efficiency cores: +12% QPS, +2× p99 |
| 16 | 20,815 | 0.546 | 4.759 | Plateau; p99 near 5ms budget |

FlatIP (exact): 16 QPS at all concurrency levels — memory-bandwidth-bound (13.5 GB scan/query).

### Key Decisions
- **`search_workers = 8`**: saturates all performance cores; efficiency cores (c=12+) add only +12% QPS at 2× p99 cost
- **`faiss_omp_threads = 1`**: FAISS internal OMP parallelism must be 1 when running c=8 threads — otherwise thread oversubscription causes scheduling contention
- **Super-linear scaling explained**: at c=1, Python loop overhead (perf_counter, array slice, tqdm) is serialised with each FAISS call; at c>1, threads interleave so Python overhead in one thread is hidden behind FAISS work in another
- **FlatIP doesn't scale with concurrency**: memory bandwidth is a shared physical resource; 4 threads scanning 13.5 GB simultaneously gives 4× worse p50 with identical total throughput

### Production Settings (saved in `src/rag_engine/config.py`, env-overridable via `RAG_*`)
| Setting | Value | Reason |
|---------|-------|--------|
| `hnsw_ef_search` | 64 | Phase 3 Pareto knee: 98.6% recall, 0.387ms p50 |
| `search_workers` | 8 | One per performance core; efficiency cores add noise |
| `faiss_omp_threads` | 1 | Avoid oversubscription at c=8 |
| `hnsw_path` | `data/hnsw.index` | 15.9 GB persisted index |

### py-spy Profiling
Profiled 5,000 single-vector search calls under py-spy. Flamegraph showed three zones:
- ~31% `np.fromfile` (vector load — I/O, nothing to optimise)
- ~44% `faiss.read_index` (HNSW load from disk — I/O)
- ~25% search loop: `main → replacement_search → search`

Key finding: `replacement_search` (faiss/class_wrappers.py) is the SWIG wrapper paid on every `index.search()` call regardless of batch size. At single-vector it's paid 5,000×; with batching it's paid once per batch.

### Batched Search Results (HNSW ef=64, c=1, 10s runs)
| batch | QPS | speedup vs baseline |
|-------|-----|---------------------|
| 1 | 2,616 | 2.8× |
| 8 | 11,009 | 12.0× |
| 32 | 17,179 | 18.7× |
| 64 | 18,958 | 20.6× |
| 128 | 20,126 | 21.9× |
| **256 ✓** | **20,822** | **22.7×** |
| 512 | 20,217 | 22.0× ↓ |

batch=512 regresses: 512 × 384 floats ≈ 768 KB, starts spilling out of L2 cache mid-multiply.

### Why batch=256 c=1 is NOT the production serving config
Despite winning on raw QPS, batching adds *queuing latency* — request #1 waits for 255 more to arrive before search fires. At 20K QPS a batch of 256 takes ~12ms to fill, blowing the 5ms P99 budget. In addition, the encoder (10–30ms per query) dominates the serving path; FAISS is never the bottleneck in real traffic. **Serving stays at c=8, batch=1.** Batch=256 is useful for offline/bulk workloads only.

### IVFPQ vs HNSW — Three-Way Tradeoff
| Index | Memory | Recall@10 | QPS c=8 | p99 ms |
|-------|--------|-----------|---------|--------|
| HNSW ef=64 | 15.9 GB | 0.9857 | 17,070 | 0.992 |
| IVFPQ nprobe=8 | 0.50 GB | 0.6374 | 22,740 | 0.675 |
| IVFPQ nprobe=128 | 0.50 GB | 0.6878 | 2,141 | 6.480 |

IVFPQ nprobe=8 beats HNSW on raw QPS (+33%) and memory (32×) — 0.5 GB stays fully hot in RAM. But recall ceiling is 68.8% regardless of nprobe; every other nprobe setting is strictly dominated by HNSW (worse recall AND worse QPS). **HNSW wins for production**: 35-point recall gap cannot be traded for 33% QPS.

### Perf Regression Gate
- `scripts/perf_check.py` — 2s warmup + 5s measurement at c=8, ef=64; exits 1 if QPS drops > 10% or p99 > 5ms
- `eval/results/perf_baseline.json` — baseline: 19,463 QPS, p99=0.706ms, floor=17,516 QPS
- `make perf` — runs the gate locally; NOT in `make ci` (no index file on CI runners, hardware varies)

### Production Settings (in `src/rag_engine/config.py`)
| Setting | Value | Reason |
|---------|-------|--------|
| `hnsw_ef_search` | 64 | Phase 3 Pareto knee: 98.6% recall, 0.387ms p50 |
| `search_workers` | 8 | One per performance core; efficiency cores add noise |
| `faiss_omp_threads` | 1 | Avoid oversubscription at c=8 |
| `hnsw_path` | `data/hnsw.index` | 15.9 GB persisted index |

### Commands Run
```bash
PYTHONPATH=src:. uv run python scripts/throughput_baseline.py   # concurrency sweep
sudo .venv/bin/py-spy record --output flamegraph.svg -- \
    .venv/bin/python scripts/profile_target.py                  # flamegraph
PYTHONPATH=src:. uv run python scripts/throughput_batched.py    # batching sweep
PYTHONPATH=src:. uv run python scripts/throughput_ivfpq.py      # IVFPQ comparison
PYTHONPATH=src:. uv run python scripts/perf_check.py --record   # record baseline
make perf                                                        # gate passes
```

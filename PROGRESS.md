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

---

## Phase 6 — Agentic Multi-hop RAG ✅
Goal: replace single-shot RAG with an iterative retrieve-reason-retrieve loop that handles bridge questions — where the answer requires two supporting articles that never co-occur in one document.

### Files Created / Modified
- `src/rag_engine/agent/__init__.py` — package re-exporting AgentResult, Hop, MultiHopAgent, complete
- `src/rag_engine/agent/llm.py` — lazy singleton OpenAI client; `complete()` wraps chat completions
- `src/rag_engine/agent/loop.py` — MultiHopAgent: full hop loop, bridge extraction, reflection, citation grounding, abstention
- `src/rag_engine/retrieval/reranker.py` — added `scores()` method returning raw float array (unblocks abstention)
- `tests/test_agent.py` — 4 unit tests: abstention, hallucination flagging, hop cap, reflection trigger
- `scripts/analyze_failures.py` — single-shot failure analysis on 100 HotpotQA questions
- `scripts/hotpotqa_agentic_eval.py` — side-by-side eval: single-shot vs multi-hop, retrieval metrics, abstention test
- `.gitignore` — added `eval/responses/` (detailed per-question logs, not committed)

### Numbers
| Metric | Single-shot | Multi-hop | Δ |
|--------|-------------|-----------|---|
| EM | 0.29 | 0.49 | **+0.20** |
| F1 | 0.45 | 0.60 | +0.15 |
| Recall@5 (hop 1) | 0.68 | 0.68 | — |
| Recall (combined) | — | 0.785 | +10.5 pp |
| nDCG@5 | 0.70 | 0.70 | — |
| MRR | 0.85 | 0.85 | — |
| Out-of-corpus abstention | — | 4/5 | — |
| Hallucinations | — | 0/100 | — |

### Key Decisions
- **GPT-4o-mini over Haiku**: already had OpenAI API key; cheapest capable model (~$0.15/1M tokens input)
- **Structured `ANSWER: <concise>` format**: verbose cited prose killed EM (0.31 → 0.00); explicit line extraction recovered it (+0.20 vs baseline)
- **LLM-based abstention over reranker-only**: BGE reranker scores stay above -4.0 even for out-of-corpus queries (Wikipedia is too broad); LLM saying "I cannot answer" is a stronger signal
- **Bridge query extraction**: LLM reads hop-1 passages and outputs a targeted search query for the missing entity; this is what drives combined Recall from 0.68 → 0.785
- **Reflection cap at max_hops=3**: prevents runaway LLM loops; 12/100 questions triggered reflection
- **BM25 index persistence**: `bm25s` save/load API; builds once (~10s), loads from `data/bm25_index/` in <1s on every subsequent run

### Concepts Covered
- Multi-hop QA — bridge questions require two supporting articles that never co-occur in one document
- Bridge entity extraction — LLM reads hop-1 passages and outputs a targeted search query
- Self-reflection — second LLM call checks if the answer is fully supported; triggers extra hop if not
- Citation grounding — verify cited passage IDs are in the retrieved set; flag hallucinations
- Abstention — two-layer: reranker score threshold (fast, pre-LLM) + LLM canonical string (semantic)
- Lazy singleton pattern — defer `OpenAI()` init to first call so tests can mock without a key
- `side_effect=list` in unittest.mock — sequential returns for multiple LLM calls in one test
- Patch target is import location — patch `rag_engine.agent.loop.complete`, not `rag_engine.agent.llm.complete`

### Commands Run
```bash
uv add openai
PYTHONPATH=src:. uv run --env-file .env python -c "from rag_engine.agent.llm import complete; ..."   # smoke test
uv run pytest tests/test_agent.py -v                                                                   # 4 passed
PYTHONPATH=src:. uv run --env-file .env python scripts/analyze_failures.py                            # single-shot baseline
PYTHONPATH=src:. uv run --env-file .env python scripts/hotpotqa_agentic_eval.py --n 20               # smoke
PYTHONPATH=src:. uv run --env-file .env python scripts/hotpotqa_agentic_eval.py --n 100              # full eval
```

---

## Phase 7 — Full Benchmark Run & Cost Accounting ✅
Goal: produce credible, reproducible benchmark numbers and prove cost is production-viable.

### Files Created / Modified
| File | Type | Change |
|------|------|--------|
| `src/rag_engine/cost.py` | library (new) | CostTracker + CostSnapshot; thread-local, reset per question |
| `src/rag_engine/agent/llm.py` | library | Instrumented `complete()` with `response.usage` token capture |
| `src/rag_engine/retrieval/reranker.py` | library | Added `cost_tracker.add_reranker()` to `scores()` and `rerank()` |
| `scripts/beir_eval.py` | script (rewrite) | Full hybrid+rerank on 3 BEIR datasets; embedding cache; pytrec_eval |
| `scripts/hotpotqa_full_eval.py` | script (new) | Multi-hop only; seeded sample; cost per query; hop breakdown |

### BEIR Results (hybrid BM25+dense → RRF → cross-encoder rerank, pytrec_eval)
| Dataset | Docs | Queries | nDCG@10 | Recall@10 | MRR |
|---------|------|---------|---------|-----------|-----|
| SciFact | 5,183 | 300 | **0.7253** | 0.8529 | 0.6917 |
| NFCorpus | 3,633 | 323 | **0.3311** | 0.1609 | 0.5383 |
| ArguAna | 8,674 | 1,406 | **0.2826** | 0.6166 | 0.1793 |

### HotpotQA Cost Probe (20 questions, seed=42)
| Metric | Value |
|--------|-------|
| EM | 0.40 |
| F1 | 0.51 |
| Avg hops | 2.00 |
| Avg input tokens | 4,159.6 |
| Avg output tokens | 41.0 |
| Avg reranker calls | 43.0 |
| **Avg cost/query** | **$0.00060** |
| Budget target | $0.005 |
| Within budget | ✓ (12% of target) |

### Key Decisions
- **`rag_engine.cost` not `rag_engine.agent.cost`**: `reranker.py` importing from `agent.cost` created a circular import (`reranker → agent.__init__ → loop → reranker`); moving to top-level broke the cycle
- **`datasets` over `beir` package**: HuggingFace `datasets` (already a dep) loads all BEIR corpora directly; avoids adding `beir`'s complex dependency tree
- **pytrec_eval over custom metrics**: official TREC evaluation library matches BEIR leaderboard methodology exactly
- **Embedding cache per dataset**: 17K docs × 384-dim takes ~2 min to embed; cached to `data/beir_embeddings/<dataset>/` — re-runs are instant
- **ArguAna below BM25 baseline**: counter-argument queries pull dense similarity toward same-topic docs rather than the single target document; known ArguAna quirk, not a bug
- **Cost at $0.00060/query**: 8× under budget; 1,000-question full run would cost ~$0.60

### Concepts Covered
- pytrec_eval / TREC evaluation methodology — nDCG@10 as the BEIR leaderboard standard
- Thread-local cost accumulation — `threading.local()` for implicit per-request tracking without signature changes
- Circular import resolution — move shared utilities above the modules that cause the cycle
- BEIR dataset characteristics — SciFact (binary), NFCorpus (graded, many relevant), ArguAna (counter-argument, 1 relevant per query)
- Cost modeling for LLM pipelines — dominant cost is LLM input tokens (context window); output tokens (~41) are negligible

### Commands Run
```bash
uv add pytrec-eval-terrier
PYTHONPATH=src:. uv run python scripts/beir_eval.py
PYTHONPATH=src:. uv run --env-file .env python scripts/hotpotqa_full_eval.py --n 20
```

---

## Phase 8 — Production Serving ✅

### Files Created / Modified
| File | Change |
|------|--------|
| `src/rag_engine/api/__init__.py` | Package root |
| `src/rag_engine/api/models.py` | QueryRequest (max_hops field), Citation (dense_score, bm25_score), QueryResult |
| `src/rag_engine/api/stream.py` | Named-event SSE formatters (`event: <name>\ndata: {json}\n\n`); passage_event with hop param and score fields |
| `src/rag_engine/api/auth.py` | Bearer token middleware; `resolve_tenant()` dict lookup; structlog ContextVar |
| `src/rag_engine/api/ratelimit.py` | Lua sliding-window rate limiter |
| `src/rag_engine/api/cache.py` | Redis semantic cache; cosine sim > 0.97; 1-hr TTL |
| `src/rag_engine/api/app.py` | Full app: parallel BM25+HNSW, `_extract_bridge`, `_rerank_sync`, multi-hop SSE loop, CORS, OMP fix |
| `src/rag_engine/agent/llm.py` | Added `stream_complete()` async token iterator |
| `scripts/run_server.py` | uvicorn startup |
| `tests/test_api.py` | 13 endpoint tests (all mocked) |
| `infra/redis.yml` | Docker Compose Redis |
| `web/index.html` | Browser UI: named-event SSE listener, score bars, agent trace, wire protocol panel |

### Key decisions made during build (vs original plan)
- **Named-event SSE**: switched from `data: {"type": "token"}` (unnamed) to `event: token\ndata: {...}` (named). Browser dispatches per event type natively — no client-side switch statement.
- **`StreamingResponse` not `EventSourceResponse`**: `sse-starlette` double-wraps; plain `StreamingResponse(media_type="text/event-stream")` is correct.
- **Parallel BM25 + HNSW** via `ThreadPoolExecutor(max_workers=2)`: both release the GIL, true parallelism. 660ms → 459ms (30% reduction).
- **`_extract_bridge` reads top-3 passages** (not just top-1): the bridge hint may be in passage 2 or 3 — single-passage version always returned NONE for the Eiffel Tower question.
- **`_rerank_sync` after hop-2**: cross-encoder re-scores the merged hop-1+hop-2 set against the original query so the LLM prompt gets the best passages regardless of which hop found them.
- **P95 budget 3 000 ms** (not 800 ms): full hybrid pipeline (460ms retrieval + neural reranker + LLM streaming) cannot fit in 800ms. 800ms was HNSW-only without reranker.
- **Retrieval timeout 30 s** (not 200 ms): 200ms was too tight for the full hybrid pipeline.
- **OMP_NUM_THREADS=1** in env: prevents Apple Silicon SIGSEGV when PyTorch called from multiple retrieval threads.

### Observed Metrics
| Metric | Result |
|--------|--------|
| End-to-end P95 (no cache) | < 3 000 ms ✓ |
| Hybrid retrieval latency | 305–460 ms |
| Retrieval speedup (parallel) | 660ms → 459ms (30%) |
| Semantic cache hit rate | ≥ 20% (after warm-up) |
| Cost per query | $0.0006 |
| Multi-hop bridge extraction | Working (verified on Taj Mahal, Eiffel Tower questions) |

### Tenant token plan for production
Current: `RAG_TENANT_TOKENS` env var (JSON dict). Fine for B2B with few customers.
For public self-serve (Phase 10): Postgres `api_keys` table with hashed keys (`hashlib.sha256`).
`resolve_tenant()` interface stays the same — only the backend lookup changes.

---

## Phase 9 — Observability & SLOs ✅
Goal: instrument every pipeline stage with Prometheus metrics, provision a Grafana dashboard, and wire three SLO-enforcing alert rules.

### Files Created / Modified
| File | Change |
|------|--------|
| `src/rag_engine/api/metrics.py` | 6 instruments: QUERY_TOTAL, QUERY_ERRORS, STAGE_LATENCY, HOP_COUNT, CACHE_HIT_RATE, FAITHFULNESS_SCORE |
| `src/rag_engine/api/app.py` | Metrics wired throughout `_sse_generator`; `GET /metrics` endpoint added; structlog extended |
| `infra/prometheus.yml` | Scrape config (15 s interval) pointing at `host.docker.internal:8000`; references alert_rules.yml |
| `infra/alert_rules.yml` | 3 alert rules: LatencySLOBreach, HighErrorRate, FaithfulnessDrift |
| `infra/observability.yml` | Docker Compose: Prometheus 2.51 + Grafana 10.4.2; named volumes; `extra_hosts: host-gateway` |
| `infra/grafana/provisioning/datasources/prometheus.yml` | Auto-wires Prometheus datasource; anonymous viewer |
| `infra/grafana/provisioning/dashboards/dashboard.yml` | File provider — loads any JSON from the folder |
| `infra/grafana/provisioning/dashboards/rag_engine.json` | 8-panel dashboard: QPS, cache hit rate, error rate, hop distribution, per-stage P95 stacked, 3 SLO gauges |

### Key Decisions
- **Pull-based scraping**: Prometheus pulls `/metrics` every 15 s rather than the app pushing — decouples instrumentation lifetime from scrape frequency and avoids push failures silently losing data.
- **`FAITHFULNESS_SCORE` Gauge + drift alert**: the gauge is set by the eval script (Phase 10 CI gate); the Prometheus rule fires when current value drops 2 pts below the 7-day rolling average — catches silent model degradation that latency metrics miss.
- **Grafana file provisioning**: datasource + dashboard loaded from YAML/JSON at startup — no manual clicking; reproducible across environments.
- **`host.docker.internal` + `extra_hosts: host-gateway`**: necessary on Linux (and Hetzner) so the Prometheus container can reach the API process running on the host. macOS has this automatically; Linux needs the explicit mapping.
- **P95 SLO threshold = 3 000 ms**: full hybrid pipeline (embed + parallel BM25+HNSW + cross-encoder rerank + LLM streaming) cannot fit in the original 800 ms budget. 3 s covers the cache-miss path with 10% headroom.

### SLO Definitions
| SLI | SLO | Alert |
|-----|-----|-------|
| P95 end-to-end latency | < 3 000 ms | `LatencySLOBreach` (warning, 5 min) |
| Query error ratio | < 5% | `HighErrorRate` (warning, 5 min) |
| Faithfulness score | ≥ rolling 7-day avg − 2 pts | `FaithfulnessDrift` (critical, 10 min) |
| Error budget | 0.1% ≈ 43 min/month | Phase 10 CI gate enforces |

### Start Observability Stack
```bash
docker compose -f infra/observability.yml up -d
# Prometheus → http://localhost:9090
# Grafana    → http://localhost:3000  (admin / admin)
```

---

## Phase 10 — Containerize · IaC · CI/CD ✅
Goal: ship a reproducible production deployment — multi-stage Docker image, five-service Compose stack, Terraform for Hetzner IaC, Postgres api_keys auth with SHA-256 hashing, and a GitHub Actions workflow that builds and deploys on demand.

### Files Created / Modified
| File | Change |
|------|--------|
| `Dockerfile` | Multi-stage: uv builder → python:3.12-slim runtime; non-root `rag` user; OMP_NUM_THREADS=1 |
| `infra/docker-compose.yml` | 5 services: api, redis, postgres, prometheus, grafana; healthchecks; `start_period: 60s` on api |
| `infra/postgres/init.sql` | `api_keys` table: `key_hash TEXT PRIMARY KEY`, `tenant_id`, `active` |
| `infra/terraform/main.tf` | Hetzner CX33, Ubuntu 24.04, fsn1, Docker user_data, firewall ports 22/8000/3000 |
| `infra/terraform/variables.tf` | `hcloud_token` (sensitive), `ssh_public_key` |
| `infra/terraform/outputs.tf` | `server_ip`, `api_url`, `grafana_url`, `ssh_command` |
| `.github/workflows/deploy.yml` | `workflow_dispatch`; buildx `linux/amd64`; push to ghcr.io; SSH deploy via appleboy/ssh-action |
| `src/rag_engine/api/auth.py` | Full rewrite: SHA-256 hashing, asyncpg pool (module-level `_pool`), env-dict fallback |
| `src/rag_engine/config.py` | Added `db_url: str = ""` for Postgres DSN |
| `pyproject.toml` | Added `asyncpg>=0.29.0` |
| `scripts/build_production_index.py` | Reverted to clean state: load_vectors() returns ndarray only; no id_map |
| `scripts/build_100k_dataset.py` | Renames 1M files to `*_1m.*`; rebuilds docs.db with renumbered sequential offsets; writes new vectors.bin |
| `scripts/gen_api_keys.py` | Generates `sk-{token_hex(24)}` tokens, inserts SHA-256 hashes into Postgres |

### 100K Production Index
The 1M files are kept as `*_1m.*` for local benchmarking. Production serves the top 100K Wikipedia articles by `incoming_links`:
- `docs.db` — 2,308,125 chunk rows, `vector_offset` renumbered 0,1,2... to match HNSW sequential IDs
- `vectors.bin` — 3.55 GB
- `hnsw.index` — 4.17 GB (copied from `hnsw_100k.index`)

**The alignment trick**: FAISS assigns IDs 0,1,2... in insertion order. `build_100k_dataset.py` renumbers `vector_offset` in the same order so `first_chunk_offsets[faiss_id]` always hits the correct article — no translation map needed.

### Postgres Auth
`resolve_tenant(token)` is now `async`:
1. If `_pool` is set (Postgres available): SHA-256 hash the token, query `api_keys WHERE key_hash=$1 AND active=TRUE`
2. Fallback: `TENANT_MAP.get(token)` (env-dict, still works for B2B / dev)

The `_pool` (asyncpg) is created in the FastAPI lifespan on startup and closed on shutdown.

### Terraform Deploy
```bash
terraform -chdir=infra/terraform apply \
    -var="ssh_public_key=$(cat ~/.ssh/id_ed25519.pub)"
# Outputs: server_ip = 167.233.55.52
```

### Key Decisions
- **Hetzner CX33 over AWS**: 4 vCPU, 8 GB RAM, €7.49/mo vs ~$30/mo for comparable AWS instance. HNSW fits in RAM with 4 GB headroom for Redis + Postgres + Prometheus.
- **`workflow_dispatch` only**: no deploy on every push — prevents accidental production overwrites. Triggered manually from GitHub Actions UI.
- **`--platform linux/amd64`**: dev machine is ARM (Apple M4 Pro); server is x86. Cross-compilation via Docker buildx.
- **`start_period: 60s`** on API healthcheck: HNSW load (4.17 GB) + model warm-up takes ~1 min; grace period prevents premature failure signals.
- **SHA-256, not bcrypt**: API tokens are 48 random hex chars (192 bits). At this entropy level SHA-256 is sufficient — bcrypt cost is designed for low-entropy passwords, not random secrets.

### 100K Eval Results
| Corpus | nDCG@10 | Recall@10 | MRR | Questions |
|--------|---------|-----------|-----|-----------|
| 1M articles (benchmark) | 0.4618 | 0.4780 | 0.5994 | 1,000 |
| 100K articles (production) | 0.3479 | 0.3490 | 0.4411 | 202 |

Note: only 202 of 1,000 HotpotQA questions have both supporting articles in the top 100K corpus. Numbers are not directly comparable to the 1M benchmark.

### Server: Hetzner CX33 at `167.233.55.52`
```bash
ssh root@167.233.55.52
# cd /opt/rag-engine/infra && docker compose up -d
# API  → http://167.233.55.52:8000
# Grafana → http://167.233.55.52:3000
```

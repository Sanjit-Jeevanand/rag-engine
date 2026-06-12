# Build Steps

Granular record of every step taken while building the project.
Use this to reproduce the build from scratch or hand off to another engineer.

---

## Phase 0 — Engineering Foundations

### Environment Setup
1. Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Add to PATH: `export PATH="$HOME/.local/bin:$PATH"`
3. Deactivate conda (conflicts with uv venv): `conda deactivate`

### Project Initialisation
4. `uv init --name rag-engine --python 3.12`
5. Delete scaffold: `rm main.py`
6. Update description in `pyproject.toml`
7. Add `[build-system]` + hatchling config to `pyproject.toml`
8. Add `[dependency-groups] dev` with ruff, mypy, pytest, pip-audit, pre-commit
9. Add `[tool.ruff]`, `[tool.mypy]`, `[tool.pytest.ini_options]` config blocks to `pyproject.toml`
10. Add `pythonpath = ["src"]` to `[tool.pytest.ini_options]` (fixes editable install issue with Python 3.12 homebrew)

### Directory Structure
11. `mkdir -p src/rag_engine eval/results tests infra docs/adr`
12. `touch src/rag_engine/__init__.py`

### Dependencies
13. `uv sync` → generates `uv.lock`, installs all packages

### Source Files
14. Create `src/rag_engine/config.py` — pydantic-settings, env prefix `RAG_`, 12-factor config
    → all config comes from env vars; `RAG_LOG_LEVEL=DEBUG` overrides the default
15. Create `src/rag_engine/log.py` — structlog, JSON output, request_id via ContextVar
    → `bind_request_id()` at request entry; every log line after carries it automatically

### Eval Gate
16. Create `eval/gate.py` — fails if `eval/results/latest.json` missing or lacks sentinel key
    → `sys.exit(1)` signals failure to make and CI; real metrics replace sentinel in Phase 2
17. Create `eval/results/latest.json` with `{"sentinel": true}`

### Tests
18. Create `tests/test_smoke.py` — smoke tests for config defaults and request_id UUID format
    → smoke test = "turn it on and see if it smokes"; proves package imports without crashing

### Makefile
19. Create `Makefile` with targets: `lint`, `typecheck`, `test`, `audit`, `eval-gate`, `ci`
    → `make ci` runs identical pipeline locally and in GitHub Actions
20. Verify each target: `make lint`, `make typecheck`, `make test`, `make audit`, `make eval-gate`
21. `make ci` → full pipeline green

### pre-commit
22. Create `.pre-commit-config.yaml` — ruff, ruff-format, mypy (local), pre-commit-hooks
23. `uv run pre-commit install` → hooks wired at `.git/hooks/pre-commit`
24. Test: add deliberate formatting violation → `git commit` → pre-commit blocks it
25. `uv run ruff format .` → auto-fix formatting

### Git & CI
26. Create `.gitignore` — excludes `__pycache__/`, `.venv/`, `.env`, `.DS_Store`, `.vscode/`
27. Create `.github/workflows/ci.yml` — checkout → install uv → uv sync → lint → typecheck → test → audit → eval-gate
28. Commit and push to main
29. Verify GitHub Actions → all steps green in ~15s
30. Break it: `git commit --no-verify -m "test: deliberate type error"` → CI catches it, blocks PR

---

## Phase 1 — Corpus Ingestion and Embedding Pipeline

### Schema
31. `git checkout -b phase/1-ingestion`
32. `mkdir -p src/rag_engine/ingest && touch src/rag_engine/ingest/__init__.py`
33. Create `src/rag_engine/ingest/schema.py`:
    - `documents` table: `(article_id, chunk_index)` primary key, chunk-aware design
    - Columns: `title`, `categories`, `timestamp`, `chunk_text`, `chunk_count`, `vector_offset`, `status`, `embedded_at`, `checksum`
    - `PRAGMA journal_mode=WAL` — allows concurrent reads during writes
    - `CREATE INDEX ON documents(status)` — fast lookup of pending/failed chunks on restart
    → see `src/rag_engine/ingest/schema.py`; key concept: `IF NOT EXISTS` makes init_db idempotent
34. Create `tests/test_schema.py` — tests table creation and idempotency (`IF NOT EXISTS`)
    → see `tests/test_schema.py`
35. `uv run ruff check --fix . && uv run ruff format .` → fix import ordering
36. `make ci` → full pipeline green

### Downloader
37. `uv add httpx` → adds httpx to `pyproject.toml` + updates `uv.lock`
38. Create `src/rag_engine/ingest/downloader.py`:
    - streams download in 1 MB chunks — never loads full 4 GB into RAM
    - writes to `.tmp` first, renames on success — prevents corrupt partial files on crash
    - skips download if destination already exists — idempotent
    → see `src/rag_engine/ingest/downloader.py`
39. Create `tests/test_downloader.py`:
    - `test_skips_if_exists` — proves idempotency
    - `test_writes_to_tmp_then_renames` — patches `httpx.Client` to avoid real HTTP calls
    → `unittest.mock.patch` replaces httpx.Client with a fake; tests stay fast and offline
40. `make test` → 6 tests passed

### Parser
41. Create `src/rag_engine/ingest/parser.py`:
    - `WikiArticle` dataclass: article_id, title, text, categories, timestamp
    - `parse_snapshot(path)` → `Iterator[WikiArticle]` — generator, one article at a time
    - skips malformed lines silently — real data is dirty
    → see `src/rag_engine/ingest/parser.py`; key concept: yield keeps memory flat across 6M+ articles
42. `make typecheck` → 7 source files clean

### Parser Tests
43. `mkdir -p tests/fixtures`
44. Create `tests/fixtures/sample_snapshot.jsonl` — 4 lines, 1 malformed
    → fixture file keeps tests offline; no real snapshot needed
45. Create `tests/test_parser.py` — 4 tests: count, non-empty fields, first paper values, malformed skip
46. `make test` → 10 passed

---

## Phase 2 — Evaluation Harness

### Metrics
47. Create `eval/__init__.py` — makes eval/ importable as a package
48. Create `eval/metrics.py` — pure-Python nDCG@k, Recall@k, MRR, Exact Match, F1; no heavy deps
    → nDCG discounts hits by rank position (log2); MRR is reciprocal of first hit's rank
49. Create `tests/test_metrics.py` — unit tests for all five metrics

### Vector Index
50. `uv add faiss-cpu` → adds FAISS to dependencies
51. Create `eval/index.py` — VectorIndex class:
    - `np.fromfile(vectors_path, dtype=np.float32).reshape(-1, 384)` — load full binary file
    - `conn.execute("SELECT vector_offset, title FROM documents WHERE status='embedded'")` — map offsets to titles
    - `faiss.IndexFlatIP(384)` — exact inner-product search (cosine on L2-normalised vectors)
    - `search()` dedupes by article title — many chunks per article, score at article level
    → key: search k×5 candidates then dedup down to k, to survive chunk-heavy articles

### Gold Set
52. `uv add datasets` → HuggingFace datasets library
53. Create `scripts/seed_gold_set.py`:
    - `load_dataset("hotpot_qa", "distractor", split="validation")` — 7,405 questions
    - filter to questions where ALL supporting titles are embedded in DB
    - `html.unescape()` on titles — HotpotQA encodes special chars (é, ü, etc.)
    - sample 1,000 and save to `eval/hotpotqa_gold.json`
54. `PYTHONPATH=src uv run python scripts/seed_gold_set.py` → saved 1,000 questions

### Eval Runner
55. Create `eval/hotpotqa_eval.py` — loops gold questions, calls index.search(), scores metrics, writes latest.json
56. `PYTHONPATH=src:. uv run python eval/hotpotqa_eval.py` → first run with partial embedding: nDCG=0.14

### Comparator + CI Gate
57. Create `eval/comparator.py`:
    - auto-promotes latest to baseline if no baseline exists
    - `sys.exit(1)` if any metric drops > TOLERANCE=0.02 below baseline
58. Update `eval/gate.py` to call `compare()` after sentinel check
59. `make eval-gate` → passes, auto-seeds baseline.json
60. Verify regression detection: manually lower a metric → gate fails

### Final Baseline
61. Wait for full embedding (8.8M/8.8M chunks)
62. `PYTHONPATH=src uv run python scripts/seed_gold_set.py` — re-seed against full corpus
63. `PYTHONPATH=src:. uv run python eval/hotpotqa_eval.py` → nDCG=0.4618, Recall=0.478, MRR=0.5994
64. `make eval-gate` → auto-promotes to baseline
65. `git add eval/results/baseline.json eval/results/latest.json && git commit`

### Qualitative Check
66. Create `scripts/sample_retrieval.py` — samples 20 questions, prints hit/miss + top-10 ranked titles
67. `PYTHONPATH=src:. uv run python scripts/sample_retrieval.py` → 7/20 full hits (35%)
    → finding: ALL missed articles are in the corpus; failures are exact-name mismatches BM25 would fix

---

## BEIR Baseline (dense-only)

68. Create `scripts/beir_eval.py`:
    - loads SciFact (5,183 docs) and NFCorpus (3,633 docs) from `BeIR/*` HuggingFace datasets
    - qrels from `BeIR/scifact-qrels` and `BeIR/nfcorpus-qrels` (test split)
    - embeds corpus docs as `"title. text"` with bge-small-en-v1.5, batch 512
    - builds per-dataset `faiss.IndexFlatIP(384)`
    - graded nDCG@10 (handles NFCorpus 0/1/2 scores), Recall@10, MRR
    - saves results to `eval/results/beir_baseline.json`
69. `PYTHONPATH=src:. uv run python scripts/beir_eval.py`
    → SciFact:  nDCG@10=0.7243, Recall@10=0.8412, MRR=0.6924 (300 queries)
    → NFCorpus: nDCG@10=0.3409, Recall@10=0.1623, MRR=0.5402 (323 queries)

---

## Phase 3 — FAISS Index Comparison

70. Create `scripts/benchmark_faiss.py`:
    - `load_vectors(path)` — loads full vectors.bin (13.5 GB)
    - `sample_queries(vectors, n=1000)` — random rows as benchmark queries
    - `build_flat_l2(vectors)` — ground truth exact index
    - `compute_ground_truth(index, queries, k)` — exact top-k for each query
    - `benchmark(index, queries, ground_truth, k, label)` — P50/P99 latency + recall@10
    - `build_hnsw(vectors, ef_construction=200)` — M=32 graph index
    - `build_ivfpq(vectors, nlist=4096, m_pq=48, nbits=8)` — trains on 500K sample
    - sweeps ef=32/64/128/256 for HNSW, nprobe=8/32/64/128 for IVFPQ
71. `PYTHONPATH=src:. uv run python scripts/benchmark_faiss.py`
    → IndexFlatL2: recall=1.0000, p50=117.8ms, p99=127.6ms
    → HNSW ef=64:  recall=0.9857, p50=0.387ms, p99=0.682ms  ← chosen (300× faster)
    → IVFPQ nprobe=128: recall=0.6878 (ceiling — PQ compression too lossy)
72. Decision: IndexHNSWFlat ef=64 for serving; IndexFlatIP stays in eval gate
    → switching eval to HNSW dropped nDCG 0.46→0.37 (dedup multiplier mismatch)
73. Create `scripts/build_hnsw_index.py`:
    - builds IndexHNSWFlat (M=32, efC=200, efSearch=64) over full 8.8M vectors
    - `faiss.write_index(index, "data/hnsw.index")` — 15.9 GB on disk
    - `uv run python scripts/build_hnsw_index.py` — runs in ~21 min
74. `uv add --dev hypothesis`
75. Create `tests/test_faiss_properties.py`:
    - synthetic 100K-vector clustered corpus (500 clusters, noise=0.009/dim) — mimics real embedding structure
    - `test_hnsw_returns_exactly_k_results` — Hypothesis, 200 examples
    - `test_hnsw_average_recall_meets_threshold` — avg recall@10 ≥ 0.90 over 500 queries
    - `test_hnsw_is_deterministic` — Hypothesis, 100 examples
    - `test_hnsw_broken_ef_collapses_recall` — ef=64 outperforms ef=2 by ≥ 0.10
76. Isolated from main suite (segfault with both in memory):
    - `pyproject.toml`: `addopts = "--ignore=tests/test_faiss_properties.py"`
    - `Makefile`: added `test-faiss` target
    - `make test-faiss` → 4 passed in 7.7s

---

## Phase 4 — Throughput Engineering

77. Create `scripts/throughput_baseline.py`:
    - loads HNSW from `data/hnsw.index` (no rebuild — reuses Phase 3 artifact)
    - loads full vectors once; builds IndexFlatIP for comparison
    - `_worker(index, queries, duration, worker_id, bar)` — tight search loop for fixed duration, tqdm progress
    - `measure(index, queries, concurrency, label)` — ThreadPoolExecutor, aggregates latencies → QPS + P50/P99
    - `faiss.omp_set_num_threads(1)` — prevents OMP oversubscription at c>1
    - sweeps concurrency 1/4/8/10/12/16
78. `PYTHONPATH=src:. uv run python scripts/throughput_baseline.py`
    → HNSW c=1:  918 QPS,  p50=0.353ms, p99=13.594ms  (Python overhead dominates)
    → HNSW c=8:  18,471 QPS, p50=0.429ms, p99=0.736ms  ← chosen (all perf cores saturated)
    → HNSW c=12: 20,694 QPS, p50=0.533ms, p99=1.370ms  (efficiency cores: +12% QPS, +2× p99)
    → FlatIP c=1: 16 QPS (memory-bandwidth-bound — doesn't scale with concurrency)
79. Decision: `search_workers=8`, `faiss_omp_threads=1` — Pareto knee at c=8
80. Update `src/rag_engine/config.py` — add serving settings block:
    - `hnsw_path = "data/hnsw.index"`
    - `hnsw_ef_search = 64`
    - `search_workers = 8`
    - `faiss_omp_threads = 1`
    - all env-overridable via `RAG_*` prefix (pydantic-settings)
81. Create `scripts/profile_target.py`:
    - 5,000 single-vector search() calls in a tight loop (no timing, no tqdm)
    - loads HNSW from disk, samples 500 query vectors
    - designed to be run under py-spy, not directly
82. `sudo .venv/bin/py-spy record --output flamegraph.svg -- .venv/bin/python scripts/profile_target.py`
    → flamegraph shows 3 zones: vector load (31%), HNSW load (44%), search loop (25%)
    → hot path: main → replacement_search (SWIG wrapper) → search
    → `replacement_search` is paid per-call regardless of batch size — batching amortises it
83. Create `scripts/throughput_batched.py`:
    - sweeps batch sizes [1, 8, 32, 64, 128, 256, 512] at c=1
    - stacks batch_size query vectors into one (batch_size, 384) matrix per search() call
    - records per-query latency = total call time / batch_size for fair QPS comparison
    - then runs best batch size at c=4 and c=8
84. `PYTHONPATH=src:. uv run python scripts/throughput_batched.py`
    → batch=256 c=1: 20,822 QPS (22.7× over single-vector baseline) ← theoretical ceiling
    → batch=512 regresses (768 KB batch > L2 cache, BLAS spills to L3)
    → batch=256 c=8: 21,490 QPS — only 3% more than c=1 (BLAS SIMD already fills one core)
    → Decision: serving stays c=8 batch=1; batching only for offline/bulk workloads
       Reason: batch=256 needs ~12ms to fill at 20K QPS, blowing the 5ms P99 budget
85. Create `scripts/throughput_ivfpq.py`:
    - builds IndexIVFPQ (nlist=4096, M=48, nbits=8) if not cached; saves to data/ivfpq.index
    - sweeps nprobe [8, 32, 64, 128] × concurrency [1, 8]; uses Phase 3 recall values
    - prints three-way tradeoff table: recall vs memory vs QPS
86. `PYTHONPATH=src:. uv run python scripts/throughput_ivfpq.py`
    → IVFPQ nprobe=8  c=8: 22,740 QPS, recall=0.6374, 0.50 GB ← fastest but 35pt recall gap
    → IVFPQ nprobe=128 c=8: 2,141 QPS, recall=0.6878 ← dominated by HNSW on both axes
    → HNSW ef=64      c=8: 17,070 QPS, recall=0.9857, 15.9 GB
    → Decision: HNSW for production; IVFPQ only viable if RAM < 2 GB
87. Create `scripts/perf_check.py`:
    - `--record` mode: 2s warmup + 5s measure at c=8 ef=64, writes eval/results/perf_baseline.json
    - check mode: re-measures and fails (exit 1) if QPS < baseline×0.90 or p99 > 5ms
88. `PYTHONPATH=src:. uv run python scripts/perf_check.py --record`
    → baseline: 19,463 QPS, p99=0.706ms, floor=17,516 QPS
89. Add `make perf` target to Makefile (local only — not in `make ci`)
    → `make perf` → perf gate passed (19,277 QPS, p99=0.742ms, delta=-1.0%)

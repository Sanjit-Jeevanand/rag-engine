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

---

## Phase 5 — Hybrid Retrieval

### Retrieval module
90. `mkdir -p src/rag_engine/retrieval && touch src/rag_engine/retrieval/__init__.py`
91. `uv add bm25s` — scipy sparse inverted index; ~100ms for 1M docs vs 13.88s for rank_bm25
92. Create `src/rag_engine/retrieval/bm25.py` — `BM25Retriever(doc_ids, doc_texts)`, `retrieve(query, k) -> list[str]`
93. Create `src/rag_engine/retrieval/dense.py` — `DenseRetriever(doc_ids, corpus_vecs)`, `retrieve(query_vec, k) -> list[str]`; uses `faiss.IndexFlatIP`
94. Create `src/rag_engine/retrieval/hybrid.py` — `reciprocal_rank_fusion(ranked_lists, k)`: sums `1/(60+rank+1)` across all lists, parameter-free
95. Create `src/rag_engine/retrieval/reranker.py` — `CrossEncoderReranker(model_name)`, `rerank(query, candidates, doc_texts, k) -> list[str]`; uses `BAAI/bge-reranker-base`
96. Export all four from `src/rag_engine/retrieval/__init__.py`

### BEIR staircase
97. `uv add sentence-transformers`
98. Create `scripts/beir_hybrid_eval.py`:
    - loads SciFact + NFCorpus from HuggingFace
    - embeds corpus with bge-small-en-v1.5 (batch 512)
    - runs three pipelines per query: dense-only → hybrid RRF → hybrid + cross-encoder rerank
    - graded nDCG@10 (`_ndcg_at_k` handles NFCorpus 0/1/2 relevance)
    - saves to `eval/results/beir_staircase.json`
99. `PYTHONPATH=src:. uv run python scripts/beir_hybrid_eval.py`
    → SciFact:  dense=0.7243, hybrid=0.6691, hybrid+rerank=0.6955 (Δ−0.0288)
    → NFCorpus: dense=0.3409, hybrid=0.3233, hybrid+rerank=0.3125 (Δ−0.0284)
    → Note: dense already near-optimal on BEIR; BM25 adds noise on scientific corpora

### HotpotQA staircase
100. Create `scripts/hotpotqa_hybrid_eval.py`:
     - loads VectorIndex (8.8M vectors) + first chunk per article for BM25
     - runs three pipelines: dense-only → hybrid RRF → hybrid + cross-encoder rerank
     - metrics: nDCG@10, Recall@10, MRR on 1,000 gold questions
     - saves to `eval/results/hotpotqa_staircase.json`
101. `PYTHONPATH=src:. uv run python scripts/hotpotqa_hybrid_eval.py`
     → Dense:         nDCG=0.4618, Recall=0.478, MRR=0.5994
     → Hybrid (RRF):  nDCG=0.5398, Recall=0.611, MRR=0.6584  (+0.078 nDCG)
     → Hybrid+Rerank: nDCG=0.7035, Recall=0.700, MRR=0.8575  (+0.242 nDCG total)
     → HotpotQA benefits strongly: entity-heavy questions gain most from BM25 + reranking

---

## Phase 6 — Agentic Multi-hop RAG

### LLM client
102. `uv add openai` — GPT-4o-mini chosen: cheapest capable model, sufficient for structured extraction
103. Create `src/rag_engine/llm.py`:
     - lazy singleton `OpenAI()` client — reads `OPENAI_API_KEY` from env at first call, not import
     - `complete(messages, *, model, max_tokens, system)` — system prepended as `{"role":"system"}` message
     - `# type: ignore[arg-type]` on messages param — structurally identical to SDK's ChatCompletionMessageParam at runtime

### Reranker extension
104. Add `scores(query, candidates, doc_texts) -> np.ndarray` to `CrossEncoderReranker`:
     - same body as `rerank()` but returns raw float array instead of sorted IDs
     - used by agent for abstention: `float(np.max(scores)) < threshold` → "cannot answer"

### Agent package
105. Create `src/rag_engine/agent/` package: `__init__.py`, `llm.py`, `loop.py`
     - `llm.py`: lazy singleton `OpenAI()` client, `complete()` function (moved from `src/rag_engine/llm.py`)
     - `loop.py`: `MultiHopAgent` + `Hop` + `AgentResult` dataclasses, three system prompts
     - `__init__.py`: re-exports `AgentResult`, `Hop`, `MultiHopAgent`, `complete`
106. `MultiHopAgent.answer()` flow in `loop.py`:
     - Hop 1: retrieve top-k on original question
     - Reranker abstention: `max(scores[:3]) < threshold` → return early without LLM call
     - LLM call 1 (bridge): extract next search query, or `"ANSWER_DIRECT"` to skip hop 2
     - Hop 2: retrieve on bridge query (skipped if `ANSWER_DIRECT` or at hop cap)
     - LLM call 2 (answer): structured `ANSWER: <concise>` + `CITATIONS: [...]` format
     - LLM-based abstention: `answer_text == _CANNOT_ANSWER` → return `abstained=True`
     - LLM call 3 (reflection): `FULLY_SUPPORTED: yes/no` + `SEARCH_QUERY: <gap>`
     - Hop 3 (if unsupported): retrieve on gap query, append to pool
     - LLM call 4 (regenerate): new answer from expanded pool
     - Citation grounding: `hallucinated_ids = [c for c in cited if c not in pool]`

### Unit tests
107. Create `tests/test_agent.py`: 4 unit tests, patch target `rag_engine.agent.loop.complete`
     - `test_abstains_when_scores_below_threshold` — reranker scores [-6,-7,-8], threshold -4.0 → abstain, 0 LLM calls
     - `test_hallucinated_citation_is_flagged` — LLM cites GHOST_DOC not in retrieved set → flagged
     - `test_hop_cap_is_respected` — max_hops=1 → 1 retrieve call, no reflection
     - `test_reflection_triggers_extra_hop` — FULLY_SUPPORTED: no → 3 hops, reflection_triggered=True
     → All 4 pass

### Single-shot failure analysis
108. Create `scripts/analyze_failures.py`: single-shot RAG on 100 HotpotQA questions
     → EM=0.31, F1=0.46; 69 failures, 41 are bridge gaps (one supporting article not in top-5)
     → Confirms multi-hop hypothesis: 59% of failures need a second retrieval hop

### Agentic eval + debugging
109. Create `scripts/hotpotqa_agentic_eval.py`: side-by-side single-shot vs multi-hop on N questions
     - `--n` arg (default 100); out-of-corpus abstention test on 5 post-corpus questions
     - First run: EM 0.31 → 0.00 (catastrophic regression)
     → Root cause: `_ANSWER_SYSTEM` asked for verbose cited prose ("Paris is the capital. [A]");
       `_extract_answer` returned the full sentence; EM against gold "Paris" → always 0
110. Fix answer format: change `_ANSWER_SYSTEM` to `ANSWER: <concise>\nCITATIONS: [...]`
     - `_extract_answer` now matches `^ANSWER:\s*(.+)$` with `re.MULTILINE`; falls back to pre-CITATIONS text
     → EM 0.00 → 0.50 on 20-question smoke run; doubled single-shot baseline
111. Fix abstention: add `_CANNOT_ANSWER` sentinel to `_ANSWER_SYSTEM`
     - Model told to output `ANSWER: I cannot answer from the available evidence.` when passages insufficient
     - `loop.py` checks `answer_text == _CANNOT_ANSWER` after extraction → returns `abstained=True`
     → Out-of-corpus abstention: 0/5 → 4/5 caught

### BM25 index persistence
112. Add `save(index_dir)` + `BM25Retriever.load(index_dir)` class method using `bm25s` native API
     - Saves `bm25s` index files + `doc_ids.json` sidecar to `data/bm25_index/`
     - Both eval scripts check `BM25_INDEX_DIR.exists()` before building
     → First run: build + save (~10s); subsequent runs: load from cache (<1s)

### Retrieval metrics
113. Add IR metrics to `hotpotqa_agentic_eval.py`: Recall@5, Precision@5, nDCG@5, MRR
     - Computed per-question for single-shot and multi-hop (hop 1 separately + combined pool)
     - Detailed timestamped responses saved to `eval/responses/` (gitignored)
114. `PYTHONPATH=src:. uv run --env-file .env python scripts/hotpotqa_agentic_eval.py --n 100`
     → Single-shot:  EM=0.29, F1=0.45, Recall@5=0.68, nDCG@5=0.70, MRR=0.85
     → Multi-hop:    EM=0.49, F1=0.60, Combined Recall=0.785
     → Δ EM = +0.20 (target was +0.10) ✓
     → Abstention:   4/5 out-of-corpus caught, 8/100 in-corpus false positives
     → Hallucinations: 0/100 ✓

---

## Phase 7 — Full Benchmark Run & Cost Accounting

### Cost tracker
115. Create `src/rag_engine/cost.py`: `CostTracker` (thread-local accumulator) + `CostSnapshot` (frozen dataclass)
     - Pricing constants: gpt-4o-mini input $0.15/1M, output $0.60/1M, text-embedding-3-small $0.02/1M
     - `add_llm(input_tokens, output_tokens)`, `add_embed(tokens)`, `add_reranker(n_docs)`
     - `reset()` before each question, `snapshot()` returns `CostSnapshot` with `estimated_usd` computed property
     - Note: placed at `rag_engine.cost` (not `rag_engine.agent.cost`) to avoid circular import with `reranker.py`
116. Instrument `src/rag_engine/agent/llm.py`: `complete()` reads `response.usage.prompt_tokens` / `completion_tokens` after every API call → `cost_tracker.add_llm()`
117. Instrument `src/rag_engine/retrieval/reranker.py`: `scores()` and `rerank()` both call `cost_tracker.add_reranker(len(candidates))`

### BEIR full run
118. `uv add pytrec-eval-terrier` — official BEIR evaluation library (matches leaderboard methodology)
119. Rewrite `scripts/beir_eval.py`: full hybrid + rerank pipeline on SciFact, NFCorpus, ArguAna
     - Loads corpora via HuggingFace `datasets` (already a dep; no `beir` package needed)
     - Dense index per dataset: `bge-small-en-v1.5` embeddings cached to `data/beir_embeddings/<dataset>/`
     - BM25 + dense → RRF (top-20) → `bge-reranker-base` → top-10
     - `pytrec_eval.RelevanceEvaluator` for nDCG@10, Recall@10, MRR
     - Removed `trust_remote_code=True` (deprecated in newer `datasets` versions)
     - Output: `eval/results/beir_YYYY-MM-DD.json`
120. `PYTHONPATH=src:. uv run python scripts/beir_eval.py`
     → SciFact:  nDCG@10=0.7253, Recall@10=0.8529, MRR=0.6917 (above BM25 baseline ~0.665)
     → NFCorpus: nDCG@10=0.3311, Recall@10=0.1609, MRR=0.5383 (on-par with baseline; Recall low by design — 38 relevant docs/query)
     → ArguAna:  nDCG@10=0.2826, Recall@10=0.6166, MRR=0.1793 (below BM25 baseline; counter-argument queries hurt dense similarity)
     → Embeddings cached to `data/beir_embeddings/`; second run loads from cache

### HotpotQA full run with cost tracking
121. Create `scripts/hotpotqa_full_eval.py`: multi-hop only (no single-shot baseline), cost tracking per question
     - Reproducible sample: shuffle 1,000-question gold set with `random.Random(SEED=42)`, take first `--n`
     - `cost_tracker.reset()` before each question, `cost_tracker.snapshot()` after → logged per record
     - Per-hop-count breakdown: group EM/F1 by `len(result.hops)` (1, 2, or 3)
     - Cost summary: avg input/output tokens, avg reranker calls, avg USD/query, total USD, within-budget flag
     - Output: `eval/results/hotpotqa_full_YYYY-MM-DD.json` + `eval/responses/hotpotqa_full_<ts>.json`
122. `PYTHONPATH=src:. uv run --env-file .env python scripts/hotpotqa_full_eval.py --n 20` (cost probe)
     → EM=0.40, F1=0.51, avg hops=2.00, abstained=4/20
     → Avg cost/query: $0.00060 (12% of $0.005 target ✓), total=$0.013 for 20 questions
     → Avg input tokens: 4,159.6, avg output tokens: 41.0, avg reranker calls: 43.0

---

## Phase 8 — Production Serving

### Infra setup
123. Add Redis to `infra/redis.yml` (Docker Compose service): `redis:7-alpine`, port 6379,
     `maxmemory 256mb`, `maxmemory-policy allkeys-lru`
     - `docker compose -f infra/redis.yml up -d` for local dev
124. `uv add fastapi uvicorn[standard] sse-starlette redis structlog`
     - `sse-starlette` for `EventSourceResponse` (SSE support in FastAPI)
     - `redis` (async client via `redis.asyncio`)

### Pydantic models
125. Create `src/rag_engine/api/models.py`:
     - `QueryRequest(BaseModel)`: `query: str`, `max_hops: int = 2`, `top_k: int = 5`
     - `Citation(BaseModel)`: `passage_id: str`, `title: str`, `text: str`, `score: float`
     - `QueryResult(BaseModel)`: `query_id: str`, `answer: str`, `citations: list[Citation]`,
       `cache_hit: bool`, `partial: bool`, `generation_unavailable: bool`,
       `tenant_id: str`, `cost_usd: float | None`

### Auth middleware
126. Create `src/rag_engine/api/auth.py`:
     - `TENANT_MAP: dict[str, str]` — maps bearer token → tenant_id (loaded from env `RAG_TENANT_TOKENS`)
     - `resolve_tenant(token: str) -> str | None` — returns None on unknown token
     - `AuthMiddleware(BaseHTTPMiddleware)`: extract `Authorization: Bearer <token>` header;
       call `resolve_tenant`; on None return 401; bind `tenant_id` via
       `structlog.contextvars.bind_contextvars(tenant_id=tid)`

### Redis semantic cache
127. Create `src/rag_engine/api/cache.py`:
     - `SemanticCache` class; constructor takes `redis.asyncio.Redis` client and `embedder`
     - `async get(query: str) -> QueryResult | None`:
         1. embed query → 384-dim vector
         2. scan Redis Hash `cache:embeddings` for cosine sim > 0.97
         3. on hit: fetch `cache:result:<cache_key>` (JSON), deserialize, return QueryResult
         4. on miss: return None
     - `async set(query: str, result: QueryResult) -> None`:
         1. embed query → store in `cache:embeddings` Hash
         2. serialize result → `SET cache:result:<cache_key> <json> EX 3600` (1-hr TTL)
     - Log `cache_hit=True/False` via structlog on every call
     - Expose `hit_rate()` property — computed from a Redis counter (`cache:hits`, `cache:total`)

### Rate limiter
128. Create `src/rag_engine/api/ratelimit.py`:
     - Lua script stored as module-level constant:
       ```lua
       local key = KEYS[1]
       local cap = tonumber(ARGV[1])
       local now = tonumber(ARGV[2])
       local window = tonumber(ARGV[3])
       redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
       local count = redis.call('ZCARD', key)
       if count < cap then
         redis.call('ZADD', key, now, now .. math.random())
         redis.call('EXPIRE', key, window)
         return 1
       end
       return 0
       ```
       (sliding-window variant; alternatively use token-bucket with INCRBY + TTL)
     - `async check(redis_client, tenant_id, cap=100, window_sec=60) -> bool`
     - On False the endpoint returns 429 with `Retry-After: <window_sec>` header

### SSE stream helpers
129. Create `src/rag_engine/api/stream.py`:
     - `token_event(text: str) -> str` → `data: {"type":"token","text":"<text>"}\n\n`
     - `done_event(query_id: str) -> str` → `data: {"type":"done","query_id":"..."}\n\n`
     - `partial_event(passages: list[Citation]) -> str` → `data: {"type":"partial",...}\n\n`
     - `error_event(msg: str) -> str` → `data: {"type":"error","message":"..."}\n\n`
     - `gen_unavailable_event(passages: list[Citation]) -> str` →
       `data: {"type":"generation_unavailable","passages":[...]}\n\n`

### FastAPI app — POST /query
130. Create `src/rag_engine/api/app.py`:
     - `lifespan` context manager: create Redis client, load HNSW index + embedder + reranker
     - Mount `AuthMiddleware`
     - `POST /query` (returns `EventSourceResponse`):
         1. Rate limit check → 429 if over limit
         2. Semantic cache check → stream cached answer if hit; return
         3. `asyncio.wait_for(retrieve_and_rerank(request.query), timeout=0.2)`
            - `TimeoutError` → yield `partial_event(passages=[])` and return
         4. Open `EventSourceResponse` generator
         5. `asyncio.wait_for(llm_stream(passages, query), timeout=5.0)`:
            - Stream tokens via `yield token_event(chunk)` as they arrive
            - `TimeoutError` → yield `gen_unavailable_event(passages)` and return
         6. Assemble `QueryResult`; write to Redis `result:<query_id>`; update cache
         7. Yield `done_event(query_id)`
     - `GET /query/{id}` → fetch `result:<id>` from Redis → return `QueryResult` JSON (404 if missing)
     - `GET /health` → `{"status": "ok"}`
     - `GET /ready` → check Redis + index loaded → `{"status": "ready"}` or 503
     - `GET /metrics/cache` → `{"hit_rate": ..., "total": ..., "hits": ...}`

### LLM streaming integration
131. Update `src/rag_engine/agent/llm.py`: add `stream_complete(messages) -> AsyncIterator[str]`
     - Uses `client.chat.completions.create(..., stream=True)` → yields `chunk.choices[0].delta.content`

### Server startup
132. Create `scripts/run_server.py`:
     - `uvicorn src.rag_engine.api.app:app --host 0.0.0.0 --port 8000 --reload`
     - Reads `RAG_PORT`, `RAG_WORKERS` from env

### Tests
133. Create `tests/test_api.py`:
     - `pytest-asyncio` + `httpx.AsyncClient` + `AsyncMock` for Redis and retrieval
     - Test: auth missing → 401
     - Test: auth invalid → 401
     - Test: rate limit exceeded → 429 + Retry-After header
     - Test: semantic cache hit → SSE stream contains cached tokens, no retrieval call
     - Test: semantic cache miss → retrieval called, tokens streamed, result stored
     - Test: retrieval timeout (mock `asyncio.wait_for` to raise `TimeoutError`) → partial event
     - Test: LLM timeout → generation_unavailable event
     - Test: `GET /query/{id}` after `POST /query` → full result with citations
     - Test: `GET /ready` before index load → 503; after → 200
134. `uv run pytest tests/test_api.py -v`

### Smoke test
135. `docker compose -f infra/redis.yml up -d`
136. `PYTHONPATH=src:. uv run python scripts/run_server.py`
137. Send a query via `curl`:
     ```bash
     curl -N -H "Authorization: Bearer <token>" \
       -H "Content-Type: application/json" \
       -d '{"query": "Who was the director of Inception?"}' \
       http://localhost:8000/query
     ```
138. Send the same query again — confirm cache hit in logs and SSE response is immediate
139. Flood one tenant → confirm 429s; confirm second tenant is unaffected
140. `GET /metrics/cache` → confirm hit_rate ≥ 0.2 after repeated-query workload

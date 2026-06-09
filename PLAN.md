# Build Plan -- 11 Phases (Production-Grade --> Exceptional)

Each phase: **teach --> build to standard --> break it (as a test or eval regression) --> understand
--> ADR + measured bullet.** Every phase must clear the **production bar** in `BRIEF.md` (tests,
types, eval gate, security, observability, reliability, operability, delivery). Phases marked with
a star are the **exceptional differentiators** -- the ones that separate this from every other RAG
repo. Never skip "break it," and always turn the failure into a regression or eval test.

## The three headline outcomes we are building toward (hard targets)
- **Outcome 1 -- ANN retrieval benchmarked, profiled, and optimized:** 6M+ vectors, recall@10 >= 97%,
  P99 < 5 ms, >= 2,000 QPS, 3x QPS improvement with flamegraph story. --> *Phase 3 (FAISS index
  comparison) + Phase 4 (profile + optimize).*
- **Outcome 2 -- Agentic multi-hop RAG on public benchmarks:** +15 EM on HotpotQA vs single-shot,
  nDCG@10 +0.20 on BEIR with hybrid + rerank. --> *Phase 5 (retrieval pipeline) + Phase 6
  (agentic loop) + Phase 7 (benchmarks).*
- **Outcome 3 -- Eval-gated, production-served:** 93% faithfulness, P95 < 800 ms, ~$0.004/query,
  CI eval gate, Grafana dashboard. --> *Phase 2 (harness), Phase 8 (serving), Phase 9 (obs + SLOs),
  Phase 10 (IaC + CI/CD).*

## The exceptional differentiators (built, not claimed)
- **Star 1 -- optimization story with before/after numbers:** Phase 3 (FAISS index type comparison,
  recall-latency Pareto curve) + Phase 4 (profile --> flamegraph --> batching fix --> 3x multiplier +
  CI perf budget). *(Stretch after Phase 10: replace FAISS with hand-written HNSW for the hard-CS
  differentiator.)*
- **Star 2 -- prove quality on a public benchmark:** Phase 7 (HotpotQA EM + BEIR nDCG@10, reproducible).
- **Star 3 -- one genuinely hard RAG technique:** Phase 6 (agentic multi-hop, self-reflection, citation grounding).
- **Star 4 -- artifact that looks production in 30 seconds:** Phase 10 (ADRs, diagram, failure log, benchmark report, cost analysis).

---

## Phase 0 -- Engineering foundations and quality gates
*Goal: a repo where it is impossible to merge broken, untyped, unformatted, or insecure code --
and where a failing eval score blocks a merge just like a failing test.*

1. `uv`-managed project, pinned lockfile, Python 3.12, clear layout (`src/`, `eval/`, `tests/`,
   `infra/`, `docs/adr/`).
2. Wire `ruff` (lint + format), `mypy --strict`, `pytest`, `pip-audit` behind a `Makefile`.
3. `pre-commit` hooks run the same checks locally; GitHub Actions runs them on every push/PR.
4. `pydantic-settings` config (12-factor, env-driven); structured JSON logging with `request_id`
   from line one.
5. **Eval gate placeholder:** a dummy eval script that fails if a sentinel metric file is missing.
   The real gate fills in later; the scaffolding enforces the habit from day one.
6. **Break it:** open a PR with a type error and an unformatted file -- watch CI go red and block
   the merge.

- **Learn:** quality gates, shift-left testing, 12-factor config, tooling-as-code, why eval gates
  belong in CI (not just notebooks).
- **Bullet:** "Set up a quality-gated Python pipeline (ruff / mypy-strict / pytest / pip-audit +
  eval gate) enforced via pre-commit + GitHub Actions, blocking non-conforming and regressing merges."

---

## Phase 1 -- Corpus ingestion and embedding pipeline
*Goal: turn raw Wikipedia text into vectors you can query -- and understand every step.*

1. Download and parse the Wikipedia dump (Wikimedia CirrusSearch JSONL); target 6M+ articles.
   Store raw text and metadata in SQLite (title, text, article_id, categories, timestamp).
2. Batch embedding with `sentence-transformers` (`bge-large-en-v1.5`, 1024-dim); async worker with
   a bounded queue; write vectors to a flat binary file (numpy mmap, `float32`).
3. Track ingestion progress in the DB (status, embedding timestamp, checksum); resumable on crash
   (idempotent per-document).
4. Unit tests for parsing and normalization; integration test that ingests 1,000 docs end-to-end;
   assert vector file shape and checksum.
5. **Break it:** corrupt a document mid-ingest, restart -- confirm it re-embeds only the failed
   doc, not the whole corpus.

- **Learn:** corpus ingestion patterns, batched async I/O, mmap for large vector files,
  idempotent pipelines, why sentence-transformers and what `bge-large` does differently from OpenAI
  embeddings.
- **Bullet:** "Ingested and embedded a 6M+ article Wikipedia corpus via an idempotent async pipeline,
  storing 1024-dim float32 vectors in a memory-mapped flat file."

---

## Phase 2 -- Evaluation harness (the gate that makes everything else honest)
*Goal: wire the eval framework before building any retrieval -- so every retrieval choice is
measured, not vibes. This phase is what makes the project credible.*

1. **Retrieval eval (BEIR-style):** implement nDCG@10, Recall@10, MRR against the BEIR benchmark
   (pick 2-3 datasets: SciFact, NFCorpus, ArguAna). Use the standard BEIR evaluation scripts; do
   not roll your own metric implementations.
2. **Generation eval (RAGAS-style faithfulness):** build a gold set of 200-500 (question, context,
   reference-answer) triples from HotpotQA dev. An LLM judge (claude-haiku-4-5) scores faithfulness
   (is the answer grounded in the retrieved context?) and answer correctness (does it match the
   reference?). Log scores per question; compute mean and P10 (worst-tail).
3. **Exact-match baseline:** run HotpotQA dev through a single-shot RAG pipeline (brute-force
   retrieval for now) and record the baseline EM score. This is the number we will beat in Phase 6.
4. Wire the eval scripts as a CI step: serialize scores to `eval/results/latest.json`; a comparator
   script fails the build if any metric regresses by more than a threshold (e.g., nDCG@10 drops 0.02
   or faithfulness drops 2 points).
5. **Break it:** introduce a retrieval bug that returns random docs -- watch nDCG@10 collapse and
   CI block the merge.

- **Learn:** BEIR as the standard IR benchmark, nDCG vs Recall vs MRR, RAGAS faithfulness vs
  answer correctness, why eval-in-CI is the engineering move that separates research from production,
  exact-match as a reproducible generation metric.
- **Bullet:** "Built a BEIR retrieval eval + RAGAS-style faithfulness harness wired into CI as a
  regression gate, blocking merges that drop nDCG@10 or faithfulness below threshold."

---

## Phase 3 -- FAISS index setup, type comparison, and first benchmarks
*Goal: get a working, tested ANN index over the full corpus -- understand the index-type trade-off
before optimizing.*

1. Understand the FAISS index hierarchy: `IndexFlatL2` (exact, ground truth), `IndexHNSWFlat`
   (graph-based ANN), `IndexIVFFlat` / `IndexIVFPQ` (inverted-file + optional quantization).
2. Build `IndexFlatL2` as the exact-search ground truth (brute-force baseline).
3. Build `IndexHNSWFlat` and `IndexIVFFlat`; benchmark recall@10 vs exact on a held-out 1,000-query
   set at varying `ef` (HNSW) and `nprobe` (IVF). Plot the **recall-latency Pareto curve** for each.
4. Persist index to disk (`faiss.write_index` / `faiss.read_index`); load without rebuilding.
5. Property-based tests (Hypothesis): ANN index returns exactly k results; recall@10 >= 95% vs
   brute-force at the chosen operating point; results are deterministic.
6. **Break it:** set `ef=2` (HNSW) -- watch recall collapse below 50%. Set `nprobe=1` (IVF) --
   same. Capture as parameterized regression tests pinning minimum parameters.

- **Target (Outcome 1):** a working, tested FAISS index over 6M vectors. Profiling and optimization
  happen in Phase 4.
- **Learn:** FAISS index hierarchy, IndexFlatL2 vs IndexHNSWFlat vs IndexIVFPQ, recall-latency
  Pareto curve, ef and nprobe as the recall-speed knobs, product quantization trade-offs.
- **Bullet:** "Benchmarked FAISS IndexFlatL2, IndexHNSWFlat, and IndexIVFPQ over 6M+ Wikipedia vectors;
  plotted the recall-latency Pareto curve; property-tested recall@10 >= 95% vs brute-force."

---

## Star Phase 4 -- Profile, tune, and tell the optimization story
*Goal: real, defensible numbers -- and a before/after story where the signal is finding and killing
the bottleneck. This is what makes the resume bullet provable.*

1. **Baseline:** batch queries at concurrency 1, 4, 8; record QPS and P99 before any changes.
   This is the "before" number -- document it immediately.
2. **Profile with py-spy:** run the query loop under load; capture a **flamegraph**. Identify the
   bottleneck -- almost always un-batched single-vector queries leaving FAISS BLAS throughput unused.
3. **Fix from evidence:** batch queries into `index.search(batch_matrix, k)` -- FAISS is built for
   batched BLAS calls; tune `ef` or `nprobe` at the Pareto knee from Phase 3; tune thread count
   (`faiss.omp_set_num_threads`). Record the **before/after QPS multiplier** (target: >= 3x).
4. **Compare IndexIVFPQ:** run the same benchmark on the quantized index; document the recall vs
   memory vs latency three-way trade-off.
5. **Throughput target:** >= 2,000 QPS at recall@10 >= 97%, P99 < 5 ms. Add a **perf-regression CI
   check** (throughput must not drop > 10% from the set baseline).
6. **Break it:** remove batching -- watch QPS collapse to the Phase 3 baseline. That is the
   concrete value of the fix.

- **Target (Outcome 1 / Star 1):** recall@10 >= 97% at P99 < 5 ms, >= 2,000 QPS -- with a
  flamegraph and a documented >= 3x before/after multiplier.
- **Learn:** FAISS batched vs single-vector search, BLAS throughput, py-spy flamegraphs, profiling
  ML systems, perf regression budgets, honest benchmarking methodology.
- **Bullet:** "Profiled FAISS under load, found un-batched single-vector queries wasting BLAS
  throughput via flamegraph, batched the search path for 3x QPS -- reaching recall@10 >= 97% at
  P99 < 5 ms, 2,000+ QPS; added CI perf regression budget."

---

## Phase 5 -- Hybrid retrieval pipeline (BM25 + dense + cross-encoder reranking)
*Goal: production-quality retrieval -- the layer between the raw index and the agentic loop.*

1. **BM25:** implement BM25 (or use `rank-bm25`) over the same corpus; build an inverted index with
   TF-IDF weights stored in a compact format. Understand why sparse and dense are complementary.
2. **Hybrid fusion:** combine BM25 and FAISS dense scores with **Reciprocal Rank Fusion (RRF)** --
   a parameter-free, rank-based combiner that is robust to score-scale differences. Retrieve top-100
   from each, fuse, take top-20 for reranking.
3. **Cross-encoder reranking:** score the top-20 candidates with a cross-encoder (BGE reranker or
   Cohere Rerank API); re-sort and take top-5 for generation. Understand why cross-encoders are
   accurate but expensive -- this is why you rerank a small set, not the whole corpus.
4. **Measure:** run the BEIR eval harness from Phase 2 after each addition (dense only --> hybrid
   --> hybrid + rerank). Record nDCG@10 at each step; the staircase of improvements is your story.
   Target: hybrid + rerank beats dense-only by >= +0.10 nDCG@10 on >= 2 BEIR datasets.
5. **Break it:** remove the reranker -- watch faithfulness and nDCG@10 drop. Capture as an
   eval regression test with the reranker toggled off.

- **Learn:** BM25 and sparse retrieval, why dense-only misses keyword-heavy queries, RRF vs
  score normalization, bi-encoder vs cross-encoder (quality vs cost), the retrieve-then-rerank
  pattern used by Cohere, Jina, and every production search system.
- **Bullet:** "Built a hybrid BM25 + dense retrieval pipeline with RRF fusion and cross-encoder
  reranking, measured on BEIR; nDCG@10 +0.20 over dense-only on SciFact and NFCorpus."

---

## Star Phase 6 -- Agentic multi-hop RAG (iterative retrieve-reason-retrieve)
*Goal: the hard ML technique. Build the loop yourself -- no LangChain. Prove it improves EM on
HotpotQA. This is the sentence that makes ML interviewers lean in.*

1. **Understand the baseline failure.** Run the single-shot RAG pipeline from Phase 2 on 100
   HotpotQA multi-hop questions. Manually inspect 10 failures: most will require bridging two facts
   that don't co-occur in any single document. This is the concrete problem you are solving.
2. **Iterative retrieve-reason-retrieve:** implement the agentic loop. On the first hop, retrieve
   top-5 docs; call the LLM to extract the intermediate entity or fact needed for the second hop;
   use that entity as the query for hop 2; retrieve top-5 again; now generate the final answer over
   the union of retrieved docs. Hand-write the loop -- do not use LangChain agents.
3. **Self-reflection:** after generating an answer, call the LLM a second time with the prompt
   "Is this answer fully supported by the retrieved passages? If not, what is missing?" If it
   identifies a gap, do one more retrieval hop targeting the gap, then regenerate. Cap at 3 hops
   total to bound cost.
4. **Citation grounding:** require the LLM to cite specific passage IDs for every claim in the
   answer. Verify that each cited passage ID was actually in the retrieved set; if a citation is
   fabricated (a passage ID not in the set), that is a hallucination -- log it and flag the answer
   for the faithfulness judge.
5. **Abstention:** compute a retrieval confidence score (e.g., max cross-encoder score of top
   result). Below a threshold (tune on a dev set), return "I cannot answer this question from the
   available evidence" rather than hallucinating. Measure abstention rate and precision on an
   out-of-corpus query set.
6. **Measure:** run the full agentic pipeline on the HotpotQA dev set (or a 500-question stratified
   sample); compute EM vs the single-shot baseline from Phase 2. Target: >= +10 EM points.
7. **Break it:** remove the second hop -- watch EM drop back to baseline. Capture as an eval
   regression. Then remove citation verification -- watch faithfulness drop. Capture that too.

- **Target (Outcome 2 / Star 4):** >= +10 EM on HotpotQA over single-shot RAG, with citation
  grounding and abstention on low-evidence queries.
- **Learn:** multi-hop reasoning, chain-of-thought retrieval, ReAct (reason + act), Self-RAG,
  FLARE (forward-looking active retrieval), why single-shot RAG fails on bridging questions,
  citation grounding vs faithfulness, abstention as a production safety feature.
- **Bullet:** "Built an agentic multi-hop RAG pipeline (iterative retrieve-reason-retrieve,
  self-reflection, citation grounding, abstention) without LangChain, improving exact-match +15
  points on HotpotQA over single-shot RAG."

---

## Phase 7 -- Full benchmark run and the public record
*Goal: produce the credible, checkable numbers that make bullets 1 and 2 defensible. This is the
phase that transforms "it worked on my questions" into a research-grade claim.*

1. **BEIR full run:** run the complete hybrid + rerank retrieval pipeline on 3 BEIR datasets
   (SciFact, NFCorpus, ArguAna or equivalent). Use the official BEIR evaluation scripts. Log
   nDCG@10, Recall@10, MRR per dataset. Record the run in `eval/results/beir_YYYY-MM-DD.json`.
2. **HotpotQA full run:** run the agentic pipeline on the full HotpotQA dev set (7,405 questions
   or a reproducible 1,000-question sample, stratified by bridge vs comparison type). Record EM,
   F1, and per-hop-count breakdown. Log the seed for reproducibility.
3. **Cost accounting:** log LLM tokens per question (input + output), embedding API calls, and
   reranker calls. Compute average cost per query. Target: <= $0.005 per end-to-end query. If over
   budget, tune (cache embeddings, use haiku for reflection calls, batch rerank).
4. **Reproducibility:** document the exact model checkpoints, dataset splits, and random seeds so
   someone else can reproduce your numbers. Add a `scripts/reproduce_eval.sh` one-liner.
5. **Break it:** re-run the BEIR eval with the reranker removed -- confirm the numbers drop and the
   CI gate catches it.

- **Target (Outcome 2):** nDCG@10 +0.20 on BEIR, EM +15 on HotpotQA -- documented, reproducible,
  with a cost per query.
- **Learn:** benchmark methodology, why reproducibility is a first-class concern, cost modeling
  for LLM-powered systems, how to report results honestly (confidence intervals, dataset caveats).
- **Bullet:** "Ran reproducible BEIR and HotpotQA benchmarks, documenting nDCG@10 +0.20 and EM
  +15 over the single-shot baseline, with cost accounting at ~$0.004/query."

---

## Phase 8 -- Production serving (streaming, auth, caching, rate limiting)
*Goal: serve the pipeline as a real API with the latency, cost, and safety properties the SLOs
demand.*

1. **Streaming FastAPI endpoint:** `POST /query` returns a streaming response (Server-Sent Events
   or chunked); first token arrives as soon as the LLM starts generating; the retrieval and rerank
   stages complete before the stream opens. `GET /query/{id}` returns the full result with citations.
2. **Auth:** API key on every endpoint (bearer token); resolve a tenant ID; log tenant ID with every
   request for cost attribution.
3. **Semantic caching:** before retrieval, embed the query and check a Redis cache keyed by
   nearest-cached-query (cosine sim > 0.97 threshold). On a hit, return the cached answer without
   hitting the LLM. Log cache hit rate. Target: >= 20% hit rate on repeated or near-duplicate
   queries, saving ~80% of the cost for those queries.
4. **Rate limiting:** per-tenant token bucket in Redis (same Lua atomicity pattern as the job queue
   project); reject over-limit requests with 429 + Retry-After.
5. **Timeouts and fallback:** if the retrieval stage exceeds 200 ms, return the best partial result;
   if the LLM call exceeds 5 s, return the retrieved passages with a "generation unavailable" flag
   rather than timing out with a 500.
6. **Break it:** flood from one tenant -- 429s; show a second tenant unaffected. Then remove the
   semantic cache and watch cost double on a repeated-query workload.

- **Target (Outcome 3):** P95 < 800 ms end-to-end, ~$0.004/query, auth, semantic caching, rate
  limiting -- all measured.
- **Learn:** streaming APIs, semantic caching (why cosine similarity cache beats exact key cache
  for NL queries), cost attribution in multi-tenant systems, partial degradation vs hard failure.
- **Bullet:** "Served the RAG pipeline as a streaming API with semantic caching (20%+ hit rate),
  per-tenant rate limiting, and graceful degradation -- P95 < 800 ms, ~$0.004/query."

---

## Phase 9 -- Observability and SLOs
*Goal: see inside the system; turn latency budget breakdowns and eval drift into operational signals.*

1. **Per-stage Prometheus metrics:** embed latency, HNSW search latency, rerank latency, LLM
   latency, cache hit rate, queries/sec, error rate (RED method). Expose `/metrics`.
2. **Structured JSON logs:** every request logs `request_id`, tenant, stage timings, hop count,
   retrieval scores, token counts, and cache hit/miss.
3. **Grafana dashboard:** latency breakdown per stage (stacked bar); throughput; cache hit rate;
   error rate; eval score trend (faithfulness, nDCG) over recent CI runs (pulled from the results
   JSON). Wire to the 99.9% availability SLO.
4. **Eval drift alerting:** if faithfulness on the rolling gold set drops > 2 points from the
   baseline, fire an alert (Prometheus alertmanager or a simple threshold check in CI).
5. **Break it:** stall the LLM call -- watch end-to-end P95 spike on the dashboard while retrieval
   latency stays flat. This makes the latency budget decomposition concrete.

- **Target (Outcome 3):** RED metrics + per-stage latency breakdown + eval drift alert in Grafana
  against a 99.9% availability target.
- **Learn:** RED method, SLI/SLO/error budgets, per-stage latency budgets (the ML-specific
  extension of RED), eval drift as an operational signal (not just a notebook concern).
- **Bullet:** "Instrumented per-stage RED metrics and eval drift alerting in Grafana against a
  99.9% SLO; structured logs carry request_id and stage timings end to end."

---

## Phase 10 -- Containerize, resilience, IaC, CI/CD
*Goal: one command brings up the whole hardened system; one push ships it safely.*

1. **Multi-stage Dockerfiles:** API server (retrieval + serving) and ingestion worker; non-root
   user, pinned base image, minimal layers, healthchecks.
2. `docker-compose up` brings up API + Redis + Postgres (metadata) + a pre-built HNSW index
   mounted from a volume. `--scale` for multiple API replicas.
3. **Graceful shutdown:** on SIGTERM, finish in-flight requests (drain), persist any partial index
   state, exit cleanly within the grace period.
4. **Terraform:** ECS Fargate (API), S3 (index artifact + vector file), ElastiCache (Redis for
   cache and rate limiting), RDS (metadata), ECR (images), least-privilege IAM. One `terraform
   apply` stands up the full stack.
5. **GitHub Actions pipeline:** `test --> lint --> type-check --> eval-gate --> pip-audit --> build
   --> trivy image scan --> deploy` with automated rollback if the post-deploy `/ready` check fails.
6. **Break it:** push a commit that drops nDCG@10 by 0.05 -- watch the eval gate block the deploy.
   Then push a commit with a type error -- watch mypy block it. Both gates visible in CI output.

- **Target (Outcome 3):** Terraform-deployed AWS stack, gated CI/CD with eval + quality gates,
  automated rollback.
- **Learn:** multi-stage builds, least-privilege IAM, IaC vs click-ops, eval gates in CI (the
  ML-specific extension of test gates), blue-green basics.
- **Bullet:** "Shipped on AWS (ECS Fargate, S3, ElastiCache) with Terraform IaC and a
  quality-gated GitHub Actions pipeline (mypy-strict, Trivy/pip-audit, eval gate) with automated
  rollback."

---

## Star Phase 11 -- The production artifact (makes depth visible in 30 seconds)
*Goal: a README a reviewer can skim in 30 seconds and understand the system's depth and ambition.*

1. **Architecture diagram:** a clean diagram (Excalidraw or similar) showing data flow from corpus
   to query, every component labeled, latency budget annotated per stage.
2. **ADRs:** one ADR per significant decision: why HNSW over IVF-PQ; why RRF over score
   normalization; why self-reflection loop is capped at 3 hops; why flat binary over a managed
   vector DB; why exact-match on HotpotQA over a self-made test set. Each ADR: context, decision,
   alternatives considered, trade-offs.
3. **"Failures I induced and how the system survived" section:** document each deliberate failure
   from previous phases -- corrupted index entry, M=2 connectivity starvation, reranker removed,
   LLM timeout, citation hallucination -- with the outcome and the test/eval that now pins it.
4. **Benchmark report:** a structured results table (BEIR nDCG@10 per dataset, HotpotQA EM
   baseline vs agentic, faithfulness score, cost per query) plus the flamegraph and the
   before/after QPS multiplier from Phase 4. Rendered in the README as a table.
5. **Cost analysis:** a breakdown of cost per query by component (embedding, retrieval, rerank,
   LLM) at different cache hit rates. Show the ~$0.004 target is achievable and how caching gets
   you there.
6. **Runbooks:** one page each for: index rebuild after corpus update; recovering from a Redis cache
   eviction spike; debugging a faithfulness regression; re-running the BEIR benchmark.

- **Target (Star 5):** a README whose depth is visible at a skim -- architecture diagram, ADRs,
  failure log, benchmark table, cost analysis, runbooks.
- **Bullet:** "Documented the system with ADRs, an architecture diagram, a failure-injection log,
  BEIR and HotpotQA benchmark results, a cost analysis, and runbooks -- artifact signals depth
  in 30 seconds."

---

## Suggested pace
~5-7 weeks at 1-2 hrs/day. **Phases 0-4 are the core differentiator** (the HNSW index you built
and benchmarked -- this is the part that makes a reviewer stop). Phases 5-7 prove the pipeline
quality on public benchmarks. Phases 8-9 add the production rigor. Phases 10-11 ship it. If short
on time, complete Phases 0-7 (the hard technical core) + Phase 9 (numbers visible) -- that already
beats almost every RAG clone.

## The payoff
You will not just have built a RAG system -- you will have built and **benchmarked** a hard
retrieval primitive (HNSW from scratch, within 8% of FAISS), proven a genuine quality improvement
on a public benchmark (+15 EM on HotpotQA with an agentic loop you wrote yourself), measured the
cost and latency of every stage, wired eval into CI so regressions block deploys, and packaged it
so the depth is visible in a 30-second skim. That is the difference between "I built a chatbot over
my PDFs" and "I understand retrieval systems and can own one" -- and between passing and acing the
MLE interview.

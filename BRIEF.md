# Project Brief: Exceptional RAG System (Production-Grade, Learn-As-You-Build)

## What we're building
A **production-grade, benchmark-proven Retrieval-Augmented Generation system** -- not a demo that
calls a managed vector DB, but a system where you own the retrieval engine, prove quality on public
benchmarks, and ship it as a measured, observable, eval-gated API. Three differentiators separate
this from every other RAG repo: (1) you implement the HNSW approximate-nearest-neighbor index
yourself, (2) you build agentic multi-hop reasoning and measure it on HotpotQA and BEIR, and (3)
you wire a faithfulness-and-retrieval eval harness into CI so regressions block merges.

Target final architecture:
```
Client --> FastAPI (streaming, authenticated) --> Agentic RAG pipeline
                                                        |
                    +-----------------------------------+-----------------------------------+
                    |                                   |                                   |
           Hand-built HNSW index            Cross-encoder reranker            Self-reflection loop
           (graph construction,             (BGE / Cohere rerank)             (iterative retrieve-
            greedy beam search,                                                 reason-retrieve,
            neighbor pruning)                                                   answer verification
                    |                                   |                        + grounded citations)
           Flat binary store                  Hybrid BM25 + dense                       |
           (6M+ vectors, mmap)               (RRF fusion)                       LLM call (Claude API)
                    |                                                                    |
         Embedding service                                              Abstention on low-evidence
         (batched, async, cached)                                       queries
                    |
      Evaluation harness (BEIR nDCG@10, HotpotQA EM, RAGAS faithfulness)
                    |
      CI gate: regressions on gold eval set block merges
                    |
      Prometheus metrics + structured logs + latency budget per stage
                    |
      Docker Compose locally --> AWS (ECS Fargate / S3) via Terraform + GitHub Actions
```

## How I want you to work with me (the learning loop -- IMPORTANT)
This is a **learning project** *built to production and research standards*. The goal is that *I*
understand information retrieval, ANN indexing, and agentic reasoning **and** what "production-ready
ML systems" actually means by the end.

For every phase and every meaningful step:
1. **Teach first.** Before writing code, explain the concept in 4-8 sentences: the problem it
   solves, the trade-offs, and how real systems (Google, Vespa, Weaviate, Cohere, RAG literature)
   handle it. Name the IR or systems-design term explicitly.
2. **Build incrementally, to standard.** Write the smallest slice that works *and meets the
   production bar below* -- tests, types, evals included, not bolted on later. One concept --> one
   change --> verify --> next. Don't generate three files at once.
3. **Make me reason.** Before each non-trivial decision, ask me what I'd do and why. If I'm wrong,
   correct me with the reasoning. Pose a "what breaks if...?" question.
4. **Force the failure -- as an automated test.** After the happy path, deliberately break it
   (corrupt an index entry, inject a hallucinated citation, flood past the rate limit) so I *see*
   the failure mode -- then capture it as a regression or eval test so it can never silently return.
5. **Connect to interviews.** End each phase with: the resume bullet it earns (with a real measured
   metric), the interview question it lets me answer, and a one-paragraph **Architecture Decision
   Record (ADR)** capturing the choice and trade-off.

## The production bar (non-negotiable -- applies to every phase)
Nothing is "done" until it clears all of these.

- **Testing.** Unit tests for logic; integration tests against real data via local fixtures and
  testcontainers (Postgres for metadata); core index and search logic covered by property-based
  tests (Hypothesis) asserting invariants. CI is the gate -- red build never merges.
- **Eval gate.** Every retrieval or generation change is measured against the gold eval set before
  merge. Faithfulness, nDCG@10, and EM regressions block the build just like a failing test.
- **Type safety and style.** `mypy --strict` clean; `ruff` lint and format clean; enforced by
  pre-commit locally and CI. No suppressed warnings without a justified `# noqa` comment.
- **Security.** Auth on every endpoint; all inputs validated with Pydantic; secrets via env / AWS
  Secrets Manager; `pip-audit` and Trivy image scanning in CI; no keys or tokens in code or git.
- **Observability.** Structured JSON logs with a `request_id` propagated end-to-end; Prometheus
  metrics with per-stage latency (embed, retrieve, rerank, generate); defined P95 budgets and a
  Grafana dashboard; alerts on latency and eval-score drift.
- **Reliability.** Graceful shutdown; `/health` and `/ready` probes; explicit timeouts on every
  external call (embedding API, LLM); bounded concurrency and connection pools; caching layer with
  TTL to cap cost.
- **Operability.** 12-factor config via `pydantic-settings`; runbooks for each failure mode; ADRs
  for each significant decision; `PROGRESS.md` checklist updated per phase.
- **Delivery.** CI/CD with ordered quality gates: `test --> lint --> type-check --> eval-gate -->
  security scan --> build --> image scan --> deploy` -- automated rollback on failed health checks.

## Headline outcomes (the success contract -- we build to hit these exactly)
These three bullets are the definition of done for the whole project.

1. **ANN retrieval: benchmarked, profiled, and optimized (the optimization story).** Benchmarked
   FAISS `IndexHNSWFlat` and `IndexIVFPQ` against exact brute-force over a **6M+ vector** Wikipedia
   corpus; profiled under load with `py-spy` to find the batching bottleneck via flamegraph;
   batched the search path for a **3x QPS improvement** -- reaching **97% recall@10 at P99 < 5 ms**
   at **2,000+ QPS**, with a CI perf regression budget. *(Stretch: swap in a hand-written HNSW
   index after Phase 10 for the hard-CS differentiator.)*

2. **Agentic multi-hop RAG, proven on public benchmarks.** Built an agentic multi-hop RAG
   (iterative retrieve-reason-retrieve, self-reflection, and answer verification with grounded
   citations), improving **exact-match +15 points on HotpotQA** over single-shot RAG, and
   **nDCG@10 +0.20 on BEIR** retrieval tasks with hybrid search + cross-encoder reranking.

3. **Eval-gated, production-served.** Engineered a benchmark-driven evaluation harness (BEIR
   retrieval metrics + RAGAS-style faithfulness on a gold set) wired as a **CI gate that blocks
   regressions**, reaching **93% answer faithfulness** with abstention on low-evidence queries;
   served behind a streaming API at **end-to-end P95 < 800 ms** with caching cutting cost to
   **~$0.004/query**.

## What makes this exceptional (beyond production -- the differentiators)
Production-grade is table stakes. These five choices separate this from every other RAG repo.

1. **Tell an optimization story with a real before/after -- and make the flamegraph the star.**
   Most people call FAISS and move on. Benchmarking index types (HNSW vs IVF-PQ), profiling under
   load with py-spy, finding the batching bottleneck from a flamegraph, and reporting a 3x
   multiplier proves the diagnostic skill interviewers actually test. *(Stretch: replace FAISS with
   a hand-written HNSW index after Phase 10 for the hard-CS differentiator.)*
2. **Prove quality on a public benchmark, not your own 10 questions.** HotpotQA exact-match and
   BEIR nDCG@10 are checkable, reproducible claims. That credibility is half the value of bullet 2.
3. **Tell an agentic reasoning story with real numbers.** The multi-hop loop, self-reflection, and
   citation grounding are each individually provable as improvements -- remove one and the eval
   score drops, captured as a regression. That staircase of evidence is the story.
4. **Add one genuinely hard RAG technique.** Agentic multi-hop (iterative retrieve-reason-retrieve
   with self-reflection) is qualitatively more interesting than vanilla RAG and maps directly to
   real research (ReAct, Self-RAG, FLARE). Combined with a public benchmark, it's a publishable
   result.
5. **Make the artifact look production in 30 seconds.** ADRs, an architecture diagram, a
   "failures I induced" section, the benchmark report with graphs, the before/after flamegraph, and
   a cost analysis. Half of "exceptional" is a reviewer *seeing* the depth at a skim.

## Non-functional requirements / SLOs
- **Retrieval throughput:** serve **>= 2,000 QPS** (batched ANN queries) on a single node.
- **Retrieval latency:** FAISS search **P99 < 5 ms** at 6M vectors (batched queries).
- **Retrieval quality:** **recall@10 >= 97%** vs brute-force (`IndexFlatL2`) on the same corpus.
- **Optimization story:** before/after QPS multiplier **>= 3x** from flamegraph-identified fix.
- **HotpotQA lift:** agentic pipeline **>= +10 EM points** over single-shot RAG baseline.
- **BEIR nDCG@10:** hybrid + rerank beats dense-only by **>= +0.10 nDCG@10** on >= 2 datasets.
- **Faithfulness:** **>= 90% faithfulness** (RAGAS-style LLM judge + human spot-check on gold set).
- **End-to-end P95:** streaming response starts within **800 ms** (retrieval + rerank + LLM).
- **Cost:** average query cost **<= $0.005** (batched embeddings + cached retrieval + LLM).
- **Abstention:** model declines to answer when retrieved evidence score is below a threshold
  (configurable), reducing hallucination rate on out-of-corpus queries.

## Tech constraints
- **Language:** Python 3.12 (FastAPI for serving).
- **Corpus:** Wikipedia (6M+ English articles, clean text, HotpotQA-native); full corpus used for
  all three headline outcomes — HNSW benchmark, multi-hop QA, and production eval.
- **Vector index:** FAISS (`IndexHNSWFlat` primary; `IndexIVFPQ` comparison). Benchmarked,
  profiled, and tuned -- not just called. Stretch goal: replace with a hand-written HNSW index
  after Phase 10 if time allows.
- **Embeddings:** open model (sentence-transformers `bge-large-en`) for reproducibility; batched
  async calls.
- **Reranker:** cross-encoder (BGE reranker or Cohere Rerank API) for the hybrid pipeline.
- **LLM:** Claude API (claude-sonnet-4-6 default; claude-haiku-4-5 for cheaper eval runs).
- **Stores:** FAISS index file on disk; SQLite / Postgres for doc metadata and eval results; Redis
  for caching and rate limiting.
- **Tooling:** `uv`, `ruff`, `mypy --strict`, `pytest` + `hypothesis`, `pre-commit`, `py-spy` for
  flamegraphs, Prometheus/Grafana, Terraform, GitHub Actions.
- **Depth-over-breadth rule:** use FAISS for the vector index but go deep -- benchmark index types,
  profile under load, tune from evidence, report a measured multiplier. The agentic loop
  (retrieve-reason-retrieve, self-reflection, citation grounding) is hand-written, not a LangChain
  abstraction. That ownership is the whole point.
- **Infra:** Docker Compose locally; AWS (ECS Fargate / S3 / ElastiCache) via Terraform in the
  cloud.
- Keep dependencies minimal and prefer the stdlib / one obvious library over frameworks that hide
  the mechanism -- but never at the cost of the production bar.

## Definition of done (per phase)
Code runs and is demoable; it clears the entire production bar (tests green in CI, types/lint
clean, eval gate passing, secured, observable); I can explain *why* every component exists and what
I rejected; I have watched the relevant failure happen, be handled, and be pinned by a test or eval
regression; and I have written the ADR + resume bullet (with a measured metric). Track progress in
`PROGRESS.md`. Don't advance phases until I confirm I understand.

## My background
Strong Python/ML, comfortable with NumPy and ML concepts (embeddings, transformers). Weak on:
information retrieval theory (BM25, ANN algorithms, reranking), production ML serving (latency
budgets, batching, caching), and production engineering practice (testing, CI/CD, IaC,
observability). Targeting MAANG new-grad SDE / MLE. Studying system design in parallel -- tie
concepts to it.

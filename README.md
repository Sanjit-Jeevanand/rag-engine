# RAG Engine

**Hybrid HNSW + BM25 retrieval, cross-encoder reranking, and a hand-rolled multi-hop agent that cites its sources — or refuses to answer.**

Built from scratch across 10 phases: corpus ingestion → eval harness → FAISS benchmarking → throughput engineering → hybrid retrieval → agentic reasoning → full benchmark run → production serving → observability → containerised deploy on a €7.49/mo Hetzner server.

```
POST /query  →  embed  →  HNSW ┐
                           BM25 ┘  →  RRF  →  rerank  →  agent loop  →  SSE answer
                ↑                                                              ↓
         semantic cache (cosine ≥ 0.97 → skip entire pipeline, ~1 ms)     citations
```

---

## Numbers

| Metric | Value |
|--------|-------|
| HotpotQA Exact Match — single-shot | 0.29 |
| HotpotQA Exact Match — multi-hop | **0.49** (+20 EM pts) |
| nDCG@10 — 1M article benchmark | 0.4618 |
| Recall@10 — 1M article benchmark | 0.4780 |
| MRR — 1M article benchmark | 0.5994 |
| Retrieval latency p50 (8.8M vectors) | **0.387 ms** |
| Search throughput | **18,471 QPS** |
| HNSW recall@10 vs. exact | 98.6% |
| Cost per query | **$0.0006** (8× under $0.005 target) |
| Fabricated citations | **0 / 100** |
| Out-of-corpus abstentions | **4 / 5** |

### BEIR

| Dataset | Corpus | Queries | nDCG@10 | Recall@10 | MRR |
|---------|--------|---------|---------|-----------|-----|
| SciFact | 5,183 docs | 300 | **0.7253** | 0.8529 | 0.6917 |
| NFCorpus | 3,633 docs | 323 | **0.3311** | 0.1609 | 0.5383 |
| ArguAna | 8,674 docs | 1,406 | **0.2826** | 0.6166 | 0.1793 |

Scored with `pytrec_eval`. SciFact beats the BM25 baseline (0.665). ArguAna's counter-argument structure is a known dense-retrieval failure mode — reported, not hidden.

---

## Architecture

### Retrieval pipeline

```
Query
  │
  ├─ Embed              bge-small-en-v1.5 · 384d
  │
  ├─ Dense retrieval    FAISS IndexHNSWFlat · M=32 · ef=64 · 0.387ms p50
  ├─ Sparse retrieval   bm25s · lexical match
  │
  ├─ Fusion             Reciprocal Rank Fusion (RRF) — parameter-free
  │
  ├─ Rerank             bge-reranker-base cross-encoder · top-10 candidates
  │
  └─ Agent loop         GPT-4o-mini · up to 3 hops
       ├─ Hop 1: retrieve on original question
       ├─ Bridge: extract next entity ("what must be looked up next?")
       ├─ Hop 2: retrieve on bridge query
       ├─ Reflect: check if answer is fully supported; search again if not
       └─ Answer: ANSWER: <concise> + CITATIONS: [n, m]
```

### Multi-hop agent behaviour

- **Bridge extraction** reads the top-3 passages before deciding the next query
- **Reranker abstention**: if `max(scores[:3]) < threshold` → returns without an LLM call
- **LLM abstention**: model outputs a sentinel if passages are insufficient → `abstained: true`
- **Citation grounding**: every cited ID is verified against the retrieved set; hallucinated IDs are flagged
- **Reflection**: a third LLM call checks `FULLY_SUPPORTED: yes/no`; on `no`, a gap query triggers hop 3

### Semantic cache

Redis Hash stores embedding vectors for past queries. Every new query is compared via cosine similarity; a hit (≥ 0.97) returns the full `QueryResult` in ~1 ms, skipping embedding, retrieval, reranking, and generation entirely.

---

## Production stack

```
┌─────────────────────────────────────────────────────┐
│  Hetzner CX33 · 4 vCPU · 8 GB RAM · €7.49/mo       │
│                                                     │
│  ┌─────────┐  ┌───────┐  ┌──────────┐              │
│  │   API   │  │ Redis │  │ Postgres │              │
│  │FastAPI  │  │ cache │  │ api_keys │              │
│  │uvicorn  │  │rate   │  │ SHA-256  │              │
│  └─────────┘  └───────┘  └──────────┘              │
│  ┌────────────┐  ┌─────────┐                        │
│  │ Prometheus │  │ Grafana │                        │
│  │ RED method │  │ 8 panels│                        │
│  └────────────┘  └─────────┘                        │
└─────────────────────────────────────────────────────┘
```

- **API**: multi-stage Docker image, non-root `rag` user, `OMP_NUM_THREADS=1` to prevent PyTorch SIGSEGV
- **Redis**: semantic cache + sliding-window rate limiter (Lua script, 100 req/min per tenant)
- **Postgres**: `api_keys` table — raw tokens never stored, only their SHA-256 hash
- **Prometheus + Grafana**: QPS, error rate, per-stage P95 latency, hop count histogram, 3 SLO gauges
- **SLOs**: P95 latency < 3 s · error ratio < 5% · faithfulness ≥ rolling 7-day avg − 2 pts

---

## Corpus

The production index covers the **top 100K Wikipedia articles by incoming links** — every topic an interviewer is likely to ask about.

| | Articles | Vectors | Index | RAM |
|-|----------|---------|-------|-----|
| Benchmark | 1,000,000 | 8.8M | 15.9 GB | ~13 GB |
| **Production** | **100,000** | **2.3M** | **4.17 GB** | **~3.5 GB** |

The 1M benchmark corpus is kept locally as `*_1m.*` files. The FAISS index assigns sequential IDs 0,1,2… and `vector_offset` in the DB is renumbered to match — no translation table.

---

## Quick start

### Prerequisites

- Python 3.12, [uv](https://github.com/astral-sh/uv)
- Docker + Docker Compose
- `OPENAI_API_KEY` in `.env`

### Run locally

```bash
git clone https://github.com/Sanjit-Jeevanand/rag-engine
cd rag-engine
uv sync

# Start Redis
docker compose -f infra/docker-compose.yml up -d redis

# Start the API
PYTHONPATH=src uv run uvicorn rag_engine.api.app:app --port 8000

# Open the UI
open web/index.html
```

Set your API base URL to `http://localhost:8000` and a bearer token matching `RAG_TENANT_TOKENS` in your `.env`.

### Run with the full stack

```bash
# Copy .env.example → .env and fill in values
cp .env.example .env

docker compose -f infra/docker-compose.yml up -d
# API      → http://localhost:8000
# Grafana  → http://localhost:3000
```

### Reproduce the eval

```bash
# HotpotQA hybrid + agentic
PYTHONPATH=src:. uv run --env-file .env python scripts/hotpotqa_agentic_eval.py --n 100

# BEIR staircase
PYTHONPATH=src:. uv run python scripts/beir_eval.py

# Perf gate (HNSW throughput)
PYTHONPATH=src:. uv run python scripts/perf_check.py --record
make perf
```

---

## Deploy

### Provision a server (Terraform)

```bash
# Generate an SSH key if you don't have one
ssh-keygen -t ed25519 -C "rag-engine" -f ~/.ssh/id_ed25519

terraform -chdir=infra/terraform init
terraform -chdir=infra/terraform apply \
    -var="ssh_public_key=$(cat ~/.ssh/id_ed25519.pub)"

# Outputs:
#   server_ip   = "167.233.55.52"
#   api_url     = "http://167.233.55.52:8000"
#   grafana_url = "http://167.233.55.52:3000"
```

### Transfer data and start

```bash
# Transfer Compose stack files
rsync -avz infra/ root@<IP>:/opt/rag-engine/infra/

# Transfer production data (~11 GB total)
rsync -avz --progress data/docs.db data/hnsw.index data/vectors.bin data/bm25_index \
    root@<IP>:/opt/rag-engine/data/

# Start all 5 services
ssh root@<IP> "cd /opt/rag-engine/infra && docker compose up -d"
```

### GitHub Actions CI/CD

Push builds a `linux/amd64` image and deploys it via SSH. Add these repository secrets:

| Secret | Value |
|--------|-------|
| `GHCR_PAT` | GitHub classic token with `write:packages` |
| `HETZNER_HOST` | Server IP |
| `SSH_PRIVATE_KEY` | Contents of `~/.ssh/id_ed25519` |

Trigger: **Actions → Deploy → Run workflow**

### Generate API keys

```bash
# Run from inside the server (Postgres is not exposed externally)
ssh root@<IP> bash -s <<'EOF'
docker compose -f /opt/rag-engine/infra/docker-compose.yml exec -T api \
    python scripts/gen_api_keys.py \
    --db-url postgresql://rag:rag@postgres:5432/rag \
    --count 10 --prefix interviewer
EOF

# Revoke a key
docker exec <postgres-container> psql -U rag rag \
    -c "UPDATE api_keys SET active = FALSE WHERE tenant_id = 'interviewer-03';"
```

---

## Project structure

```
src/rag_engine/
├── api/
│   ├── app.py          FastAPI app, SSE generator, lifespan
│   ├── auth.py         SHA-256 hashing, asyncpg pool, env-dict fallback
│   ├── cache.py        Redis semantic cache (cosine similarity)
│   ├── metrics.py      Prometheus instruments (6 metrics)
│   ├── models.py       Pydantic request/response models
│   ├── ratelimit.py    Lua sliding-window rate limiter
│   └── stream.py       Named-event SSE formatters
├── agent/
│   ├── loop.py         MultiHopAgent: bridge · cite · reflect · abstain
│   └── llm.py          OpenAI client, complete(), stream_complete()
├── ingest/
│   ├── downloader.py   Streaming download, .tmp rename
│   ├── parser.py       Wikipedia JSONL → WikiArticle iterator
│   └── schema.py       SQLite schema, WAL mode
├── retrieval/
│   ├── bm25.py         BM25Retriever (bm25s)
│   ├── dense.py        DenseRetriever (FAISS IndexFlatIP)
│   ├── hybrid.py       Reciprocal Rank Fusion
│   └── reranker.py     CrossEncoderReranker (bge-reranker-base)
├── config.py           Pydantic-settings, RAG_ env prefix
├── cost.py             Thread-local cost tracker ($0.0006/query)
└── log.py              structlog JSON, request_id ContextVar

scripts/
├── build_production_index.py   Top-100K HNSW index from memmap
├── build_100k_dataset.py       Rename 1M files, rebuild aligned DB + vectors
├── gen_api_keys.py             Generate sk-* tokens → Postgres
├── hotpotqa_agentic_eval.py    Single-shot vs multi-hop, n=100
├── hotpotqa_full_eval.py       Full run with cost tracking
└── beir_eval.py                BEIR staircase (SciFact · NFCorpus · ArguAna)

infra/
├── docker-compose.yml          5 services: api · redis · postgres · prometheus · grafana
├── postgres/init.sql           api_keys table
├── prometheus-stack.yml        Scrape config + alert_rules.yml
└── terraform/                  Hetzner CX33 IaC

eval/
├── metrics.py                  nDCG@k · Recall@k · MRR · EM · F1 (pure Python)
├── hotpotqa_eval.py            Hybrid pipeline eval on 1K gold questions
├── comparator.py               Regression gate (±2% tolerance)
└── results/
    ├── baseline.json           nDCG=0.4618 · Recall=0.478 · MRR=0.5994
    └── latest.json             Last eval run
```

---

## CI

Every commit runs:

```
lint (ruff)  →  typecheck (mypy --strict)  →  test (pytest)  →  audit (pip-audit)  →  eval gate
```

The eval gate blocks merge if any metric drops more than 2 percentage points below the stored baseline. A performance gate (`make perf`) checks that HNSW throughput stays within 10% of the recorded 18,471 QPS baseline.

---

## Stack

| Layer | Technology |
|-------|-----------|
| Embeddings | `bge-small-en-v1.5` · 384d · sentence-transformers |
| Dense index | FAISS `IndexHNSWFlat` · M=32 · ef=64 |
| Sparse index | `bm25s` |
| Fusion | Reciprocal Rank Fusion (parameter-free) |
| Reranker | `bge-reranker-base` cross-encoder |
| LLM | GPT-4o-mini |
| API | FastAPI · uvicorn · SSE (`StreamingResponse`) |
| Cache | Redis · cosine similarity over embedding vectors |
| Auth | Postgres `api_keys` · SHA-256 hashed tokens · asyncpg |
| Observability | Prometheus · Grafana · structlog JSON |
| Packaging | `uv` · Python 3.12 |
| Container | Docker multi-stage · non-root user |
| IaC | Terraform · Hetzner `hcloud` provider |
| CI/CD | GitHub Actions · ghcr.io · SSH deploy |

---

## Phases

| # | Phase | Key result |
|---|-------|-----------|
| 0 | Scaffolding | CI pipeline: lint → typecheck → test → audit → eval gate |
| 1 | Corpus ingestion | 6M Wikipedia articles parsed, chunked, embedded (8.8M vectors) |
| 2 | Eval harness | nDCG@10 · Recall@10 · MRR · EM · F1; regression gate with 2% tolerance |
| 3 | HNSW vs IVFPQ | HNSW ef=64: 0.387ms p50, 98.6% recall — 300× faster than exact search |
| 4 | Throughput | 18,471 QPS at c=8; batch=256 ceiling 20,822 QPS (off critical path) |
| 5 | Hybrid retrieval | RRF: nDCG 0.46 → 0.54 (+0.08); +rerank → 0.70 (+0.24) |
| 6 | Agentic multi-hop | EM 0.29 → 0.49 (+20 pts); 0/100 hallucinations; 4/5 abstentions |
| 7 | Full benchmark | BEIR staircase; $0.0006/query; cost tracked thread-locally per request |
| 8 | Production serving | FastAPI SSE; Redis semantic cache; rate limiting; 13 API tests |
| 9 | Observability | Prometheus RED metrics; Grafana dashboard; 3 SLO alert rules |
| 10 | Deploy | Docker · Terraform · Postgres auth · GitHub Actions · Hetzner CX33 |

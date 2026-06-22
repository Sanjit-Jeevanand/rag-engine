"""
Generate a comprehensive reproduction guide PDF for the RAG Engine project.
Run with:  uv run python scripts/gen_reproduction_guide.py
Output:    rag_engine_reproduction_guide.pdf
"""

from __future__ import annotations

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

W, H = A4
MARGIN = 2 * cm
INNER_W = W - 2 * MARGIN

# ── Colour palette ────────────────────────────────────────────────────────────
C_BG_DARK = colors.HexColor("#0f172a")
C_ACCENT = colors.HexColor("#6366f1")
C_AMBER = colors.HexColor("#f59e0b")
C_GREEN = colors.HexColor("#10b981")
C_CODE_BG = colors.HexColor("#1e293b")
C_CODE_FG = colors.HexColor("#e2e8f0")
C_LIGHT_BORDER = colors.HexColor("#334155")
C_MUTED = colors.HexColor("#94a3b8")
C_WHITE = colors.white

# ── Styles ─────────────────────────────────────────────────────────────────────
base = getSampleStyleSheet()

TITLE_S = ParagraphStyle(
    "Title",
    fontName="Helvetica-Bold",
    fontSize=28,
    textColor=C_WHITE,
    spaceAfter=6,
    alignment=TA_CENTER,
)
SUBTITLE_S = ParagraphStyle(
    "Subtitle",
    fontName="Helvetica",
    fontSize=13,
    textColor=C_MUTED,
    spaceAfter=4,
    alignment=TA_CENTER,
)
H1_S = ParagraphStyle(
    "H1",
    fontName="Helvetica-Bold",
    fontSize=18,
    textColor=C_ACCENT,
    spaceBefore=20,
    spaceAfter=8,
)
H2_S = ParagraphStyle(
    "H2",
    fontName="Helvetica-Bold",
    fontSize=13,
    textColor=C_AMBER,
    spaceBefore=14,
    spaceAfter=6,
)
H3_S = ParagraphStyle(
    "H3",
    fontName="Helvetica-Bold",
    fontSize=11,
    textColor=C_GREEN,
    spaceBefore=10,
    spaceAfter=4,
)
BODY_S = ParagraphStyle(
    "Body",
    fontName="Helvetica",
    fontSize=9.5,
    textColor=colors.HexColor("#cbd5e1"),
    leading=15,
    spaceAfter=6,
)
BULLET_S = ParagraphStyle(
    "Bullet",
    parent=BODY_S,
    leftIndent=14,
    bulletIndent=0,
    spaceAfter=3,
)
NOTE_S = ParagraphStyle(
    "Note",
    fontName="Helvetica-Oblique",
    fontSize=8.5,
    textColor=C_MUTED,
    leftIndent=10,
    spaceAfter=4,
)
CODE_S = ParagraphStyle(
    "Code",
    fontName="Courier",
    fontSize=7.8,
    textColor=C_CODE_FG,
    backColor=C_CODE_BG,
    leftIndent=8,
    rightIndent=8,
    spaceAfter=8,
    spaceBefore=4,
    leading=12,
    borderPad=6,
)

# ── Helpers ────────────────────────────────────────────────────────────────────


def h1(text: str) -> Paragraph:
    return Paragraph(text, H1_S)


def h2(text: str) -> Paragraph:
    return Paragraph(text, H2_S)


def h3(text: str) -> Paragraph:
    return Paragraph(text, H3_S)


def body(text: str) -> Paragraph:
    return Paragraph(text, BODY_S)


def note(text: str) -> Paragraph:
    return Paragraph(f"<i>{text}</i>", NOTE_S)


def bullet(text: str) -> Paragraph:
    return Paragraph(f"•  {text}", BULLET_S)


def code(text: str) -> Preformatted:
    return Preformatted(text.strip("\n"), CODE_S)


def hr() -> HRFlowable:
    return HRFlowable(width="100%", thickness=1, color=C_LIGHT_BORDER, spaceAfter=10)


def sp(n: float = 6) -> Spacer:
    return Spacer(1, n)


def metric_table(rows: list[tuple[str, str]]) -> Table:
    data = [["Metric", "Value"]] + list(rows)
    t = Table(data, colWidths=[INNER_W * 0.55, INNER_W * 0.45])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), C_ACCENT),
                ("TEXTCOLOR", (0, 0), (-1, 0), C_WHITE),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("FONTNAME", (0, 1), (-1, -1), "Courier"),
                ("TEXTCOLOR", (0, 1), (-1, -1), C_CODE_FG),
                ("BACKGROUND", (0, 1), (-1, -1), C_CODE_BG),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [C_CODE_BG, colors.HexColor("#0f1e2e")],
                ),
                ("GRID", (0, 0), (-1, -1), 0.4, C_LIGHT_BORDER),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return t


# ── Page templates ─────────────────────────────────────────────────────────────


def on_page(canvas, doc):  # type: ignore[no-untyped-def]
    canvas.saveState()
    canvas.setFillColor(C_BG_DARK)
    canvas.rect(0, 0, W, H, fill=1, stroke=0)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(C_MUTED)
    canvas.drawString(MARGIN, 0.7 * cm, "RAG Engine — Reproduction Guide")
    canvas.drawRightString(W - MARGIN, 0.7 * cm, f"Page {doc.page}")
    canvas.restoreState()


# ── Content builders ───────────────────────────────────────────────────────────


def cover_page() -> list:
    return [
        sp(80),
        Paragraph("RAG Engine", TITLE_S),
        Paragraph("Complete Reproduction Guide", SUBTITLE_S),
        sp(4),
        Paragraph(
            "10-phase build: corpus ingestion → HNSW benchmarking → agentic multi-hop"
            " → production deploy on Hetzner",
            SUBTITLE_S,
        ),
        sp(30),
        metric_table(
            [
                ("HotpotQA EM (multi-hop)", "0.49  (+20 pts over single-shot)"),
                ("nDCG@10 — 1M article benchmark", "0.4618"),
                ("Retrieval latency p50 (8.8M vectors)", "0.387 ms"),
                ("Search throughput", "18,471 QPS"),
                ("Cost per query", "$0.0006"),
                ("Fabricated citations", "0 / 100"),
                ("Production server", "Hetzner CX33 — EUR 7.49 / mo"),
            ]
        ),
        PageBreak(),
    ]


def prerequisites() -> list:
    return [
        h1("Prerequisites & Environment"),
        hr(),
        h2("System requirements"),
        bullet("Python 3.12 (via pyenv or system package)"),
        bullet(
            "uv package manager  —  curl -LsSf https://astral.sh/uv/install.sh | sh"
        ),
        bullet("Docker + Docker Compose plugin (Docker Desktop or engine)"),
        bullet("Terraform >= 1.6  (for Hetzner deploy only)"),
        bullet("~30 GB free disk (Wikipedia dump + vectors + indexes)"),
        bullet("~8 GB RAM minimum; 16 GB recommended during index build"),
        sp(),
        h2("API keys needed"),
        bullet(
            "OPENAI_API_KEY — GPT-4o-mini for answer generation & bridge extraction"
        ),
        bullet("HCLOUD_TOKEN — Hetzner Cloud API token (deploy phase only)"),
        sp(),
        h2("Clone and install"),
        code("""git clone https://github.com/Sanjit-Jeevanand/rag-engine
cd rag-engine
uv sync                        # installs all deps including dev extras"""),
        h2("Create .env"),
        code("""# .env  (never commit this file)
OPENAI_API_KEY=sk-...
RAG_REDIS_URL=redis://localhost:6379
RAG_DB_URL=                    # leave empty to use env-dict auth locally
RAG_TENANT_TOKENS='{\"my-dev-token\": \"dev\"}'
HCLOUD_TOKEN=...               # only needed for Terraform"""),
        note(
            "All RAG_ env vars are read by pydantic-settings (src/rag_engine/config.py)."
            " The prefix is RAG_."
        ),
        PageBreak(),
    ]


def phase0_scaffolding() -> list:
    return [
        h1("Phase 0 — Scaffolding & CI"),
        hr(),
        body(
            "Every commit runs a five-step pipeline: lint (ruff) → typecheck (mypy --strict)"
            " → test (pytest) → audit (pip-audit) → eval gate. "
            "The Makefile mirrors GitHub Actions exactly so failures are caught locally first."
        ),
        sp(),
        h2("Makefile targets"),
        code("""make lint       # ruff check + ruff format --check
make typecheck  # mypy --strict over src/
make test       # pytest (excludes heavy FAISS property tests)
make audit      # pip-audit; ignores known no-fix CVEs
make eval-gate  # checks eval/results/latest.json sentinel key
make ci         # all of the above in order"""),
        h2("pyproject.toml — key sections"),
        code("""[project]
name = "rag-engine"
requires-python = ">=3.12"
dependencies = [
    "faiss-cpu>=1.14.2",
    "sentence-transformers>=5.5.1",
    "torch>=2.0",
    "bm25s>=0.2.12",
    "openai>=2.41.1",
    "fastapi>=0.136.3",
    "uvicorn[standard]>=0.49.0",
    "redis>=8.0.0",
    "asyncpg>=0.29.0",
    "prometheus-client>=0.25.0",
    "structlog>=24.1",
    "pydantic-settings>=2.3",
]

# CPU-only torch (avoids 4 GB NVIDIA CUDA download)
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cpu" }
torchvision = { index = "pytorch-cpu" }

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]"""),
        h2("Pre-commit hooks"),
        code(
            """uv run pre-commit install    # installs ruff + mypy hooks on git commit"""
        ),
        PageBreak(),
    ]


def phase1_ingest() -> list:
    return [
        h1("Phase 1 — Corpus Ingestion"),
        hr(),
        body(
            "Download the Wikipedia JSONL dump (~10 GB compressed), parse every article,"
            " split into 1500-char chunks with 200-char overlap, and store in SQLite."
            " Then embed all chunks with bge-small-en-v1.5 (384d) into a memory-mapped"
            " float32 binary (vectors.bin)."
        ),
        sp(),
        h2("Step 1 — Download Wikipedia dump"),
        code("""# Downloads wiki-dump.json.gz to data/
uv run python scripts/download_wiki_dump.py"""),
        h2("Step 2 — Parse & chunk into SQLite"),
        code("""# Produces data/docs.db with table 'documents'
uv run python scripts/run_pipeline.py \\
    --snapshot data/wiki-dump.json.gz \\
    --db       data/docs.db"""),
        h3("Schema (src/rag_engine/ingest/schema.py)"),
        code("""CREATE TABLE IF NOT EXISTS documents (
    article_id    TEXT    NOT NULL,
    chunk_index   INTEGER NOT NULL,
    title         TEXT    NOT NULL,
    categories    TEXT    NOT NULL,
    timestamp     TEXT    NOT NULL,
    chunk_text    TEXT    NOT NULL,
    chunk_count   INTEGER NOT NULL,
    vector_offset INTEGER,            -- row index into vectors.bin
    status        TEXT    NOT NULL DEFAULT 'pending',
    PRIMARY KEY (article_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS idx_status ON documents (status);"""),
        h3("Chunking logic (src/rag_engine/ingest/pipeline.py)"),
        code("""CHUNK_CHARS  = 1500   # chars per chunk
OVERLAP_CHARS = 200   # overlap between adjacent chunks

def split_text(text, chunk_chars=1500, overlap_chars=200):
    step = chunk_chars - overlap_chars
    return [text[i : i + chunk_chars] for i in range(0, len(text), step)]"""),
        h2("Step 3 — Embed all chunks"),
        code("""# Reads docs.db, embeds pending chunks with bge-small-en-v1.5,
# writes float32 vectors to data/vectors.bin and updates vector_offset in DB.
# ~6–8 hours on CPU for 6M Wikipedia articles (8.8M chunks).
PYTHONPATH=src uv run python scripts/run_embedder.py \\
    --db       data/docs.db \\
    --out      data/vectors.bin \\
    --batch    256"""),
        note(
            "bge-small-en-v1.5 outputs 384-dimensional L2-normalised vectors."
            " The binary file is a raw float32 memmap: shape (N, 384)."
        ),
        PageBreak(),
    ]


def phase2_eval() -> list:
    return [
        h1("Phase 2 — Eval Harness"),
        hr(),
        body(
            "A pure-Python eval suite — no frameworks — computes nDCG@10, Recall@10,"
            " MRR, Exact Match, and F1. A regression gate blocks any commit that drops"
            " any metric by more than 2 percentage points."
        ),
        sp(),
        h2("Core metrics (eval/metrics.py)"),
        code("""def ndcg_at_k(retrieved, relevant, k=10):
    dcg  = sum(1/log2(i+2) for i,t in enumerate(retrieved[:k]) if t in relevant)
    idcg = sum(1/log2(i+2) for i in range(min(k, len(relevant))))
    return dcg / idcg if idcg > 0 else 0.0

def recall_at_k(retrieved, relevant, k=10):
    return len(set(retrieved[:k]) & relevant) / len(relevant) if relevant else 0.0

def mrr(retrieved, relevant):
    for i, t in enumerate(retrieved):
        if t in relevant: return 1.0 / (i + 1)
    return 0.0

def exact_match(prediction, gold):
    return 1.0 if _normalize(prediction) == _normalize(gold) else 0.0

def _normalize(text):
    text = text.lower()
    text = re.sub(r'\\b(a|an|the)\\b', ' ', text)
    text = re.sub(r'[^a-z0-9 ]', '', text)
    return re.sub(r'\\s+', ' ', text).strip()"""),
        h2("Seed the gold set"),
        code(
            """uv run python scripts/seed_gold_set.py   # writes eval/hotpotqa_gold.json"""
        ),
        h2("Run BEIR evaluation"),
        code("""PYTHONPATH=src:. uv run python scripts/beir_eval.py
# SciFact 0.7253  NFCorpus 0.3311  ArguAna 0.2826"""),
        h2("Regression gate"),
        code("""# eval/gate.py — run by 'make eval-gate'
# Compares eval/results/latest.json against eval/results/baseline.json
# Fails if any metric drops > 2 pp.
make eval-gate"""),
        PageBreak(),
    ]


def phase3_hnsw() -> list:
    return [
        h1("Phase 3 — HNSW Index"),
        hr(),
        body(
            "Build a FAISS IndexHNSWFlat over all 8.8M embedded vectors."
            " Benchmarked ef=64 as the Pareto knee: 98.6% recall@10 at 0.387ms p50."
        ),
        sp(),
        h2("Build the full 1M-article benchmark index"),
        code("""uv run python scripts/build_hnsw_index.py \\
    --vectors  data/vectors.bin \\
    --db       data/docs.db \\
    --out      data/hnsw_1m.index \\
    --m        32 \\
    --ef-construction 200 \\
    --ef-search 64"""),
        h2("Build the production 100K-article index"),
        code("""# Selects top 100K articles by incoming_links, loads their vectors
# from the existing memmap — no re-embedding required.
uv run python scripts/build_production_index.py --limit 100000
# Output: data/hnsw_100k.index

# Install as production files (renames + rebuilds aligned docs.db)
uv run python scripts/build_100k_dataset.py"""),
        h2("HNSW parameters"),
        metric_table(
            [
                (
                    "Index type",
                    "IndexHNSWFlat (cosine via inner-product on L2-normalised vecs)",
                ),
                ("M (connections per node)", "32"),
                ("efConstruction", "200"),
                ("efSearch (runtime)", "64  — set in config.py + overridden on load"),
                ("Vector dimension", "384"),
                ("Production vectors", "2.3M (100K articles, ~23 chunks/article)"),
                ("Index file size", "4.17 GB"),
                ("RAM usage", "~3.5 GB"),
            ]
        ),
        sp(8),
        h2("Key insight — ef vs latency/recall"),
        code("""# FAISS enforces: efSearch = max(efSearch, k)
# Searching k=1000 silently sets ef=1000 — 15x slower than ef=64.
# Always set _DENSE_K == efSearch to avoid this trap.

# src/rag_engine/api/app.py
_DENSE_K = 64   # matches hnsw_ef_search in config — intentional"""),
        h2("Run the HNSW benchmark"),
        code("""PYTHONPATH=src:. uv run python scripts/benchmark.py
# Reports: p50 / p95 / p99 latency, recall@10, QPS at various concurrency"""),
        PageBreak(),
    ]


def phase4_throughput() -> list:
    return [
        h1("Phase 4 — Throughput Engineering"),
        hr(),
        body(
            "Target: 10,000+ QPS on a single machine. Achieved 18,471 QPS at c=8"
            " by pinning OMP_NUM_THREADS=1, tuning batch sizes, and running parallel"
            " dense+sparse retrieval in a ThreadPoolExecutor."
        ),
        sp(),
        h2("Critical env flags"),
        code("""OMP_NUM_THREADS=1       # prevents PyTorch/FAISS thread oversubscription
TOKENIZERS_PARALLELISM=false   # avoids HuggingFace fork warnings"""),
        h2("Parallel dense + sparse retrieval"),
        code("""# src/rag_engine/api/app.py  — _retrieve_sync()
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    dense_fut  = pool.submit(_dense)    # FAISS HNSW search
    sparse_fut = pool.submit(_sparse)   # bm25s keyword search
    dense_ids  = dense_fut.result()
    sparse_ids = sparse_fut.result()    # both run concurrently"""),
        h2("Perf gate"),
        code("""# scripts/perf_check.py — run by 'make perf'
# Fails if QPS drops > 10% below baseline (18,471 QPS)
make perf"""),
        PageBreak(),
    ]


def phase5_hybrid() -> list:
    return [
        h1("Phase 5 — Hybrid Retrieval (HNSW + BM25 + RRF)"),
        hr(),
        body(
            "Dense HNSW retrieval excels at semantic similarity; BM25 handles exact"
            " keyword matching. Reciprocal Rank Fusion (RRF) combines both lists"
            " without requiring score calibration."
        ),
        sp(),
        h2("BM25 index build"),
        code("""# Built automatically during the ingest pipeline.
# Loads from data/bm25_index/ at startup.

# src/rag_engine/retrieval/bm25.py
class BM25Retriever:
    def __init__(self, doc_ids, doc_texts):
        tokens = bm25s.tokenize(doc_texts, stopwords='en', show_progress=False)
        self._bm25 = bm25s.BM25()
        self._bm25.index(tokens)

    @classmethod
    def load(cls, index_dir):
        doc_ids = json.loads((index_dir / 'doc_ids.json').read_text())
        obj = object.__new__(cls)
        obj._doc_ids = doc_ids
        obj._bm25 = bm25s.BM25.load(str(index_dir), load_corpus=False)
        return obj

    def retrieve(self, query, k):
        tokens = bm25s.tokenize([query], stopwords='en', show_progress=False)
        results, _ = self._bm25.retrieve(tokens, k=min(k, len(self._doc_ids)))
        return [self._doc_ids[int(i)] for i in results[0]]"""),
        h2("Reciprocal Rank Fusion (eval/hybrid.py)"),
        code("""_RRF_K = 60   # standard constant; insensitive to small changes

def reciprocal_rank_fusion(ranked_lists, k=10):
    scores = defaultdict(float)
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] += 1.0 / (_RRF_K + rank + 1)
    return sorted(scores, key=lambda d: scores[d], reverse=True)[:k]"""),
        h2("CrossEncoder reranker"),
        code("""# src/rag_engine/retrieval/reranker.py
# BAAI/bge-reranker-base — BERT-base cross-encoder
# Used in the multi-hop merge step, NOT in the hot query path
# (inference on 10 pairs = ~15s on EPYC CPU — too slow for P95 < 3s SLO)

class CrossEncoderReranker:
    def __init__(self, model_name='BAAI/bge-reranker-base'):
        self._model = CrossEncoder(model_name)

    def scores(self, query, candidates, doc_texts):
        pairs = [(query, doc_texts.get(doc_id, '')) for doc_id in candidates]
        return np.asarray(self._model.predict(pairs), dtype=np.float32)"""),
        h2("Eval results — hybrid vs dense-only"),
        metric_table(
            [
                ("Dense only (HNSW)", "nDCG@10 = 0.46"),
                ("+ BM25 RRF fusion", "nDCG@10 = 0.54  (+0.08)"),
                ("+ CrossEncoder rerank", "nDCG@10 = 0.70  (+0.24)"),
            ]
        ),
        PageBreak(),
    ]


def phase6_agent() -> list:
    return [
        h1("Phase 6 — Agentic Multi-Hop Reasoning"),
        hr(),
        body(
            "MultiHopAgent wraps the retrieval pipeline with up to 3 LLM calls:"
            " (1) bridge extraction — what to search next, (2) answer generation,"
            " (3) self-reflection — is the answer fully supported? "
            "EM jumps from 0.29 (single-shot) to 0.49 (+20 pts). Zero hallucinations."
        ),
        sp(),
        h2("Agent loop (src/rag_engine/agent/loop.py)"),
        code("""class MultiHopAgent:
    def answer(self, question):
        # Hop 1: retrieve on original question
        hop1 = self._retrieve(question, self._top_k)

        # Abstention: reranker score < threshold -> refuse before any LLM call
        top_scores = self._reranker.scores(question, hop1[:3], self._doc_texts)
        if np.max(top_scores) < self._abstention_threshold:   # default -4.0
            return AgentResult(abstained=True, ...)

        # LLM call 1: bridge extraction
        bridge = complete(messages, system=_BRIDGE_SYSTEM, max_tokens=64)
        # BRIDGE_SYSTEM: "Reply with ONLY a short search query or ANSWER_DIRECT"

        # Hop 2: retrieve on bridge entity
        if bridge.upper() != 'ANSWER_DIRECT':
            hop2 = self._retrieve(bridge, self._top_k)
            pool.extend(d for d in hop2 if d not in pool)

        # LLM call 2: answer generation with citations
        raw = complete(messages, system=_ANSWER_SYSTEM, max_tokens=512)
        # ANSWER_SYSTEM: forces "ANSWER: <text>\\nCITATIONS: [...]" format

        # LLM abstention: model says "I cannot answer from available evidence"
        if answer_text == _CANNOT_ANSWER:
            return AgentResult(abstained=True, ...)

        # LLM call 3: self-reflection
        raw_reflect = complete(messages, system=_REFLECT_SYSTEM, max_tokens=128)
        # REFLECT_SYSTEM: "FULLY_SUPPORTED: yes/no\\nMISSING: ...\\nSEARCH_QUERY: ..."
        if 'FULLY_SUPPORTED: no' in raw_reflect:
            # Hop 3: retrieve on gap query and regenerate
            gap = _extract_field(raw_reflect, 'SEARCH_QUERY')
            hop3 = self._retrieve(gap, self._top_k)
            raw = complete(...)   # LLM call 4

        # Citation grounding: verify every cited title is in retrieved set
        hallucinated = [c for c in cited if c not in retrieved_set]
        return AgentResult(cited_ids=cited, hallucinated_ids=hallucinated, ...)"""),
        h2("Run HotpotQA agentic eval"),
        code("""PYTHONPATH=src:. uv run --env-file .env \\
    python scripts/hotpotqa_agentic_eval.py --n 100
# Outputs to eval/results/agentic_eval_YYYYMMDD_HHMMSS.json
# Prints single-shot EM, multi-hop EM, hallucination count, abstention count"""),
        h2("Results"),
        metric_table(
            [
                ("Single-shot EM (no agent)", "0.29"),
                ("Multi-hop EM (with agent)", "0.49  (+20 pts)"),
                ("Hallucinated citations", "0 / 100"),
                ("Out-of-corpus abstentions", "4 / 5"),
                ("Cost per query", "$0.0006  (8x under $0.005 target)"),
            ]
        ),
        PageBreak(),
    ]


def phase7_benchmark() -> list:
    return [
        h1("Phase 7 — Full Benchmark (BEIR)"),
        hr(),
        body(
            "BEIR staircase: SciFact (5K docs) → NFCorpus (3.6K) → ArguAna (8.7K)."
            " Each run embeds the corpus, builds a mini HNSW, and evaluates with pytrec_eval."
        ),
        sp(),
        h2("Run BEIR"),
        code("""PYTHONPATH=src:. uv run python scripts/beir_eval.py
# Downloads each BEIR dataset on first run (~1-2 min each)
# Scores with pytrec_eval
# Output saved to eval/results/beir_DATASET_YYYYMMDD.json"""),
        h2("Results"),
        metric_table(
            [
                ("SciFact  nDCG@10", "0.7253  (BM25 baseline 0.665)"),
                ("SciFact  Recall@10", "0.8529"),
                ("NFCorpus nDCG@10", "0.3311"),
                (
                    "ArguAna  nDCG@10",
                    "0.2826  (counter-argument structure — known failure mode)",
                ),
            ]
        ),
        sp(8),
        h2("Cost tracking (src/rag_engine/cost.py)"),
        code("""# Thread-local cost accumulator — accurate per-request cost
# Input tokens:  $0.15 / 1M  (gpt-4o-mini)
# Output tokens: $0.60 / 1M

class CostTracker:
    _INPUT_COST  = 0.15 / 1_000_000
    _OUTPUT_COST = 0.60 / 1_000_000

    def add_llm(self, input_tokens, output_tokens):
        self._local.cost += (
            input_tokens  * self._INPUT_COST +
            output_tokens * self._OUTPUT_COST
        )"""),
        PageBreak(),
    ]


def phase8_api() -> list:
    return [
        h1("Phase 8 — Production API"),
        hr(),
        body(
            "FastAPI app with Server-Sent Events (SSE) streaming, Redis semantic cache,"
            " sliding-window rate limiting (Lua script), and SHA-256 bearer-token auth."
            " All responses stream in real-time — the client sees tokens as they arrive."
        ),
        sp(),
        h2("Start the API locally"),
        code("""# Start Redis first
docker compose -f infra/docker-compose.yml up -d redis

# Start the API
PYTHONPATH=src uv run uvicorn rag_engine.api.app:app \\
    --host 0.0.0.0 --port 8000 --workers 1

# Health check
curl http://localhost:8000/health
curl http://localhost:8000/ready"""),
        h2("Query endpoint"),
        code("""# SSE streaming query
curl -N \\
  -H "Authorization: Bearer my-dev-token" \\
  -H "Content-Type: application/json" \\
  -d '{"query":"What is the capital of France?","top_k":5,"max_hops":2}' \\
  http://localhost:8000/query

# SSE event stream:
# event: query_id      data: {"id":"uuid..."}
# event: cache_hit     data: {"hit":false}
# event: trace_step    data: {"step":"embed","label":"...","ms":12}
# event: passage       data: {"num":1,"title":"Paris","snippet":"..."}
# event: token         data: {"text":"Paris "}
# event: done          data: {"lat":{"embed":12,"retrieve":18,...},"total_ms":910}"""),
        h2("Semantic cache (src/rag_engine/api/cache.py)"),
        code("""class SemanticCache:
    # cosine similarity >= 0.97 => cache hit, skip entire pipeline (~1ms)

    async def get(self, query):
        q_vec = self._embed(query)          # bge-small encode (once)
        stored = await self._redis.hgetall('cache:embeddings')
        best_sim = max(cosine(q_vec, pickle.loads(v)) for v in stored.values())
        if best_sim >= 0.97:
            return cached_result, q_vec     # return vec so callers skip re-encoding
        return None, q_vec

    async def set(self, query, result, q_vec=None):
        # q_vec passed from get() — no second encode needed"""),
        h2("Rate limiter (src/rag_engine/api/ratelimit.py)"),
        code("""# Sliding window via sorted set — atomic Lua script
# Default: 20 queries per 7 days per tenant
_LUA = \"\"\"
local key    = KEYS[1]
local cap    = tonumber(ARGV[1])
local now    = tonumber(ARGV[2])
local window = tonumber(ARGV[3])
redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
local count = redis.call('ZCARD', key)
if count < cap then
    redis.call('ZADD', key, now, now .. '-' .. math.random(1000000))
    redis.call('EXPIRE', key, window)
    return 1
end
return 0
\"\"\"
async def check(redis, tenant_id, cap=20, window_sec=604800):  # 7 days"""),
        h2("Auth (src/rag_engine/api/auth.py)"),
        code("""# Dual-mode: env-dict (dev) or Postgres (production)
# Token never stored — only SHA-256 hash

async def resolve_tenant(token):
    if _pool is not None:          # Postgres pool connected
        row = await _pool.fetchrow(
            'SELECT tenant_id FROM api_keys WHERE key_hash=$1 AND active=TRUE',
            hashlib.sha256(token.encode()).hexdigest()
        )
        if row: return row['tenant_id']
    return TENANT_MAP.get(token)   # fallback to env dict"""),
        h2("Run tests"),
        code("""uv run pytest tests/test_api.py -v
# 13 tests: auth, rate limit, cache hit, retrieval timeout,
#           LLM timeout, GET /query/{id}, health, ready, cache metrics"""),
        PageBreak(),
    ]


def phase9_observability() -> list:
    return [
        h1("Phase 9 — Observability"),
        hr(),
        body(
            "Prometheus RED metrics (Rate, Error, Duration) + Grafana dashboard with"
            " 8 panels. Three SLO alert rules fire if P95 latency > 3s, error ratio > 5%,"
            " or faithfulness drops more than 2 pts below rolling 7-day average."
        ),
        sp(),
        h2("What Prometheus is"),
        body(
            "Prometheus is an open-source time-series metrics database and alerting system."
            " It scrapes HTTP /metrics endpoints on a configurable interval (15s default)."
            " Each metric type is a counter, gauge, or histogram. The query language is"
            " PromQL."
        ),
        h2("What Grafana is"),
        body(
            "Grafana is a dashboarding tool. It runs queries against Prometheus (or other"
            " data sources) and renders time-series graphs, gauges, and heatmaps."
            " Dashboards are provisioned as JSON files under infra/grafana/."
        ),
        h2("RED method"),
        metric_table(
            [
                ("Rate", "rag_queries_total — requests per second"),
                ("Error", "rag_query_errors_total — errors per stage"),
                ("Duration", "rag_stage_latency_seconds — histogram, per stage"),
            ]
        ),
        sp(8),
        h2("All 6 Prometheus instruments (src/rag_engine/api/metrics.py)"),
        code("""QUERY_TOTAL     = Counter('rag_queries_total',           labels=['tenant','cache_hit'])
QUERY_ERRORS    = Counter('rag_query_errors_total',     labels=['tenant','stage'])
STAGE_LATENCY   = Histogram('rag_stage_latency_seconds',labels=['stage'],
                    buckets=[0.005,0.01,0.025,0.05,0.1,0.25,0.5,1.0,2.5,5.0])
HOP_COUNT       = Histogram('rag_hop_count',            buckets=[1,2,3])
CACHE_HIT_RATE  = Gauge('rag_cache_hit_rate')
FAITHFULNESS_SCORE = Gauge('rag_eval_faithfulness')"""),
        h2("SLO alert rules (infra/alert_rules.yml)"),
        code("""# P95 latency SLO
- alert: HighP95Latency
  expr: histogram_quantile(0.95,
          rate(rag_stage_latency_seconds_bucket[5m])) > 3
  for: 2m

# Error ratio SLO
- alert: HighErrorRate
  expr: rate(rag_query_errors_total[5m]) /
        rate(rag_queries_total[5m]) > 0.05
  for: 2m

# Faithfulness SLO
- alert: FaithfulnessDrift
  expr: rag_eval_faithfulness
        < avg_over_time(rag_eval_faithfulness[7d]) - 0.02
  for: 1h"""),
        h2("Access Grafana"),
        code("""# After docker compose up (Phase 10)
open http://localhost:3000     # admin / admin
# Dashboard: 'RAG Engine' — auto-provisioned from infra/grafana/"""),
        PageBreak(),
    ]


def phase10_deploy() -> list:
    return [
        h1("Phase 10 — Production Deploy"),
        hr(),
        body(
            "Docker multi-stage build (builder + slim runtime), Terraform for Hetzner CX33"
            " provisioning, Postgres for API key storage, and GitHub Actions CI/CD."
        ),
        sp(),
        h2("Dockerfile — two-stage build"),
        code("""# Stage 1: dependency installer (uv)
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev   # deps only
COPY src/ ./src/
RUN uv sync --frozen --no-dev                         # install project

# Stage 2: lean runtime (no uv, no build tools)
FROM python:3.12-slim AS runtime
WORKDIR /app
RUN groupadd --system rag && useradd --system --gid rag --no-create-home rag
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src   /app/src
COPY web/ /app/web/
COPY scripts/ /app/scripts/

ENV PATH="/app/.venv/bin:$PATH" \\
    OMP_NUM_THREADS=1 \\
    TOKENIZERS_PARALLELISM=false

RUN mkdir -p /app/.cache/huggingface /app/data && chown -R rag:rag /app
USER rag
EXPOSE 8000
CMD ["uvicorn", "rag_engine.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]"""),
        h2("Build & push Docker image"),
        code("""docker buildx build \\
  --platform linux/amd64 \\
  --tag ghcr.io/sanjit-jeevanand/rag-engine:latest \\
  --push .

# Then on the server:
ssh root@167.233.55.52 \\
  "cd /opt/rag-engine/infra && docker compose pull && docker compose up -d api" """),
        h2("Terraform — provision Hetzner CX33"),
        code("""# One-time setup
ssh-keygen -t ed25519 -C "rag-engine" -f ~/.ssh/id_ed25519

terraform -chdir=infra/terraform init
terraform -chdir=infra/terraform apply \\
    -var="hcloud_token=$HCLOUD_TOKEN" \\
    -var="ssh_public_key=$(cat ~/.ssh/id_ed25519.pub)"

# Outputs:
#   server_ip   = "167.233.55.52"
#   api_url     = "http://167.233.55.52:8000"
#   grafana_url = "http://167.233.55.52:3000" """),
        h2("Transfer data to server (first deploy)"),
        code("""# Transfer Compose stack
rsync -avz infra/ root@167.233.55.52:/opt/rag-engine/infra/

# Transfer production data (~11 GB)
rsync -avz --progress \\
    data/docs.db data/hnsw.index data/vectors.bin data/bm25_index \\
    root@167.233.55.52:/opt/rag-engine/data/

# Create .env on server
ssh root@167.233.55.52 "cat > /opt/rag-engine/.env" << 'EOF'
OPENAI_API_KEY=sk-...
RAG_DB_URL=postgresql://rag:rag@postgres:5432/rag
EOF"""),
        h2("Start all 5 services"),
        code("""ssh root@167.233.55.52 \\
  "cd /opt/rag-engine/infra && docker compose up -d"

# Services:
#   api        → :8000  (FastAPI + uvicorn)
#   redis      → :6379  (semantic cache + rate limit)
#   postgres   → :5432  (api_keys table)
#   prometheus → :9090  (metrics)
#   grafana    → :3000  (dashboards)"""),
        h2("Postgres API keys — generate tokens"),
        code("""# Run from inside server (Postgres not exposed externally)
ssh root@167.233.55.52 bash -s << 'EOF'
docker compose -f /opt/rag-engine/infra/docker-compose.yml exec -T api \\
    python scripts/gen_api_keys.py \\
    --db-url postgresql://rag:rag@postgres:5432/rag \\
    --count 10 --prefix demo
EOF

# Revoke a key
docker exec <postgres-container> psql -U rag rag \\
  -c "UPDATE api_keys SET active=FALSE WHERE tenant_id='demo-03';" """),
        h2("Postgres schema (infra/postgres/init.sql)"),
        code("""CREATE TABLE IF NOT EXISTS api_keys (
    key_hash   TEXT        PRIMARY KEY,   -- SHA-256(raw_token), hex
    tenant_id  TEXT        NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    active     BOOLEAN     NOT NULL DEFAULT TRUE
);"""),
        h2("GitHub Actions CI/CD"),
        code("""# .github/workflows/deploy.yml  (trigger: manual or push to main)
# Steps:
#   1. docker buildx build --platform linux/amd64 --push ghcr.io/...
#   2. ssh into server: docker compose pull && docker compose up -d api

# Required repository secrets:
#   GHCR_PAT         GitHub classic token (write:packages)
#   HETZNER_HOST     Server IP
#   SSH_PRIVATE_KEY  Contents of ~/.ssh/id_ed25519

# Trigger manually:
#   Actions → Deploy → Run workflow"""),
        PageBreak(),
    ]


def end_to_end_checklist() -> list:
    return [
        h1("End-to-End Reproduction Checklist"),
        hr(),
        body(
            "Run these commands in order on a clean machine with Python 3.12, uv,"
            " and Docker installed. Estimated wall time: ~10 hours (mostly embedding)."
        ),
        sp(),
        h2("Setup (5 min)"),
        code("""git clone https://github.com/Sanjit-Jeevanand/rag-engine && cd rag-engine
uv sync
cp .env.example .env && $EDITOR .env    # fill in OPENAI_API_KEY"""),
        h2("Phase 1 — Corpus Ingestion (~8 hours on CPU)"),
        code("""uv run python scripts/download_wiki_dump.py         # ~10 GB download
uv run python scripts/run_pipeline.py \\             # parse + chunk -> docs.db
    --snapshot data/wiki-dump.json.gz --db data/docs.db
PYTHONPATH=src uv run python scripts/run_embedder.py \\  # embed -> vectors.bin
    --db data/docs.db --out data/vectors.bin --batch 256"""),
        h2("Phase 2 — Eval Harness (10 min)"),
        code("""uv run python scripts/seed_gold_set.py
PYTHONPATH=src:. uv run python scripts/beir_eval.py"""),
        h2("Phase 3 — Build Indexes (20 min)"),
        code("""# Full 1M benchmark index
uv run python scripts/build_hnsw_index.py \\
    --vectors data/vectors.bin --db data/docs.db \\
    --out data/hnsw_1m.index --m 32 --ef-construction 200 --ef-search 64

# Production 100K index
uv run python scripts/build_production_index.py --limit 100000
uv run python scripts/build_100k_dataset.py     # installs as production files"""),
        h2("Phase 5 — Hybrid Retrieval (auto, no extra step)"),
        note("BM25 index is built during run_pipeline.py and loaded at API startup."),
        h2("Phase 6 — Agentic eval (5 min)"),
        code("""PYTHONPATH=src:. uv run --env-file .env \\
    python scripts/hotpotqa_agentic_eval.py --n 100"""),
        h2("Phase 8 — Run API locally (instant)"),
        code("""docker compose -f infra/docker-compose.yml up -d redis postgres
PYTHONPATH=src uv run uvicorn rag_engine.api.app:app --port 8000
# Test:
curl -N -H "Authorization: Bearer my-dev-token" \\
     -H "Content-Type: application/json" \\
     -d '{"query":"What is the capital of France?","top_k":5,"max_hops":2}' \\
     http://localhost:8000/query"""),
        h2("Phase 10 — Deploy to Hetzner (15 min)"),
        code("""# 1. Provision server
terraform -chdir=infra/terraform apply \\
    -var="hcloud_token=$HCLOUD_TOKEN" \\
    -var="ssh_public_key=$(cat ~/.ssh/id_ed25519.pub)"

# 2. Build and push Docker image (linux/amd64)
docker buildx build --platform linux/amd64 \\
    --tag ghcr.io/sanjit-jeevanand/rag-engine:latest --push .

# 3. Transfer data
rsync -avz infra/ root@SERVER_IP:/opt/rag-engine/infra/
rsync -avz data/docs.db data/hnsw.index data/vectors.bin data/bm25_index \\
    root@SERVER_IP:/opt/rag-engine/data/

# 4. Start stack
ssh root@SERVER_IP \\
    "cd /opt/rag-engine/infra && docker compose up -d"

# 5. Verify
curl http://SERVER_IP:8000/ready      # should return {"status":"ready"}
open http://SERVER_IP:8000/ui
open http://SERVER_IP:3000            # Grafana"""),
        h2("CI validation"),
        code("""make ci   # lint + typecheck + test + audit + eval-gate"""),
        PageBreak(),
    ]


def troubleshooting() -> list:
    return [
        h1("Troubleshooting"),
        hr(),
        h2("Double-encode latency (13s retrieval)"),
        body(
            "Symptom: retrieval takes 13+ seconds. Root cause: cache.get() computes"
            " q_vec but the caller re-encodes with embedder.encode() a second time."
            " Fix: cache.get() returns (result, q_vec) as a tuple; the caller passes"
            " q_vec directly to _dense() which calls qvec.reshape(1,-1) for FAISS."
        ),
        code("""# WRONG (double encode):
cached = await cache.get(query)       # computes q_vec internally, discards it
passages = await _retrieve(query, k)  # encodes again -> +13s

# CORRECT:
cached, q_vec = await cache.get(query)  # keep the vector
passages = await _retrieve(query, k, q_vec)  # pass through, skip second encode"""),
        h2("FAISS HNSW silent ef overriding"),
        body(
            "FAISS enforces efSearch = max(efSearch, k). If k=1000 and efSearch=64,"
            " efSearch silently becomes 1000 — 15x slower. Always set _DENSE_K = efSearch."
        ),
        code("""_DENSE_K = 64  # must equal hnsw_ef_search in config.py
# In _dense(): hnsw.search(vec, _DENSE_K)  -- NOT hnsw.search(vec, top_k)"""),
        h2("Docker image 6 GB (NVIDIA CUDA)"),
        body(
            "PyPI torch includes CUDA libraries even on CPU-only machines,"
            " bloating the image to 6 GB and causing disk-full on Hetzner CX33."
            " Fix: use pytorch-cpu index in pyproject.toml."
        ),
        code("""[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cpu" }    # pulls torch+cpu ~600 MB instead of 4 GB

# Then regenerate lock:
uv lock --upgrade-package torch"""),
        h2("FAISS 1D vs 2D shape error"),
        code("""# cache._embed() returns shape (384,) — FAISS needs (1, 384)
# Fix in _dense():
vec = qvec.reshape(1, -1) if qvec is not None else embedder.encode(...)"""),
        h2("JS syntax error in web/index.html"),
        body(
            "Curly/smart quotes (U+201C U+201D) used as JS string delimiters break"
            " the entire page. Check with: node --check web/index.html."
            " Fix: replace all \\xe2\\x80\\x9c/\\x9d with ASCII 0x22."
        ),
        code("""python3 -c "
with open('web/index.html','rb') as f: c = f.read()
c = c.replace(b'\\xe2\\x80\\x9c', b'\"').replace(b'\\xe2\\x80\\x9d', b'\"')
with open('web/index.html','wb') as f: f.write(c)
"
node --check web/index.html   # should print nothing on success"""),
        PageBreak(),
    ]


def project_structure() -> list:
    return [
        h1("Project Structure"),
        hr(),
        code("""rag-engine/
├── src/rag_engine/
│   ├── api/
│   │   ├── app.py        FastAPI app — SSE generator, lifespan, _retrieve_sync
│   │   ├── auth.py       SHA-256 hashing, asyncpg pool, env-dict fallback
│   │   ├── cache.py      Redis semantic cache (cosine >= 0.97)
│   │   ├── metrics.py    6 Prometheus instruments
│   │   ├── models.py     Pydantic models: QueryRequest, Citation, QueryResult
│   │   ├── ratelimit.py  Lua sliding-window rate limiter
│   │   └── stream.py     SSE event formatters
│   ├── agent/
│   │   ├── loop.py       MultiHopAgent: bridge -> cite -> reflect -> abstain
│   │   └── llm.py        OpenAI sync/async client wrappers
│   ├── ingest/
│   │   ├── schema.py     SQLite schema, WAL mode
│   │   ├── pipeline.py   Article chunking (1500 chars, 200 overlap)
│   │   ├── parser.py     Wikipedia JSONL -> WikiArticle iterator
│   │   └── embedder.py   bge-small-en-v1.5, writes vectors.bin memmap
│   ├── retrieval/
│   │   ├── bm25.py       BM25Retriever (bm25s)
│   │   ├── dense.py      DenseRetriever (FAISS IndexFlatIP — for evals)
│   │   ├── hybrid.py     Reciprocal Rank Fusion
│   │   └── reranker.py   CrossEncoderReranker (bge-reranker-base)
│   ├── config.py         Pydantic-settings, RAG_ env prefix
│   ├── cost.py           Thread-local cost tracker
│   └── log.py            structlog JSON, request_id ContextVar
├── scripts/
│   ├── build_production_index.py  Top-100K HNSW from memmap
│   ├── build_100k_dataset.py      Rename + rebuild aligned docs
│   ├── hotpotqa_agentic_eval.py   EM comparison single-shot vs multi-hop
│   ├── beir_eval.py               BEIR staircase (SciFact, NFCorpus, ArguAna)
│   └── perf_check.py              HNSW throughput regression gate
├── infra/
│   ├── docker-compose.yml         5 services: api, redis, postgres, prom, grafana
│   ├── postgres/init.sql          api_keys table
│   ├── prometheus-stack.yml       Scrape config
│   ├── alert_rules.yml            3 SLO alerts
│   └── terraform/                 Hetzner CX33 IaC
├── eval/
│   ├── metrics.py                 nDCG, Recall, MRR, EM, F1
│   ├── comparator.py              Regression gate (+/-2% tolerance)
│   └── results/
│       ├── baseline.json          nDCG=0.4618, Recall=0.478, MRR=0.5994
│       └── latest.json
├── tests/
│   └── test_api.py                13 API tests
├── web/
│   └── index.html                 Single-page demo UI
├── Dockerfile
├── Makefile
└── pyproject.toml"""),
    ]


# ── Build PDF ──────────────────────────────────────────────────────────────────


def build() -> None:
    out = "/Users/sanjitjeevanand/Desktop/rag_engine_reproduction_guide.pdf"
    doc = SimpleDocTemplate(
        out,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=2.2 * cm,
        bottomMargin=1.8 * cm,
    )

    story: list = []
    story += cover_page()
    story += prerequisites()
    story += phase0_scaffolding()
    story += phase1_ingest()
    story += phase2_eval()
    story += phase3_hnsw()
    story += phase4_throughput()
    story += phase5_hybrid()
    story += phase6_agent()
    story += phase7_benchmark()
    story += phase8_api()
    story += phase9_observability()
    story += phase10_deploy()
    story += end_to_end_checklist()
    story += troubleshooting()
    story += project_structure()

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(f"PDF written to {out}")


if __name__ == "__main__":
    build()

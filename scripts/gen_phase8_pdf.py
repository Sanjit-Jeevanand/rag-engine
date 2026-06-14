"""Generate Phase 8 explainer PDF."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

W, H = letter
MARGIN = 0.85 * inch

# ── colours ──────────────────────────────────────────────────────────────────
NAVY = colors.HexColor("#1a2a4a")
TEAL = colors.HexColor("#0d7377")
LIGHT = colors.HexColor("#f0f4f8")
GOLD = colors.HexColor("#c9a84c")
MID = colors.HexColor("#4a5568")
CODE_BG = colors.HexColor("#1e1e2e")
CODE_FG = colors.HexColor("#cdd6f4")

# ── styles ────────────────────────────────────────────────────────────────────
base = getSampleStyleSheet()


def S(name, **kw):
    return ParagraphStyle(name, parent=base["Normal"], **kw)


sTitle = S(
    "sTitle",
    fontSize=28,
    textColor=colors.white,
    leading=34,
    fontName="Helvetica-Bold",
    spaceAfter=4,
)
sSubtitle = S(
    "sSubtitle",
    fontSize=13,
    textColor=GOLD,
    leading=18,
    fontName="Helvetica",
    spaceAfter=0,
)
sH1 = S(
    "sH1",
    fontSize=17,
    textColor=NAVY,
    leading=22,
    fontName="Helvetica-Bold",
    spaceBefore=18,
    spaceAfter=6,
)
sH2 = S(
    "sH2",
    fontSize=13,
    textColor=TEAL,
    leading=17,
    fontName="Helvetica-Bold",
    spaceBefore=12,
    spaceAfter=4,
)
sH3 = S(
    "sH3",
    fontSize=11,
    textColor=NAVY,
    leading=15,
    fontName="Helvetica-Bold",
    spaceBefore=8,
    spaceAfter=3,
)
sBody = S(
    "sBody", fontSize=10, textColor=MID, leading=15, fontName="Helvetica", spaceAfter=6
)
sCode = S(
    "sCode",
    fontSize=8.5,
    textColor=CODE_FG,
    leading=13,
    fontName="Courier",
    backColor=CODE_BG,
    leftIndent=10,
    rightIndent=10,
    spaceBefore=4,
    spaceAfter=4,
    borderPad=6,
)
sNote = S(
    "sNote",
    fontSize=9,
    textColor=colors.HexColor("#6b7280"),
    leading=13,
    fontName="Helvetica-Oblique",
    spaceAfter=4,
)
sBullet = S(
    "sBullet",
    fontSize=10,
    textColor=MID,
    leading=15,
    fontName="Helvetica",
    leftIndent=16,
    spaceAfter=3,
)
sLabel = S("sLabel", fontSize=9, textColor=NAVY, leading=12, fontName="Helvetica-Bold")
sLabelVal = S("sLabelVal", fontSize=9, textColor=MID, leading=12, fontName="Helvetica")


def hr():
    return HRFlowable(
        width="100%", thickness=1, color=colors.HexColor("#e2e8f0"), spaceAfter=8
    )


def sp(n=6):
    return Spacer(1, n)


def body(text):
    return Paragraph(text, sBody)


def code(text):
    escaped = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
        .replace("  ", "&nbsp;&nbsp;")
        .replace("\t", "&nbsp;&nbsp;&nbsp;&nbsp;")
    )
    return Paragraph(escaped, sCode)


def h1(text):
    return Paragraph(text, sH1)


def h2(text):
    return Paragraph(text, sH2)


def h3(text):
    return Paragraph(text, sH3)


def bullets(items):
    return ListFlowable(
        [
            ListItem(Paragraph(t, sBullet), bulletColor=TEAL, leftIndent=20)
            for t in items
        ],
        bulletType="bullet",
        leftIndent=16,
        spaceAfter=4,
    )


def kv_table(rows):
    """Two-column key-value table."""
    data = [[Paragraph(k, sLabel), Paragraph(v, sLabelVal)] for k, v in rows]
    t = Table(data, colWidths=[1.8 * inch, 5.2 * inch])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), LIGHT),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                (
                    "ROWBACKGROUNDS",
                    (0, 0),
                    (-1, -1),
                    [colors.white, colors.HexColor("#f8fafc")],
                ),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("ROUNDEDCORNERS", [4]),
            ]
        )
    )
    return t


def files_table(rows):
    """Two-column file list."""
    header = [Paragraph(h, sLabel) for h in ["File", "What it does"]]
    data = [header] + [
        [
            Paragraph(f'<font name="Courier" size="8">{p}</font>', sLabelVal),
            Paragraph(d, sLabelVal),
        ]
        for p, d in rows
    ]
    t = Table(data, colWidths=[2.8 * inch, 4.2 * inch])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor("#f8fafc")],
                ),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return t


def metrics_table(rows):
    header = [Paragraph(h, sLabel) for h in ["Metric", "Value"]]
    data = [header] + [
        [Paragraph(m, sLabelVal), Paragraph(v, sLabelVal)] for m, v in rows
    ]
    t = Table(data, colWidths=[3.5 * inch, 3.5 * inch])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), TEAL),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor("#f8fafc")],
                ),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return t


# ── cover page ────────────────────────────────────────────────────────────────
def cover_page():
    cover = Table(
        [
            [Paragraph("Phase 8", sTitle)],
            [Paragraph("Production Serving", sTitle)],
            [sp(4)],
            [
                Paragraph(
                    "Streaming API · Auth · Semantic Caching · Rate Limiting"
                    " · Hybrid Retrieval · Multi-hop Agent",
                    sSubtitle,
                )
            ],
        ],
        colWidths=[W - 2 * MARGIN],
    )
    cover.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), NAVY),
                ("LEFTPADDING", (0, 0), (-1, -1), 24),
                ("RIGHTPADDING", (0, 0), (-1, -1), 24),
                ("TOPPADDING", (0, 0), (-1, -1), 28),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 28),
                ("ROUNDEDCORNERS", [8]),
            ]
        )
    )
    return cover


# ── build story ───────────────────────────────────────────────────────────────
story = []

# Cover
story += [cover_page(), sp(20)]

story += [
    body(
        "Phase 8 turns the RAG pipeline built across Phases 0-7 into a real production "
        "API. Users get streaming token-by-token responses, similar queries hit a semantic "
        "cache, each tenant has its own rate limit, retrieval runs as a full hybrid pipeline "
        "(HNSW + BM25 + RRF + cross-encoder reranking), and a multi-hop agent loop can "
        "chain two retrieval passes to answer bridge questions."
    ),
    sp(4),
]

# ── 1. What problem does Phase 8 solve? ───────────────────────────────────────
story += [h1("1. What Problem Does Phase 8 Solve?"), hr()]
story += [
    body(
        "Phases 0-7 gave you a pipeline that works — but only when called directly in Python. "
        "Phase 8 exposes it over HTTP and wires the frontend to the real backend."
    ),
    body("Four real problems come up the moment you do that:"),
    bullets(
        [
            "<b>Speed perception.</b> An LLM takes 2-5 seconds to generate an answer. "
            "Streaming tokens as they arrive (like ChatGPT) makes the experience feel instant "
            "even though the full response takes 3 seconds.",
            "<b>Cost.</b> Every LLM call costs money. A semantic cache short-circuits repeated "
            "or paraphrased questions — the second ask is answered in ~22 ms for free.",
            "<b>Fairness and safety.</b> Without rate limiting one tenant can flood the server. "
            "Without auth anyone can run up your bill.",
            "<b>Multi-hop questions.</b> Single-shot retrieval fails on bridge questions "
            "(e.g. 'capital of the country where Eiffel Tower stands'). "
            "A two-hop agent loop extracts the bridge entity and runs a second retrieval.",
        ]
    ),
    sp(4),
]

# ── 2. The wire protocol ───────────────────────────────────────────────────────
story += [h1("2. Wire Protocol — Server-Sent Events"), hr()]
story += [
    body(
        "The frontend and backend communicate over a single HTTP connection using "
        "<b>Server-Sent Events (SSE)</b>. The key requirement is the named-event format: "
        "each message must have an <font name='Courier'>event:</font> line so the browser "
        "can dispatch it to the right handler."
    ),
    body("Every SSE frame looks like this:"),
    code(
        "event: token\n"
        'data: {"text": "The"}\n'
        "\n"
        "event: token\n"
        'data: {"text": " capital"}\n'
        "\n"
        "event: done\n"
        'data: {"query_id": "abc-123", "mode": "grounded", "total_ms": 1875}\n'
        "\n"
    ),
    body(
        "Without the <font name='Courier'>event:</font> line the browser's EventSource "
        "receives every message as an unnamed 'message' event — the frontend can't "
        "distinguish a token from a done signal. This was the critical fix when wiring "
        "the backend to the frontend: switching from <font name='Courier'>sse_starlette</font>'s "
        "EventSourceResponse (which double-wraps) to a plain "
        "<font name='Courier'>StreamingResponse</font> with "
        "<font name='Courier'>media_type='text/event-stream'</font> that passes "
        "pre-formatted strings through as-is."
    ),
    sp(4),
]

story += [h2("Full Event Sequence")]
story += [
    body("For a non-cached query, the backend emits these events in order:"),
    sp(4),
]
story.append(
    kv_table(
        [
            (
                "query_id",
                '{"id": "uuid4"} — emitted first so the frontend can poll GET /query/{id}',
            ),
            (
                "cache_hit",
                '{"hit": false} — or true with cached answer following immediately',
            ),
            (
                "trace_step",
                '{"step": "embed", "label": "Embed query", "ms": 16} — one per pipeline stage',
            ),
            (
                "trace_step",
                '{"step": "retrieve", "label": "Hop 1 — retrieve", "ms": 370}',
            ),
            (
                "passage",
                '{"num": 1, "title": "...", "dense": 0.99, "bm25": 0.82, "rerank": 0.97, "hop": 1}',
            ),
            (
                "trace_step",
                '{"step": "hop1", "label": "Bridge → Gustave Eiffel", "reflect": true}',
            ),
            ("trace_step", '{"step": "hop2", "label": "Hop 2 — retrieve", "ms": 290}'),
            (
                "passage",
                '{"num": 6, "title": "...", "hop": 2} — additional passages from hop 2',
            ),
            ("generation_start", "{} — signals LLM is about to stream"),
            ("token", '{"text": "The"} — one per LLM token'),
            (
                "trace_step",
                '{"step": "generate", "label": "Generate answer", "ms": 2919}',
            ),
            (
                "done",
                '{"mode": "grounded", "lat": {"embed":16,"retrieve":370,...}, "total_ms": 3308}',
            ),
        ]
    )
)
story.append(sp(8))

# ── 3. Hybrid Retrieval Pipeline ───────────────────────────────────────────────
story += [PageBreak(), h1("3. Hybrid Retrieval Pipeline"), hr()]
story += [
    body(
        "Retrieval runs two independent searches in parallel, then fuses and reranks:"
    ),
    sp(4),
]
story.append(
    kv_table(
        [
            (
                "HNSW dense search",
                "Encodes the query with bge-small-en-v1.5, searches 1,000 nearest neighbours "
                "in the FAISS HNSW index (15.9 GB, 8.8M chunks). Returns up to 100 unique "
                "article titles by cosine similarity.",
            ),
            (
                "BM25 sparse search",
                "Runs the query through the bm25s index. Returns up to 100 titles ranked by "
                "TF-IDF term overlap. Runs in parallel with HNSW via ThreadPoolExecutor.",
            ),
            (
                "Reciprocal Rank Fusion",
                "Merges the two ranked lists with RRF (k=60). Score = sum(1/(60+rank+1)) "
                "across all lists. Top 10 candidates pass to the reranker.",
            ),
            (
                "Cross-encoder rerank",
                "bge-reranker-base scores each (query, passage) pair with BERT-style "
                "attention. Final order is by cross-encoder score, not embedding similarity.",
            ),
        ]
    )
)
story.append(sp(8))

story += [h2("Score Breakdown in the UI")]
story += [
    body("Each retrieved passage shows three scores. All are on a 0-1 scale:"),
    bullets(
        [
            "<b>DENSE</b> — raw cosine similarity from HNSW search "
            "(captured from faiss.search() distances). High = semantically close to the query.",
            "<b>BM25</b> — normalized RRF rank contribution from the sparse list. "
            "Rank 0 = 1.0, rank 99 = ~0.38, not retrieved = 0.0. "
            "A passage with BM25=0 was found only by dense search.",
            "<b>RERANK</b> — cross-encoder score. This is what determines final order. "
            "A passage can have low DENSE but high RERANK if the cross-encoder finds it "
            "highly relevant in context.",
        ]
    ),
    body("The Citation model carries all three:"),
    code(
        "class Citation(BaseModel):\n"
        "    passage_id: str\n"
        "    title: str\n"
        "    text: str\n"
        "    score: float        # cross-encoder rerank score\n"
        "    dense_score: float  # HNSW cosine similarity\n"
        "    bm25_score: float   # normalized RRF BM25 contribution\n"
    ),
    sp(6),
]

# ── 4. Multi-hop Agent Loop ────────────────────────────────────────────────────
story += [h1("4. Multi-hop Agent Loop"), hr()]
story += [
    body(
        "Single-shot retrieval fails on bridge questions — questions where the answer "
        "to step 1 is required as input to step 2. Example:"
    ),
    bullets(
        [
            "Q: 'What is the birth city of the engineer who designed the tower on the Champ de Mars?'",
            "Step 1: retrieve 'Champ de Mars' → top passage: Eiffel Tower → mentions Gustave Eiffel",
            "Step 2: retrieve 'Gustave Eiffel' → passage mentions born in Dijon",
            "Without step 2, the LLM hallucinates from whatever civil engineers happen to be in the top-5",
        ]
    ),
    sp(4),
]

story += [h2("Bridge Extraction")]
story += [
    body(
        "After hop 1, a fast LLM call extracts the bridge entity from the top-ranked passage:"
    ),
    code(
        "Question: {original_query}\n"
        "Passage title: {top_title}\n"
        "Passage excerpt: {top_text[:400]}\n\n"
        "What single entity or concept must be looked up next to answer the question?\n"
        "Reply with ONLY the search term, or NONE if the passage already answers it."
    ),
    body(
        "The bridge is emitted as a <font name='Courier'>trace_step</font> event with "
        "<font name='Courier'>reflect=true</font> (shown with a ↺ icon in the agent trace). "
        "A second retrieval runs with the bridge as the query. "
        "New passages (not already in hop 1 results) are emitted with "
        "<font name='Courier'>hop=2</font> and get a 'HOP 2' badge in the UI."
    ),
    sp(4),
]

story += [h2("Merged Reranking")]
story += [
    body(
        "After both hops, all passages are merged and re-scored against the "
        "<b>original query</b> using the cross-encoder. This ensures the final "
        "passage order is relevant to the original intent, not the bridge query. "
        "The LLM generates its answer from the merged, re-ranked set."
    ),
    body("The agent trace in the UI shows all steps in real time:"),
    code(
        "✓  Embed query          bge-small-en-v1.5 · 16 ms\n"
        "✓  Hop 1 — retrieve     hybrid · top-5    · 370 ms\n"
        '↺  Bridge → Gustave Eiffel   from "Eiffel Tower"\n'
        "✓  Hop 2 — retrieve     query: Gustave Eiffel · 290 ms\n"
        "✓  Generate answer      gpt-4o-mini       · 2400 ms\n"
    ),
    sp(6),
]

# ── 5. Architecture ────────────────────────────────────────────────────────────
story += [PageBreak(), h1("5. Architecture — Request Flow"), hr()]
story += [
    body("Here is what happens for every POST /query request, in order:"),
    sp(4),
]

steps_arch = [
    (
        "Auth middleware",
        "Reads Authorization header → resolves tenant_id → binds to log context. "
        "Returns 401 if missing or unknown.",
    ),
    (
        "Rate limit check",
        "Calls the Lua script with the tenant's Redis key. If over limit → return 429 "
        "with Retry-After header.",
    ),
    (
        "Semantic cache lookup",
        "Embeds the query → scans cached embeddings → if cosine sim > 0.97, stream "
        "the cached answer as SSE tokens. Done — no retrieval, no LLM call.",
    ),
    (
        "Hop 1 retrieval (30s timeout)",
        "HNSW + BM25 in parallel → RRF fusion → cross-encoder rerank. "
        "Emit trace_step and passage events.",
    ),
    (
        "Bridge extraction (8s timeout)",
        "LLM call extracts bridge entity from top passage. "
        "Emit reflect trace_step. Only runs when max_hops > 1.",
    ),
    (
        "Hop 2 retrieval (30s timeout)",
        "Retrieve with bridge entity → merge with hop-1 results → rerank merged set "
        "against original query. Emit hop-2 passage events.",
    ),
    (
        "LLM generation (5s per-token timeout)",
        "Stream tokens via OpenAI streaming API. Emit token events. "
        "If LLM takes > 5s → emit generation_unavailable in done event.",
    ),
    (
        "Store result",
        "Write full result (answer + citations) to Redis keyed by query_id. "
        "Also store in semantic cache. Emit done event.",
    ),
    (
        "GET /query/{id}",
        "Separate endpoint. Fetches stored result from Redis — returns full answer "
        "with citations as regular JSON (no streaming).",
    ),
]
story.append(kv_table([(f"{i + 1}. {k}", v) for i, (k, v) in enumerate(steps_arch)]))
story.append(sp(8))

# ── 6. Files ───────────────────────────────────────────────────────────────────
story += [h1("6. Files"), hr()]
story += [
    body("All API code lives under src/rag_engine/api/. Each file has one job."),
    sp(4),
]
story.append(
    files_table(
        [
            (
                "src/rag_engine/api/models.py",
                "Pydantic shapes: QueryRequest (query, top_k, max_hops), "
                "Citation (passage_id, title, text, score, dense_score, bm25_score), "
                "QueryResult (answer, citations, cache_hit, generation_unavailable flags)",
            ),
            (
                "src/rag_engine/api/auth.py",
                "Middleware: reads Authorization: Bearer <token> → resolves tenant_id "
                "→ binds to log context → 401 on unknown token",
            ),
            (
                "src/rag_engine/api/ratelimit.py",
                "Async function wrapping an atomic Redis Lua script: "
                "check(redis, tenant_id) → bool. Returns False when over limit.",
            ),
            (
                "src/rag_engine/api/stream.py",
                "Pure functions that emit properly-formatted SSE frames: "
                "query_id_event, cache_hit_event, trace_step_event, passage_event (with optional hop=), "
                "generation_start_event, token_event, done_event, error_event. "
                "All use event: <name>\\ndata: {json}\\n\\n format.",
            ),
            (
                "src/rag_engine/api/cache.py",
                "SemanticCache: get(query) → QueryResult | None, set(query, result). "
                "Embeds query, scans stored embeddings for cosine sim > 0.97, "
                "maintains hit/total counters in Redis.",
            ),
            (
                "src/rag_engine/api/app.py",
                "FastAPI app. Lifespan loads HNSW index, BM25, cross-encoder, DB mappings. "
                "_retrieve_sync runs HNSW+BM25 in parallel threads, fuses with RRF, reranks. "
                "_extract_bridge calls LLM to get hop-2 query. "
                "_rerank_sync re-scores merged passage lists. "
                "_sse_generator orchestrates the full multi-hop pipeline and yields SSE frames.",
            ),
            (
                "scripts/run_server.py",
                "Starts uvicorn. Sets OMP_NUM_THREADS=1 and TOKENIZERS_PARALLELISM=false "
                "to prevent OpenMP deadlock on Apple Silicon when PyTorch runs in multiple threads.",
            ),
            (
                "tests/test_api.py",
                "Pytest tests with mocked Redis and mocked retrieval. "
                "Covers: 401 missing/invalid token, 429 rate limit, cache hit SSE, "
                "retrieval timeout → generation_unavailable, LLM timeout → generation_unavailable, "
                "GET /query/{id} found and 404, /health, /ready, /metrics/cache.",
            ),
            (
                "web/index.html",
                "Single-file frontend. Connects to backend SSE stream, renders agent trace "
                "(embed → hop1 → bridge → hop2 → generate), passage cards with "
                "dense/BM25/rerank score bars, latency budget waterfall, wire protocol log.",
            ),
        ]
    )
)
story.append(sp(8))

# ── 7. Key concepts ────────────────────────────────────────────────────────────
story += [PageBreak(), h1("7. Key Concepts"), hr()]

story += [h2("StreamingResponse vs EventSourceResponse")]
story += [
    body(
        "<font name='Courier'>sse_starlette</font>'s EventSourceResponse prepends "
        "<font name='Courier'>data:</font> to every yielded string. If your generator "
        "already yields formatted SSE frames, you get double-wrapping: "
        "<font name='Courier'>data: event: token\\ndata: data: {...}</font>. "
        "The fix: use Starlette's plain "
        "<font name='Courier'>StreamingResponse(generator, media_type='text/event-stream')</font> "
        "and yield complete, correctly-formatted SSE strings."
    ),
    sp(6),
]

story += [h2("asyncio.to_thread for CPU-bound work")]
story += [
    body(
        "FAISS search, BM25 retrieval, and cross-encoder inference are all CPU-bound. "
        "Running them in the async event loop would block all other requests. "
        "<font name='Courier'>asyncio.to_thread(fn, *args)</font> runs them in a "
        "thread-pool executor so the event loop stays free, and "
        "<font name='Courier'>asyncio.wait_for()</font> can still fire a timeout."
    ),
    sp(6),
]

story += [h2("OMP_NUM_THREADS=1 on Apple Silicon")]
story += [
    body(
        "PyTorch ships its own OpenMP runtime (libomp.dylib). When two threads call "
        "PyTorch simultaneously — e.g. the cache embedding on the main thread and "
        "retrieval embedding in a worker thread — their OMP runtimes deadlock with a SIGSEGV. "
        "Setting OMP_NUM_THREADS=1 and TOKENIZERS_PARALLELISM=false before importing "
        "torch (in run_server.py) prevents this."
    ),
    sp(6),
]

story += [h2("Reciprocal Rank Fusion")]
story += [
    body(
        "RRF combines two ranked lists without needing to normalize their scores. "
        "For each document, add 1/(k + rank + 1) across all lists that contain it "
        "(k=60 dampens the effect of very high ranks). Documents appearing in both "
        "lists get a double contribution; documents in only one list get a single contribution. "
        "Sort by total score descending."
    ),
    code(
        "def reciprocal_rank_fusion(ranked_lists, k=60, top_k=10):\n"
        "    scores = defaultdict(float)\n"
        "    for ranked in ranked_lists:\n"
        "        for rank, doc_id in enumerate(ranked):\n"
        "            scores[doc_id] += 1.0 / (k + rank + 1)\n"
        "    return sorted(scores, key=lambda d: scores[d], reverse=True)[:top_k]\n"
    ),
    sp(6),
]

story += [h2("CORS Middleware")]
story += [
    body(
        "The frontend is served on localhost:3000 and the backend on localhost:8000 — "
        "different origins. Browsers block cross-origin requests by default. "
        "Adding CORSMiddleware to FastAPI adds the necessary "
        "<font name='Courier'>Access-Control-Allow-Origin: *</font> header "
        "to every response."
    ),
    sp(4),
]

# ── 8. Target metrics ─────────────────────────────────────────────────────────
story += [h1("8. Observed Metrics"), hr()]
story.append(
    metrics_table(
        [
            ("P95 latency budget (full pipeline)", "3000 ms"),
            ("Embed step", "~16 ms (bge-small-en-v1.5 on CPU)"),
            ("Retrieval (HNSW + BM25 parallel + rerank)", "305-460 ms"),
            ("HNSW ANNS recall@10", "98.6% (ef=64)"),
            ("HNSW search P50", "0.387 ms (index lookup only)"),
            ("LLM generation", "1400-3000 ms (gpt-4o-mini streaming)"),
            ("Semantic cache hit latency", "~22 ms (embed only, no retrieval/LLM)"),
            ("Retrieval timeout", "30 s"),
            ("LLM per-token timeout", "5 s"),
        ]
    )
)
story.append(sp(8))

# ── 9. Common mistakes ────────────────────────────────────────────────────────
story += [h1("9. Common Mistakes"), hr()]
story.append(
    bullets(
        [
            "<b>Missing event: line in SSE.</b> Without the named-event line the browser "
            "dispatches everything as an unnamed 'message' event — your type-specific "
            "handlers never fire.",
            "<b>Using EventSourceResponse from sse_starlette.</b> It prepends data: to "
            "every yielded string, double-wrapping your pre-formatted frames.",
            "<b>Running CPU-bound retrieval in the async event loop.</b> Blocks all requests. "
            "Use asyncio.to_thread() for FAISS, BM25, and cross-encoder calls.",
            "<b>Exact string keys for the semantic cache.</b> Natural language queries almost "
            "never repeat exactly. You need cosine similarity >= 0.97 on embeddings.",
            "<b>Setting the cosine threshold too low (e.g. 0.85).</b> 'What caused WWI?' and "
            "'What ended WWI?' might both hit the same cache entry. Start at 0.97.",
            "<b>Not adding CORS middleware.</b> Frontend (port 3000) and backend (port 8000) "
            "are different origins — browsers block the fetch without Access-Control headers.",
            "<b>Hardcoding the retrieval timeout too tight.</b> 200 ms works for HNSW-only "
            "search but not for the full hybrid + rerank pipeline. Use 30 s.",
            "<b>Loading the HNSW index per request.</b> A 15.9 GB index takes 30+ seconds to "
            "load. Load once in the FastAPI lifespan context manager.",
        ]
    )
)
story.append(sp(8))

# ── 10. Dependencies ────────────────────────────────────────────────────────────
story += [h1("10. Dependencies Added in Phase 8"), hr()]
story.append(
    kv_table(
        [
            (
                "fastapi",
                "Web framework. Routing, Pydantic validation, auto-docs at /docs.",
            ),
            (
                "uvicorn[standard]",
                "ASGI server. The [standard] extra adds faster event loops.",
            ),
            (
                "redis",
                "Async Redis client (redis.asyncio). Semantic cache, rate limit counters, "
                "result storage.",
            ),
            (
                "structlog",
                "Structured logging. Attaches tenant_id to every log line automatically.",
            ),
            (
                "starlette",
                "StreamingResponse used for SSE (bundled with FastAPI). "
                "CORSMiddleware added to allow browser fetch from a different port.",
            ),
        ]
    )
)
story.append(sp(8))

# ── closing ────────────────────────────────────────────────────────────────────
story += [
    hr(),
    Paragraph(
        "After Phase 8 the RAG pipeline is a real service: streaming, authenticated, "
        "cost-controlled, fault-tolerant, and capable of two-hop reasoning. "
        "Phase 9 adds observability (Prometheus metrics, Grafana dashboards, eval drift alerting). "
        "Phase 10 containerizes and deploys to AWS.",
        sNote,
    ),
]

# ── build PDF ─────────────────────────────────────────────────────────────────
OUT = Path("docs/phase8_guide.pdf")
OUT.parent.mkdir(exist_ok=True)

doc = SimpleDocTemplate(
    str(OUT),
    pagesize=letter,
    leftMargin=MARGIN,
    rightMargin=MARGIN,
    topMargin=MARGIN,
    bottomMargin=MARGIN,
    title="Phase 8 — Production Serving",
    author="RAG Engine",
)
doc.build(story)
print(f"Written: {OUT}")

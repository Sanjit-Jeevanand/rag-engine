"""Generate Phase 8 explainer PDF."""

import os

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
    return Paragraph(text.replace("\n", "<br/>").replace(" ", "&nbsp;"), sCode)


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
    """Three-column file list."""
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
    header = [Paragraph(h, sLabel) for h in ["Metric", "Target"]]
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
                    "Streaming API · Auth · Semantic Caching · Rate Limiting · Timeouts",
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
        "By the end of Phase 8 the RAG pipeline you built across Phases 0–7 is served "
        "as a real API. Users get streaming token-by-token responses, similar queries hit "
        "a cache instead of the LLM, each tenant has its own rate limit, and the system "
        "degrades gracefully instead of crashing when something is slow."
    ),
    sp(4),
]

# ── 1. What problem does Phase 8 solve? ───────────────────────────────────────
story += [h1("1. What Problem Does Phase 8 Solve?"), hr()]
story += [
    body(
        "Phases 0–7 gave you a pipeline that works — but it only works when you call it "
        "directly in Python. Phase 8 turns it into something anyone can call over HTTP."
    ),
    body("Three real problems come up the moment you expose a pipeline as an API:"),
    bullets(
        [
            "<b>Speed perception.</b> An LLM takes 2–5 seconds to generate an answer. "
            "If you buffer the entire response and send it at the end, the user stares at a "
            "blank screen. If you stream tokens as they're generated — like ChatGPT does — "
            "the experience feels instant.",
            "<b>Cost.</b> Every LLM call costs money. If you get 100 queries about "
            '"What is the capital of France?" you should pay for one LLM call, not 100. '
            "A semantic cache answers repeated (or similar) questions for free.",
            "<b>Fairness and safety.</b> Without rate limiting, one tenant can flood "
            "your server and degrade service for everyone else. Without auth, anyone can "
            "call your API and run up your bill.",
        ]
    ),
    sp(4),
]

# ── 2. The five features ───────────────────────────────────────────────────────
story += [h1("2. The Five Features — Plain English"), hr()]

# SSE
story += [h2("Feature 1 — Streaming (Server-Sent Events)")]
story += [
    body(
        "Normally an HTTP request works like ordering takeout: you place the order, "
        "wait, and receive everything at once. Streaming works like watching a chef cook "
        "in front of you — you see progress as it happens."
    ),
    body(
        "<b>Server-Sent Events (SSE)</b> keep the HTTP connection open and let the "
        "server push small chunks of text down the wire as they become available. "
        "Each chunk looks like this:"
    ),
    code(
        'data: {"type": "token", "text": "The"}\n\n'
        'data: {"type": "token", "text": " director"}\n\n'
        'data: {"type": "token", "text": " of"}\n\n'
        'data: {"type": "done",  "query_id": "abc-123"}\n\n'
    ),
    body(
        "The browser (or curl) reads each chunk as it arrives and shows it to the user. "
        "This means the user sees the first word in under 300 ms even though the full "
        "response takes 3 seconds."
    ),
    body(
        "<b>In Phase 8:</b> retrieval and reranking finish first (they're fast — under "
        "200 ms). Only then does the SSE stream open and the LLM start generating. "
        "This way the user never sees a half-retrieved answer."
    ),
    sp(6),
]

# Auth
story += [h2("Feature 2 — Auth (Bearer Tokens + Tenant IDs)")]
story += [
    body(
        "A <b>bearer token</b> is just a secret string — like an API key — that you "
        "include in every request. The server checks the token against a lookup table "
        "and finds out which tenant (customer / user) sent the request."
    ),
    body("Every request includes a header like:"),
    code("Authorization: Bearer sk-acme-corp-abc123"),
    body(
        "The server maps <font name='Courier'>sk-acme-corp-abc123</font> → "
        "<font name='Courier'>acme-corp</font>. That tenant ID is attached to every "
        "log line for that request — so at the end of the month you know exactly how "
        "many LLM tokens each tenant consumed. Unknown or missing token → 401 Unauthorized."
    ),
    body(
        "<b>Middleware</b> means this check runs automatically on every request, before "
        "it even reaches your endpoint code. You write it once and forget it."
    ),
    sp(6),
]

# Semantic cache
story += [h2("Feature 3 — Semantic Caching")]
story += [
    body(
        "A normal cache uses exact string matching as the key. For natural language "
        "queries this almost never works:"
    ),
    bullets(
        [
            '"What caused the 2008 financial crisis?" — cache miss',
            '"What were the causes of the 2008 financial crisis?" — also a miss',
            '"Why did the 2008 financial crisis happen?" — also a miss',
        ]
    ),
    body(
        "These are the same question. A <b>semantic cache</b> fixes this by using "
        "the query's embedding (the 384-dimensional vector your pipeline already "
        "computes) as the key, and measuring similarity instead of equality."
    ),
    body("How it works step by step:"),
    bullets(
        [
            "New query arrives → embed it (same model used for retrieval)",
            "Scan previously-cached query embeddings for cosine similarity > 0.97",
            "Above threshold → same question, different words → return the stored answer "
            "without calling the LLM",
            "Below threshold → cache miss → run full retrieval + LLM → store result",
        ]
    ),
    body(
        "<b>Why 0.97?</b> Cosine similarity of 1.0 = identical vectors. At 0.97 you "
        "catch paraphrases and rewordings but not genuinely different questions. "
        "Lower the threshold and you risk returning wrong cached answers."
    ),
    body(
        "<b>Target:</b> ≥ 20% hit rate, which eliminates 20% of LLM calls entirely "
        "— and those 20% become nearly instant responses."
    ),
    sp(6),
]

# Rate limiting
story += [h2("Feature 4 — Rate Limiting (Redis Lua Token Bucket)")]
story += [
    body(
        "Rate limiting means: each tenant gets N requests per minute. Exceed that → "
        "HTTP 429 Too Many Requests with a Retry-After header telling them when to try again."
    ),
    body("<b>Why a Lua script?</b> The naive implementation has a race condition:"),
    bullets(
        [
            "Request A reads count from Redis → sees 99 (under the limit of 100)",
            "Request B reads count from Redis → also sees 99",
            "Both proceed → now you've allowed 101 requests",
        ]
    ),
    body(
        "A <b>Lua script</b> runs atomically on the Redis server — it's a single "
        "indivisible operation. No other command can run between the read and the "
        "decrement. The race is impossible."
    ),
    body("The Lua script is short — here's the core logic:"),
    code(
        "local count = redis.call('GET', KEYS[1]) or 0\n"
        "if tonumber(count) < tonumber(ARGV[1]) then\n"
        "    redis.call('INCR', KEYS[1])\n"
        "    redis.call('EXPIRE', KEYS[1], ARGV[2])\n"
        "    return 1   -- allowed\n"
        "end\n"
        "return 0       -- rejected"
    ),
    body(
        "KEYS[1] is the per-tenant Redis key (e.g. ratelimit:acme-corp). "
        "ARGV[1] is the cap (100). ARGV[2] is the window in seconds (60). "
        "Returns 1 = allowed, 0 = rejected."
    ),
    sp(6),
]

# Timeouts
story += [h2("Feature 5 — Timeouts and Graceful Degradation")]
story += [
    body(
        "Two things can be slow: retrieval (usually fast, but can spike) "
        "and the LLM (usually slow). Without explicit timeouts, a slow upstream "
        "hangs your server indefinitely."
    ),
    body("The rules:"),
    bullets(
        [
            "<b>Retrieval timeout: 200 ms.</b> If HNSW search + reranking takes longer "
            "than 200 ms, stop waiting and return whatever passages were retrieved so far "
            "with a partial=True flag. The user gets partial results rather than nothing.",
            "<b>LLM timeout: 5 seconds.</b> If the LLM hasn't finished in 5 seconds, "
            "stop the stream and return the retrieved passages with a "
            "generation_unavailable=True flag. The user still gets relevant documents — "
            "they just don't get the generated answer.",
        ]
    ),
    body(
        "This is called <b>graceful degradation</b>: the system returns something "
        "useful at every failure level instead of crashing with a 500 error. "
        "In Python this uses asyncio.wait_for() with a timeout argument."
    ),
    sp(4),
]

# ── 3. Architecture ────────────────────────────────────────────────────────────
story += [PageBreak(), h1("3. Architecture — Request Flow"), hr()]
story += [
    body("Here is what happens for every POST /query request, in order:"),
    sp(4),
]

steps_arch = [
    (
        "Auth middleware",
        "Runs before your endpoint. Reads Authorization header → resolves tenant_id → "
        "binds to log context. Returns 401 if missing or unknown.",
    ),
    (
        "Rate limit check",
        "Calls the Lua script with the tenant's Redis key. If over limit → return 429 "
        "with Retry-After header. Otherwise continue.",
    ),
    (
        "Semantic cache lookup",
        "Embeds the query → scans cached embeddings → if cosine sim > 0.97, stream "
        "the cached answer as SSE tokens. Done — no retrieval, no LLM call.",
    ),
    (
        "Retrieval (with 200ms timeout)",
        "Run BM25 + HNSW dense retrieval → RRF fusion → cross-encoder rerank. "
        "If this takes > 200ms, return partial SSE event with whatever was retrieved.",
    ),
    (
        "Open SSE stream",
        "All retrieval is done before the stream opens. "
        "Now yield tokens as the LLM generates them.",
    ),
    (
        "LLM generation (with 5s timeout)",
        "Stream tokens via the OpenAI streaming API. "
        "If LLM takes > 5s → emit generation_unavailable event with passages.",
    ),
    (
        "Store result",
        "Write the full result (answer + citations) to Redis keyed by query_id. "
        "Also write to the semantic cache for future hits. Emit done event.",
    ),
    (
        "GET /query/{id}",
        "Separate endpoint. Fetches the stored result from Redis by query_id — "
        "returns the full answer with citations as regular JSON (no streaming).",
    ),
]
story.append(kv_table([(f"{i+1}. {k}", v) for i, (k, v) in enumerate(steps_arch)]))
story.append(sp(8))

# ── 4. The seven files you'll build ───────────────────────────────────────────
story += [h1("4. Files You'll Build"), hr()]
story += [
    body(
        "Phase 8 adds a new package — src/rag_engine/api/ — plus tests, an infra "
        "file, and a startup script. Each file has one job."
    ),
    sp(4),
]
story.append(
    files_table(
        [
            (
                "src/rag_engine/api/__init__.py",
                "Package root — makes api/ importable as rag_engine.api",
            ),
            (
                "src/rag_engine/api/models.py",
                "Pydantic data shapes: QueryRequest (what comes in), Citation (a source passage), "
                "QueryResult (full response including cache_hit, partial, generation_unavailable flags)",
            ),
            (
                "src/rag_engine/api/auth.py",
                "Middleware: reads Authorization header → resolves tenant_id → binds to log "
                "context → 401 on unknown token",
            ),
            (
                "src/rag_engine/api/ratelimit.py",
                "Async function wrapping the Lua script: check_and_decrement(redis, tenant_id) → "
                "bool. Returns False when over limit.",
            ),
            (
                "src/rag_engine/api/stream.py",
                "Pure functions that format SSE data: lines — token_event, done_event, "
                "partial_event, gen_unavailable_event, error_event",
            ),
            (
                "src/rag_engine/api/cache.py",
                "SemanticCache class: get(query) → QueryResult | None, "
                "set(query, result) → None. Maintains hit/total counters in Redis.",
            ),
            (
                "src/rag_engine/api/app.py",
                "The FastAPI app. Wires together all of the above. "
                "POST /query (SSE), GET /query/{id}, /health, /ready, /metrics/cache",
            ),
            (
                "scripts/run_server.py",
                "One-liner: starts uvicorn on port 8000, reads RAG_PORT / RAG_WORKERS from env",
            ),
            (
                "tests/test_api.py",
                "Pytest tests with mocked Redis and mocked retrieval. "
                "Tests every failure path: 401, 429, cache hit, retrieval timeout, LLM timeout",
            ),
            (
                "infra/redis.yml",
                "Docker Compose file — starts a local Redis container in one command",
            ),
        ]
    )
)
story.append(sp(8))

# ── 5. Key concepts explained ─────────────────────────────────────────────────
story += [PageBreak(), h1("5. Key Concepts Explained"), hr()]

story += [h2("What is Redis?")]
story += [
    body(
        "Redis is an in-memory key-value store — think of it as a Python dict that "
        "lives in its own process. You can set a key to expire after N seconds, "
        "atomically increment counters, and run Lua scripts server-side. "
        "It's used here for three things: the semantic cache, rate limit counters, "
        "and storing query results for GET /query/{id}."
    ),
    sp(6),
]

story += [h2("What is FastAPI?")]
story += [
    body(
        "FastAPI is a Python web framework. You define functions decorated with "
        "@app.post('/query') and FastAPI handles routing, input validation (via "
        "Pydantic), and auto-generated API docs at /docs. It's async-native — "
        "all your endpoint functions can be async def, which matters for streaming."
    ),
    sp(6),
]

story += [h2("What is a Pydantic Model?")]
story += [
    body(
        "A Pydantic model is a Python class that describes the shape of data and "
        "validates it automatically. You've already used one in config.py. "
        "In the API layer you define what a request looks like (QueryRequest) and "
        "what a response looks like (QueryResult). FastAPI uses these to validate "
        "incoming JSON and serialize outgoing JSON."
    ),
    code(
        "class QueryRequest(BaseModel):\n"
        "    query: str\n"
        "    max_hops: int = 2\n"
        "    top_k: int = 5\n\n"
        "class Citation(BaseModel):\n"
        "    passage_id: str\n"
        "    title: str\n"
        "    text: str\n"
        "    score: float"
    ),
    sp(6),
]

story += [h2("What is Middleware?")]
story += [
    body(
        "Middleware is code that wraps every request/response cycle. Think of it as "
        "a layer around your endpoint functions. Auth middleware runs before any "
        "endpoint is called — if it rejects the request (wrong token), the endpoint "
        "never runs. You write one middleware class and it applies everywhere "
        "automatically — no need to add auth checks inside each endpoint."
    ),
    sp(6),
]

story += [h2("What is asyncio.wait_for()?")]
story += [
    body(
        "In async Python, asyncio.wait_for(coroutine, timeout=N) runs a coroutine "
        "but cancels it after N seconds if it hasn't finished. It raises "
        "asyncio.TimeoutError which you catch and handle — in Phase 8 that means "
        "returning a partial or degraded response instead of hanging forever."
    ),
    code(
        "try:\n"
        "    passages = await asyncio.wait_for(\n"
        "        retrieve_and_rerank(query), timeout=0.2\n"
        "    )\n"
        "except asyncio.TimeoutError:\n"
        "    yield partial_event([])  # return what we have\n"
        "    return"
    ),
    sp(6),
]

story += [h2("What is cosine similarity?")]
story += [
    body(
        "You already use this for retrieval — it's the dot product of two normalized "
        "vectors. Two identical sentences have cosine similarity 1.0. "
        "Two completely unrelated sentences might be 0.2–0.4. "
        "Paraphrases of the same question tend to land at 0.95–0.99. "
        "The cache uses 0.97 as the threshold — above that, treat as the same question."
    ),
    sp(4),
]

# ── 6. Build order ────────────────────────────────────────────────────────────
story += [PageBreak(), h1("6. Build Order and Why It Matters"), hr()]
story += [
    body(
        "Each piece depends on the ones before it. Build in this order so you can "
        "test each step in isolation before wiring them together."
    ),
    sp(6),
]
build_order = [
    (
        "1. Infra + deps",
        "Start Redis locally. Install libraries. Nothing else works without Redis running.",
    ),
    (
        "2. Pydantic models",
        "Define QueryRequest, Citation, QueryResult first. Everything else imports these "
        "shapes. No logic here — just data contracts.",
    ),
    (
        "3. SSE helpers",
        "Pure functions that format SSE strings. Test them with a simple print — "
        "no server needed. Ensures the streaming format is correct before wiring to FastAPI.",
    ),
    (
        "4. Auth middleware",
        "One class, one job. Test it standalone by passing fake headers. "
        "Once tested, mount on the app and forget it.",
    ),
    (
        "5. Rate limiter",
        "One async function wrapping the Lua script. Test with a real local Redis. "
        "Confirm the atomic guarantee: flood 200 concurrent calls, count should "
        "never exceed cap.",
    ),
    (
        "6. Semantic cache",
        "SemanticCache class. Test get() and set() with a real local Redis and a "
        "small embedder. Confirm cosine threshold works.",
    ),
    (
        "7. LLM streaming",
        "Add stream_complete() to llm.py. Test it standalone — just print tokens "
        "as they arrive. Confirm the generator works before plugging into the endpoint.",
    ),
    (
        "8. FastAPI app",
        "Wire everything together. The app's lifespan loads the HNSW index, embedder, "
        "reranker, and Redis client once at startup — not per request.",
    ),
    (
        "9. Tests",
        "Mock Redis and retrieval. Test each failure path (401, 429, timeout, "
        "cache hit) without needing a real index or LLM.",
    ),
    (
        "10. Smoke test",
        "Run the real server. curl the endpoint. Repeat a query and confirm "
        "the cache hit in logs. Flood one tenant and confirm the other is unaffected.",
    ),
]
story.append(kv_table(build_order))
story.append(sp(8))

# ── 7. Target metrics ─────────────────────────────────────────────────────────
story += [h1("7. Target Metrics"), hr()]
story.append(
    metrics_table(
        [
            ("End-to-end P95 latency", "< 800 ms"),
            ("Time to first token (after retrieval)", "< 300 ms"),
            ("Semantic cache hit rate", "≥ 20%"),
            ("Cost per query (cache miss path)", "≤ $0.005"),
            ("Retrieval timeout", "200 ms — return partial results"),
            ("LLM timeout", "5 s — return passages + generation_unavailable flag"),
            ("Rate limit precision", "Atomic — no 2× burst possible"),
        ]
    )
)
story.append(sp(8))

# ── 8. Common mistakes ────────────────────────────────────────────────────────
story += [h1("8. Common Mistakes to Avoid"), hr()]
story.append(
    bullets(
        [
            "<b>Opening the SSE stream before retrieval finishes.</b> If you stream "
            "while retrieval is still running, you can't add citations to the stream "
            "because you don't know them yet. Finish retrieval first, then open the stream.",
            "<b>Using SETNX (set-if-not-exists) for rate limiting.</b> This has a "
            "race — two requests can both see the key missing and both set it. "
            "Use the Lua script.",
            "<b>Using exact string keys for the cache.</b> Natural language queries "
            "almost never repeat exactly. You'll get near-zero hit rate.",
            "<b>Setting the cosine threshold too low (e.g. 0.85).</b> At 0.85, "
            "questions about different topics but similar words can collide — "
            '"What caused World War I?" and "What ended World War I?" '
            "might both hit the same cache entry. Start at 0.97.",
            "<b>Not setting a TTL on cache entries.</b> Without expiry, cache entries "
            "accumulate forever and stale answers stay in the cache after the corpus "
            "is rebuilt. Set a 1-hour TTL.",
            "<b>Loading the HNSW index per request.</b> Loading a 15 GB index file "
            "takes 30+ seconds. Load it once in the FastAPI lifespan context manager "
            "at startup and reuse it for every request.",
        ]
    )
)
story.append(sp(8))

# ── 9. New dependencies ────────────────────────────────────────────────────────
story += [h1("9. New Dependencies"), hr()]
story.append(
    kv_table(
        [
            (
                "fastapi",
                "The web framework. Handles routing, Pydantic validation, auto-docs at /docs.",
            ),
            (
                "uvicorn[standard]",
                "The ASGI server that runs FastAPI. The [standard] extra adds websocket "
                "support and faster event loops.",
            ),
            (
                "sse-starlette",
                "Adds EventSourceResponse to FastAPI — wraps your async generator and "
                "formats it as a proper SSE stream.",
            ),
            (
                "redis",
                "Async Redis client (redis.asyncio). Used for the semantic cache, "
                "rate limit counters, and result storage.",
            ),
            (
                "structlog",
                "Structured logging library. You already use it — Phase 8 adds "
                "bind_contextvars() to attach tenant_id to every log line automatically.",
            ),
        ]
    )
)
story.append(sp(8))

# ── closing ────────────────────────────────────────────────────────────────────
story += [
    hr(),
    Paragraph(
        "After Phase 8, the RAG pipeline is a real service: streaming, authenticated, "
        "cost-controlled, and fault-tolerant. Phase 9 adds observability (Prometheus "
        "metrics, Grafana dashboards, eval drift alerting). Phase 10 containerizes "
        "and deploys it to AWS.",
        sNote,
    ),
]

# ── build PDF ─────────────────────────────────────────────────────────────────
OUT = "docs/phase8_guide.pdf"

os.makedirs("docs", exist_ok=True)

doc = SimpleDocTemplate(
    OUT,
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

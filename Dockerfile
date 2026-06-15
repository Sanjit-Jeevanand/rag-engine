# ── Stage 1: dependency installer ────────────────────────────────────────────
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./

RUN uv sync --frozen --no-install-project --no-dev

COPY src/ ./src/
RUN uv sync --frozen --no-dev

# ── Stage 2: lean runtime image ───────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

RUN groupadd --system rag && useradd --system --gid rag --no-create-home rag

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src   /app/src

COPY web/ /app/web/
COPY scripts/ /app/scripts/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/app/.cache/huggingface

RUN mkdir -p /app/.cache/huggingface /app/data && chown -R rag:rag /app

VOLUME ["/app/data", "/app/.cache/huggingface"]

USER rag

EXPOSE 8000

CMD ["uvicorn", "rag_engine.api.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]

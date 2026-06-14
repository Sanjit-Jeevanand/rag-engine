"""Start the RAG Engine API server."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Prevent OMP deadlock when PyTorch is called from multiple threads (Apple Silicon)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import uvicorn  # noqa: E402

if __name__ == "__main__":
    uvicorn.run(
        "rag_engine.api.app:app",
        host="0.0.0.0",
        port=int(os.environ.get("RAG_PORT", "8000")),
        workers=int(os.environ.get("RAG_WORKERS", "1")),
        reload=os.environ.get("RAG_RELOAD", "true").lower() == "true",
    )

from pathlib import Path

from rag_engine.ingest.embedder import DEVICE, MODEL_NAME, run_embedder

print(f"Model: {MODEL_NAME}  device: {DEVICE}")
run_embedder(Path("data/docs.db"), Path("data/vectors.bin"), show_progress=True)
print("done")

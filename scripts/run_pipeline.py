"""
Two-pass pipeline: selects the top MAX_ARTICLES Wikipedia articles by
incoming_links (most-linked = most important), then ingests only those.
"""

from pathlib import Path

from tqdm import tqdm

from rag_engine.ingest.parser import parse_snapshot
from rag_engine.ingest.pipeline import BATCH_SIZE, _insert_batch, split_text
from rag_engine.ingest.schema import init_db

MAX_ARTICLES = 1_000_000
SNAPSHOT = Path("data/wiki-dump.json.gz")
DB = Path("data/docs.db")

# ── Pass 1: collect (article_id, incoming_links) for all articles ────────────
print("Pass 1: scanning incoming_links across all articles…")
scores: list[tuple[int, str]] = []  # (incoming_links, article_id)

for article in tqdm(parse_snapshot(SNAPSHOT), unit=" articles"):
    scores.append((article.incoming_links, article.article_id))

print(f"  Total articles found: {len(scores):,}")

scores.sort(reverse=True)
top_ids: set[str] = {aid for _, aid in scores[:MAX_ARTICLES]}
cutoff = scores[MAX_ARTICLES - 1][0]
print(f"  Keeping top {MAX_ARTICLES:,} — incoming_links cutoff: {cutoff:,}")

# ── Pass 2: ingest only the top articles ─────────────────────────────────────
print("Pass 2: ingesting top articles…")
conn = init_db(DB)
batch = []
inserted = 0

for article in tqdm(parse_snapshot(SNAPSHOT), unit=" articles"):
    if article.article_id not in top_ids:
        continue
    chunks = split_text(article.text)
    for i, text in enumerate(chunks):
        batch.append(
            (
                article.article_id,
                i,
                article.title,
                article.categories,
                article.timestamp,
                text,
                len(chunks),
            )
        )
        if len(batch) >= BATCH_SIZE:
            _insert_batch(conn, batch)
            batch.clear()
    inserted += 1

if batch:
    _insert_batch(conn, batch)

conn.close()
print(f"Done — {inserted:,} articles, check chunk count in docs.db")

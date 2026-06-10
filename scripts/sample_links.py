"""Sample 50K articles to understand the incoming_links distribution."""

from pathlib import Path

from tqdm import tqdm

from rag_engine.ingest.parser import parse_snapshot

SNAPSHOT = Path("data/wiki-dump.json.gz")
SAMPLE = 50_000

counts = []
bar = tqdm(total=SAMPLE, desc="Sampling")

for article in parse_snapshot(SNAPSHOT):
    counts.append(article.incoming_links)
    bar.update(1)
    if len(counts) >= SAMPLE:
        break

bar.close()
counts.sort(reverse=True)

print(f"Sampled {len(counts):,} articles")
print(f"Max:    {counts[0]:,}")
print(f"p99:    {counts[int(len(counts) * 0.01)]:,}")
print(f"p90:    {counts[int(len(counts) * 0.10)]:,}")
print(f"p75:    {counts[int(len(counts) * 0.25)]:,}")
print(f"p50:    {counts[int(len(counts) * 0.50)]:,}")
print(f"p25:    {counts[int(len(counts) * 0.75)]:,}")
print(f"Min:    {counts[-1]:,}")

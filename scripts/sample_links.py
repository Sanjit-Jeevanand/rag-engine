from pathlib import Path

from tqdm import tqdm

from rag_engine.ingest.parser import parse_snapshot

SNAPSHOT = Path("data/wiki-dump.json.gz")
SAMPLE = 50_000

counts = []
with tqdm(total=SAMPLE, desc="Sampling") as bar:
    for article in parse_snapshot(SNAPSHOT):
        counts.append(article.incoming_links)
        bar.update(1)
        if len(counts) >= SAMPLE:
            break

counts.sort(reverse=True)
n = len(counts)

print(f"Sampled {n:,} articles")
print(f"Max:    {counts[0]:,}")
print(f"p99:    {counts[int(n * 0.01)]:,}")
print(f"p90:    {counts[int(n * 0.10)]:,}")
print(f"p75:    {counts[int(n * 0.25)]:,}")
print(f"p50:    {counts[int(n * 0.50)]:,}")
print(f"p25:    {counts[int(n * 0.75)]:,}")
print(f"Min:    {counts[-1]:,}")

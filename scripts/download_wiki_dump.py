#!/usr/bin/env python3
"""
Download the Wikipedia CirrusSearch dump.

No dependencies beyond the Python stdlib — run with any Python 3.8+.

Usage:
    python scripts/download_wiki_dump.py

Downloads to data/wiki-dump.json.gz (~20 GB compressed).
Automatically resumes if interrupted — safe to Ctrl-C and re-run.
"""

import sys
import urllib.request
from pathlib import Path

URL = (
    "https://dumps.wikimedia.org/other/cirrussearch/20251229/"
    "enwiki-20251229-cirrussearch-content.json.gz"
)
DEST = Path("data/wiki-dump.json.gz")
CHUNK = 1024 * 1024  # 1 MB


def _progress(downloaded: int, total: int) -> None:
    if total <= 0:
        gb = downloaded / 1024**3
        print(f"\r  {gb:.2f} GB downloaded", end="", flush=True)
        return
    pct = downloaded / total * 100
    filled = int(pct / 2)
    bar = "█" * filled + "░" * (50 - filled)
    gb_done = downloaded / 1024**3
    gb_total = total / 1024**3
    msg = f"\r  [{bar}] {pct:.1f}%  {gb_done:.2f}/{gb_total:.2f} GB"
    print(msg, end="", flush=True)


def download() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)

    existing = DEST.stat().st_size if DEST.exists() else 0
    if existing:
        print(f"Resuming from {existing / 1024**3:.2f} GB already downloaded...")
        req = urllib.request.Request(URL, headers={"Range": f"bytes={existing}-"})
        mode = "ab"
    else:
        req = urllib.request.Request(URL)
        mode = "wb"

    with urllib.request.urlopen(req) as response:
        content_length = int(response.headers.get("Content-Length", 0))
        total = existing + content_length
        downloaded = existing

        with DEST.open(mode) as f:
            while True:
                chunk = response.read(CHUNK)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                _progress(downloaded, total)

    print(f"\nDone — {DEST} ({DEST.stat().st_size / 1024**3:.2f} GB)")


if __name__ == "__main__":
    print("Wikipedia CirrusSearch dump")
    print(f"  destination : {DEST}")
    print(f"  source      : {URL}\n")
    try:
        download()
    except KeyboardInterrupt:
        print("\nInterrupted. Run again to resume from where it stopped.")
        sys.exit(1)

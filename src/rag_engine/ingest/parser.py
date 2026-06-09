import gzip
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass
class WikiArticle:
    article_id: str
    title: str
    text: str
    categories: str
    timestamp: str


def parse_snapshot(path: Path) -> Iterator[WikiArticle]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        while True:
            index_line = f.readline()
            if not index_line:
                break
            content_line = f.readline()
            if not content_line:
                break
            try:
                index = json.loads(index_line)
                content = json.loads(content_line)
                if content.get("namespace") != 0:
                    continue
                text = (content.get("text") or "").strip()
                if not text:
                    continue
                yield WikiArticle(
                    article_id=str(index["index"]["_id"]),
                    title=content["title"].strip(),
                    text=text,
                    categories=" ".join(content.get("categories") or []),
                    timestamp=content.get("timestamp", ""),
                )
            except (KeyError, json.JSONDecodeError):
                continue

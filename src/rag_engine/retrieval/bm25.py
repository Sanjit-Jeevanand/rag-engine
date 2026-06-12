import json
from pathlib import Path

import bm25s  # type: ignore[import-untyped]


class BM25Retriever:
    def __init__(self, doc_ids: list[str], doc_texts: list[str]) -> None:
        self._doc_ids = doc_ids
        tokens = bm25s.tokenize(doc_texts, stopwords="en", show_progress=False)
        self._bm25 = bm25s.BM25()
        self._bm25.index(tokens)

    @classmethod
    def load(cls, index_dir: Path) -> "BM25Retriever":
        doc_ids: list[str] = json.loads((index_dir / "doc_ids.json").read_text())
        obj = object.__new__(cls)
        obj._doc_ids = doc_ids
        obj._bm25 = bm25s.BM25.load(str(index_dir), load_corpus=False)
        return obj

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        self._bm25.save(str(index_dir))
        (index_dir / "doc_ids.json").write_text(json.dumps(self._doc_ids))

    def retrieve(self, query: str, k: int) -> list[str]:
        tokens = bm25s.tokenize([query], stopwords="en", show_progress=False)
        results, _ = self._bm25.retrieve(tokens, k=min(k, len(self._doc_ids)))
        return [self._doc_ids[int(i)] for i in results[0]]

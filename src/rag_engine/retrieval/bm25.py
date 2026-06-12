import bm25s  # type: ignore[import-untyped]


class BM25Retriever:
    def __init__(self, doc_ids: list[str], doc_texts: list[str]) -> None:
        self._doc_ids = doc_ids
        tokens = bm25s.tokenize(doc_texts, stopwords="en", show_progress=False)
        self._bm25 = bm25s.BM25()
        self._bm25.index(tokens)

    def retrieve(self, query: str, k: int) -> list[str]:
        tokens = bm25s.tokenize([query], stopwords="en", show_progress=False)
        results, _ = self._bm25.retrieve(tokens, k=min(k, len(self._doc_ids)))
        return [self._doc_ids[int(i)] for i in results[0]]

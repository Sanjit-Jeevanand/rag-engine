from collections import defaultdict

_RRF_K = 60


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 10) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] += 1.0 / (_RRF_K + rank + 1)
    return sorted(scores, key=lambda d: scores[d], reverse=True)[:k]

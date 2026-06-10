import math
import re


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int = 10) -> float:
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, title in enumerate(retrieved[:k])
        if title in relevant
    )
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(retrieved: list[str], relevant: set[str], k: int = 10) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for title in retrieved[:k] if title in relevant)
    return hits / len(relevant)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    for i, title in enumerate(retrieved):
        if title in relevant:
            return 1.0 / (i + 1)
    return 0.0


def exact_match(prediction: str, gold: str) -> float:
    return 1.0 if _normalize(prediction) == _normalize(gold) else 0.0


def f1(prediction: str, gold: str) -> float:
    pred_tokens = _normalize(prediction).split()
    gold_tokens = _normalize(gold).split()
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = sum(pred_tokens.count(t) for t in common) / len(pred_tokens)
    recall = sum(gold_tokens.count(t) for t in common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9 ]", "", text)
    return re.sub(r"\s+", " ", text).strip()

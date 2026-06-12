from unittest.mock import MagicMock, patch

import numpy as np

from rag_engine.agent.loop import MultiHopAgent

DOC_TEXTS = {
    "A": "Paris is the capital of France.",
    "B": "France is a country in Western Europe.",
    "C": "Berlin is the capital of Germany.",
    "D": "The Eiffel Tower is in Paris.",
    "E": "Something completely unrelated.",
}

_PATCH = "rag_engine.agent.loop.complete"


def _reranker(scores: list[float]) -> MagicMock:
    r = MagicMock()
    r.scores.return_value = np.array(scores, dtype=np.float32)
    return r


def _retrieve(q: str, k: int) -> list[str]:
    return list(DOC_TEXTS.keys())[:k]


# ── Test 1: abstention ────────────────────────────────────────────────────────


def test_abstains_when_scores_below_threshold() -> None:
    agent = MultiHopAgent(
        retrieve=_retrieve,
        doc_texts=DOC_TEXTS,
        reranker=_reranker([-6.0, -7.0, -8.0]),
        abstention_threshold=-4.0,
    )
    with patch(_PATCH) as mock_llm:
        result = agent.answer("Who invented the telephone?")

    assert result.abstained
    assert result.answer == "I cannot answer from the available evidence."
    # No LLM calls before abstaining
    mock_llm.assert_not_called()


# ── Test 2: citation hallucination ───────────────────────────────────────────


def test_hallucinated_citation_is_flagged() -> None:
    agent = MultiHopAgent(
        retrieve=lambda q, k: ["A", "B", "C"][:k],
        doc_texts=DOC_TEXTS,
        reranker=_reranker([5.0, 4.0, 3.0]),
    )
    responses = [
        "ANSWER_DIRECT",  # bridge → skip hop 2
        'Paris is the capital. [A] CITATIONS: ["A", "GHOST_DOC"]',  # cites fake
        "FULLY_SUPPORTED: yes\nMISSING: nothing\nSEARCH_QUERY: none",  # reflect
    ]
    with patch(_PATCH, side_effect=responses):
        result = agent.answer("What is the capital of France?")

    assert "GHOST_DOC" in result.hallucinated_ids
    assert "A" not in result.hallucinated_ids


# ── Test 3: hop cap ───────────────────────────────────────────────────────────


def test_hop_cap_is_respected() -> None:
    retrieve_calls = [0]

    def counting_retrieve(q: str, k: int) -> list[str]:
        retrieve_calls[0] += 1
        return ["A", "B"][:k]

    agent = MultiHopAgent(
        retrieve=counting_retrieve,
        doc_texts=DOC_TEXTS,
        reranker=_reranker([5.0, 4.0, 3.0]),
        max_hops=1,
    )
    # max_hops=1: bridge fires, hop 2 and reflection both skipped
    responses = [
        "some bridge query",  # bridge call (hop 2 blocked by cap)
        'Paris. [A] CITATIONS: ["A"]',  # answer
    ]
    with patch(_PATCH, side_effect=responses):
        result = agent.answer("What is the capital of France?")

    assert len(result.hops) == 1
    assert retrieve_calls[0] == 1
    assert not result.reflection_triggered


# ── Test 4: reflection triggers hop 3 ────────────────────────────────────────


def test_reflection_triggers_extra_hop() -> None:
    retrieve_calls = [0]

    def counting_retrieve(q: str, k: int) -> list[str]:
        retrieve_calls[0] += 1
        return ["A", "B"][:k]

    agent = MultiHopAgent(
        retrieve=counting_retrieve,
        doc_texts=DOC_TEXTS,
        reranker=_reranker([5.0, 4.0, 3.0]),
        max_hops=3,
    )
    responses = [
        "bridge entity",  # bridge → hop 2
        'Partial answer. [A] CITATIONS: ["A"]',  # first answer
        "FULLY_SUPPORTED: no\nMISSING: birthplace\nSEARCH_QUERY: birthplace query",
        'Full answer. [A] [B] CITATIONS: ["A", "B"]',  # regenerated after hop 3
    ]
    with patch(_PATCH, side_effect=responses):
        result = agent.answer("Where was the founder of X born?")

    assert result.reflection_triggered
    assert len(result.hops) == 3  # hop1 + hop2 + hop3
    assert retrieve_calls[0] == 3

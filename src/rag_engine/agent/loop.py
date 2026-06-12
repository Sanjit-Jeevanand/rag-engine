import json
import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from rag_engine.agent.llm import complete
from rag_engine.retrieval.reranker import CrossEncoderReranker

_BRIDGE_SYSTEM = (
    "You extract the next search query needed for multi-hop question answering. "
    "Respond with ONLY a short search query (3-10 words), or 'ANSWER_DIRECT' if "
    "the question can already be answered from the provided passages."
)

_CANNOT_ANSWER = "I cannot answer from the available evidence."

_ANSWER_SYSTEM = (
    "Answer the question using ONLY the provided passages. "
    "Use exactly this format:\n"
    "ANSWER: <concise factual answer — a name, date, place, or yes/no>\n"
    'CITATIONS: ["Title1", "Title2"]\n\n'
    f"If the passages do not contain enough information, respond with:\n"
    f"ANSWER: {_CANNOT_ANSWER}\n"
    "CITATIONS: []"
)

_REFLECT_SYSTEM = (
    "Check if an answer is fully supported by the provided passages. "
    "Respond in exactly this format:\n"
    "FULLY_SUPPORTED: yes/no\n"
    "MISSING: <what is missing or 'nothing'>\n"
    "SEARCH_QUERY: <search query to fill the gap or 'none'>"
)


@dataclass
class Hop:
    query: str
    retrieved: list[str]


@dataclass
class AgentResult:
    answer: str
    hops: list[Hop]
    cited_ids: list[str]
    hallucinated_ids: list[str]
    abstained: bool
    reflection_triggered: bool


class MultiHopAgent:
    def __init__(
        self,
        retrieve: Callable[[str, int], list[str]],
        doc_texts: dict[str, str],
        reranker: CrossEncoderReranker,
        *,
        model: str = "gpt-4o-mini",
        top_k: int = 5,
        max_hops: int = 3,
        abstention_threshold: float = -4.0,
    ) -> None:
        self._retrieve = retrieve
        self._doc_texts = doc_texts
        self._reranker = reranker
        self._model = model
        self._top_k = top_k
        self._max_hops = max_hops
        self._abstention_threshold = abstention_threshold

    def answer(self, question: str) -> AgentResult:
        # ── Hop 1 ────────────────────────────────────────────────────────────
        hop1 = self._retrieve(question, self._top_k)
        hops: list[Hop] = [Hop(query=question, retrieved=hop1)]
        pool = list(hop1)

        # Abstention: check before any LLM call
        if hop1:
            top_scores = self._reranker.scores(question, hop1[:3], self._doc_texts)
            should_abstain = float(np.max(top_scores)) < self._abstention_threshold
        else:
            should_abstain = True

        if should_abstain:
            return AgentResult(
                answer="I cannot answer from the available evidence.",
                hops=hops,
                cited_ids=[],
                hallucinated_ids=[],
                abstained=True,
                reflection_triggered=False,
            )

        # ── LLM call 1: bridge extraction ────────────────────────────────────
        passages = _format_passages(pool, self._doc_texts)
        msg = f"Question: {question}\n\nPassages:\n{passages}"
        bridge = complete(
            [{"role": "user", "content": msg}],
            model=self._model,
            max_tokens=64,
            system=_BRIDGE_SYSTEM,
        ).strip()

        # ── Hop 2 ────────────────────────────────────────────────────────────
        if bridge.upper() != "ANSWER_DIRECT" and len(hops) < self._max_hops:
            hop2 = self._retrieve(bridge, self._top_k)
            hops.append(Hop(query=bridge, retrieved=hop2))
            pool.extend(d for d in hop2 if d not in pool)

        # ── LLM call 2: answer generation ────────────────────────────────────
        passages = _format_passages(pool, self._doc_texts)
        msg = f"Question: {question}\n\nPassages:\n{passages}"
        raw = complete(
            [{"role": "user", "content": msg}],
            model=self._model,
            max_tokens=512,
            system=_ANSWER_SYSTEM,
        )
        answer_text = _extract_answer(raw)
        cited = _extract_citations(raw)

        # LLM-based abstention: model said it can't answer from passages
        if answer_text == _CANNOT_ANSWER:
            return AgentResult(
                answer=answer_text,
                hops=hops,
                cited_ids=[],
                hallucinated_ids=[],
                abstained=True,
                reflection_triggered=False,
            )

        # ── LLM call 3: self-reflection ───────────────────────────────────────
        reflection_triggered = False
        if len(hops) < self._max_hops:
            msg = (
                f"Question: {question}\n"
                f"Answer: {answer_text}\n\n"
                f"Passages:\n{passages}"
            )
            raw_reflect = complete(
                [{"role": "user", "content": msg}],
                model=self._model,
                max_tokens=128,
                system=_REFLECT_SYSTEM,
            )
            if "FULLY_SUPPORTED: no" in raw_reflect:
                gap = _extract_field(raw_reflect, "SEARCH_QUERY")
                if gap and gap.lower() != "none":
                    reflection_triggered = True

                    # ── Hop 3 ────────────────────────────────────────────────
                    hop3 = self._retrieve(gap, self._top_k)
                    hops.append(Hop(query=gap, retrieved=hop3))
                    pool.extend(d for d in hop3 if d not in pool)

                    # ── LLM call 4: regenerate with extra context ─────────────
                    passages = _format_passages(pool, self._doc_texts)
                    msg = f"Question: {question}\n\nPassages:\n{passages}"
                    raw = complete(
                        [{"role": "user", "content": msg}],
                        model=self._model,
                        max_tokens=512,
                        system=_ANSWER_SYSTEM,
                    )
                    answer_text = _extract_answer(raw)
                    cited = _extract_citations(raw)
                    if answer_text == _CANNOT_ANSWER:
                        return AgentResult(
                            answer=answer_text,
                            hops=hops,
                            cited_ids=[],
                            hallucinated_ids=[],
                            abstained=True,
                            reflection_triggered=reflection_triggered,
                        )

        # ── Citation grounding ───────────────────────────────────────────────
        retrieved_set = set(pool)
        hallucinated = [c for c in cited if c not in retrieved_set]

        return AgentResult(
            answer=answer_text,
            hops=hops,
            cited_ids=cited,
            hallucinated_ids=hallucinated,
            abstained=False,
            reflection_triggered=reflection_triggered,
        )


def _format_passages(doc_ids: list[str], doc_texts: dict[str, str]) -> str:
    parts = []
    for doc_id in doc_ids:
        text = doc_texts.get(doc_id, "")
        if text:
            # 800 chars ≈ 200 tokens; keeps prompt within budget across 15 passages
            parts.append(f"[{doc_id}]\n{text[:800]}")
    return "\n\n".join(parts)


def _extract_answer(raw: str) -> str:
    # Prefer the explicit ANSWER: line for EM-comparable extraction
    match = re.search(r"^ANSWER:\s*(.+)$", raw, re.MULTILINE)
    if match:
        return match.group(1).strip()
    # Fallback: everything before CITATIONS:
    idx = raw.find("CITATIONS:")
    return raw[:idx].strip() if idx != -1 else raw.strip()


def _extract_citations(raw: str) -> list[str]:
    match = re.search(r"CITATIONS:\s*(\[.*?\])", raw, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(1))
            if isinstance(result, list):
                return [str(x) for x in result]
        except json.JSONDecodeError:
            pass
    # Fallback: pull every [Title] from the answer body
    return re.findall(r"\[([^\]]+)\]", raw)


def _extract_field(text: str, field: str) -> str:
    match = re.search(rf"{field}:\s*(.+)", text)
    return match.group(1).strip() if match else "none"

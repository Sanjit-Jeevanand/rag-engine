import threading
from dataclasses import dataclass

# ── Pricing (USD per 1M tokens) ───────────────────────────────────────────────
_GPT4O_MINI_INPUT = 0.150
_GPT4O_MINI_OUTPUT = 0.600
_EMBED_SMALL = 0.020  # text-embedding-3-small


@dataclass(frozen=True)
class CostSnapshot:
    llm_calls: int
    llm_input_tokens: int
    llm_output_tokens: int
    embed_calls: int
    embed_tokens: int
    reranker_calls: int

    @property
    def estimated_usd(self) -> float:
        llm_cost = (
            self.llm_input_tokens * _GPT4O_MINI_INPUT
            + self.llm_output_tokens * _GPT4O_MINI_OUTPUT
        ) / 1_000_000
        embed_cost = self.embed_tokens * _EMBED_SMALL / 1_000_000
        return round(llm_cost + embed_cost, 8)

    def as_dict(self) -> dict[str, int | float]:
        return {
            "llm_calls": self.llm_calls,
            "llm_input_tokens": self.llm_input_tokens,
            "llm_output_tokens": self.llm_output_tokens,
            "embed_calls": self.embed_calls,
            "embed_tokens": self.embed_tokens,
            "reranker_calls": self.reranker_calls,
            "estimated_usd": self.estimated_usd,
        }


class _CostTracker:
    """Thread-local accumulator. Call reset() before each question."""

    def __init__(self) -> None:
        self._local = threading.local()

    def _state(self) -> dict[str, int]:
        if not hasattr(self._local, "s"):
            self._local.s = {
                "llm_calls": 0,
                "llm_input_tokens": 0,
                "llm_output_tokens": 0,
                "embed_calls": 0,
                "embed_tokens": 0,
                "reranker_calls": 0,
            }
        s: dict[str, int] = self._local.s
        return s

    def reset(self) -> None:
        self._local.s = {
            "llm_calls": 0,
            "llm_input_tokens": 0,
            "llm_output_tokens": 0,
            "embed_calls": 0,
            "embed_tokens": 0,
            "reranker_calls": 0,
        }

    def add_llm(self, input_tokens: int, output_tokens: int) -> None:
        s = self._state()
        s["llm_calls"] += 1
        s["llm_input_tokens"] += input_tokens
        s["llm_output_tokens"] += output_tokens

    def add_embed(self, tokens: int) -> None:
        s = self._state()
        s["embed_calls"] += 1
        s["embed_tokens"] += tokens

    def add_reranker(self, n_docs: int = 1) -> None:
        self._state()["reranker_calls"] += n_docs

    def snapshot(self) -> CostSnapshot:
        s = self._state()
        return CostSnapshot(**s)


cost_tracker = _CostTracker()

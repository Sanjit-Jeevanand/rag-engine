from typing import Any

from openai import OpenAI

from rag_engine.cost import cost_tracker

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def complete(
    messages: list[dict[str, str]],
    *,
    model: str = "gpt-4o-mini",
    max_tokens: int = 512,
    system: str = "",
) -> str:
    client = _get_client()
    msgs: list[dict[str, Any]] = (
        [{"role": "system", "content": system}, *messages] if system else list(messages)
    )
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=msgs,  # type: ignore[arg-type]
    )
    if response.usage:
        cost_tracker.add_llm(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )
    return response.choices[0].message.content or ""

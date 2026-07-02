from collections.abc import AsyncIterator
from typing import Any, cast

from openai import AsyncOpenAI, AsyncStream, OpenAI
from openai.types.chat import ChatCompletionChunk

from rag_engine.cost import cost_tracker

_client: OpenAI | None = None
_async_client: AsyncOpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def _get_async_client() -> AsyncOpenAI:
    global _async_client
    if _async_client is None:
        _async_client = AsyncOpenAI()
    return _async_client


def complete(
    messages: list[dict[str, str]],
    *,
    model: str = "gpt-5-mini",
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


async def stream_complete(
    messages: list[dict[str, str]],
    *,
    model: str = "gpt-5-mini",
    max_tokens: int = 512,
    system: str = "",
) -> AsyncIterator[str]:
    client = _get_async_client()
    msgs: list[dict[str, Any]] = (
        [{"role": "system", "content": system}, *messages] if system else list(messages)
    )
    stream = cast(
        AsyncStream[ChatCompletionChunk],
        await client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=msgs,  # type: ignore[arg-type]
            stream=True,
        ),
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta

from __future__ import annotations

from typing import AsyncIterator

from anthropic import AsyncAnthropic

from llm.provider import LLMProvider

DEFAULT_MODEL = "claude-opus-4-8"


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str = ""):
        self._client = AsyncAnthropic(api_key=api_key)
        self.model = model

    async def chat(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> str:
        system, conv = _split_system(messages)
        kwargs: dict = dict(
            model=self.model,
            max_tokens=max_tokens,
            messages=conv,
        )
        if system:
            kwargs["system"] = system
        stream = await self._client.messages.stream(**kwargs).__aenter__()
        msg = await stream.get_final_message()
        await stream.__aexit__(None, None, None)
        return "".join(
            block.text for block in msg.content if hasattr(block, "text")
        )

    async def stream(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        system, conv = _split_system(messages)
        kwargs: dict = dict(
            model=self.model,
            max_tokens=max_tokens,
            messages=conv,
        )
        if system:
            kwargs["system"] = system
        async with self._client.messages.stream(**kwargs) as s:
            async for text in s.text_stream:
                yield text


def _split_system(messages: list[dict]) -> tuple[str | None, list[dict]]:
    system = next((m["content"] for m in messages if m["role"] == "system"), None)
    conv = [m for m in messages if m["role"] != "system"]
    return system, conv

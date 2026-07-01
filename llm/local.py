from __future__ import annotations

from typing import AsyncIterator

from openai import AsyncOpenAI

from llm.provider import LLMProvider

DEFAULT_MODEL = "llama3"
DEFAULT_BASE_URL = "http://localhost:11434"


class LocalProvider(LLMProvider):
    """
    Local LLM via Ollama's OpenAI-compatible API.
    Requires Ollama running at base_url with the target model pulled.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
    ):
        self._client = AsyncOpenAI(
            api_key="ollama",
            base_url=f"{base_url.rstrip('/')}/v1",
        )
        self.model = model

    async def chat(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> str:
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    async def stream(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        async with await self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        ) as s:
            async for chunk in s:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

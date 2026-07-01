from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator


class LLMProvider(ABC):
    """
    Minimal async LLM interface.
    Messages follow OpenAI-style format: list of {"role": ..., "content": ...}.
    Roles: "system", "user", "assistant".
    """

    @abstractmethod
    async def chat(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> str:
        """Send messages and return the full response text."""
        ...

    async def stream(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Stream response text chunks. Default: wraps chat() as a single yield."""
        result = await self.chat(messages, max_tokens=max_tokens, temperature=temperature)
        yield result

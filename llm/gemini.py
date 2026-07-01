from __future__ import annotations

from typing import AsyncIterator

import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from llm.provider import LLMProvider

DEFAULT_MODEL = "gemini-1.5-pro"


class GeminiProvider(LLMProvider):
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str = ""):
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model)
        self.model = model

    async def chat(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> str:
        prompt, config = _build_request(messages, max_tokens, temperature)
        response = await self._model.generate_content_async(prompt, generation_config=config)
        return response.text

    async def stream(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        prompt, config = _build_request(messages, max_tokens, temperature)
        async for chunk in await self._model.generate_content_async(
            prompt, generation_config=config, stream=True
        ):
            if chunk.text:
                yield chunk.text


def _build_request(
    messages: list[dict], max_tokens: int, temperature: float
) -> tuple[list[dict], GenerationConfig]:
    """
    Convert OpenAI-style messages to Gemini format.
    System messages are prepended to the first user turn.
    """
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    conv = [m for m in messages if m["role"] != "system"]

    gemini_messages = []
    for i, m in enumerate(conv):
        role = "user" if m["role"] == "user" else "model"
        content = m["content"]
        # Prepend system instruction to the first user message
        if role == "user" and i == 0 and system_parts:
            content = "\n\n".join(system_parts) + "\n\n" + content
        gemini_messages.append({"role": role, "parts": [content]})

    config = GenerationConfig(max_output_tokens=max_tokens, temperature=temperature)
    return gemini_messages, config

from __future__ import annotations

from functools import lru_cache

from llm.provider import LLMProvider


@lru_cache(maxsize=1)
def get_llm() -> LLMProvider | None:
    """
    Return the configured LLM provider, or None if no provider is set.
    Reads from settings at first call and caches the instance.
    """
    from config.settings import settings

    provider = settings.llm_provider.lower()

    if provider == "anthropic":
        if not settings.anthropic_api_key:
            return None
        from llm.anthropic import AnthropicProvider
        return AnthropicProvider(
            model=settings.llm_model or "claude-opus-4-8",
            api_key=settings.anthropic_api_key,
        )

    if provider == "openai":
        if not settings.openai_api_key:
            return None
        from llm.openai import OpenAIProvider
        return OpenAIProvider(
            model=settings.llm_model or "gpt-4o",
            api_key=settings.openai_api_key,
        )

    if provider == "gemini":
        if not settings.gemini_api_key:
            return None
        from llm.gemini import GeminiProvider
        return GeminiProvider(
            model=settings.llm_model or "gemini-1.5-pro",
            api_key=settings.gemini_api_key,
        )

    if provider == "local":
        from llm.local import LocalProvider
        return LocalProvider(
            model=settings.llm_model or settings.local_llm_model,
            base_url=settings.local_llm_base_url,
        )

    return None

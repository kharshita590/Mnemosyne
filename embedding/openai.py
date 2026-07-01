from __future__ import annotations

from openai import AsyncOpenAI

from config.settings import settings
from embedding.provider import EmbeddingProvider

_client: AsyncOpenAI | None = None


def get_openai_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


class OpenAIEmbedder(EmbeddingProvider):
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
    ):
        self.model_name = model
        self.dimensions = dimensions

    async def embed(self, text: str) -> list[float]:
        text = text.replace("\n", " ").strip()
        response = await get_openai_client().embeddings.create(
            input=[text],
            model=self.model_name,
        )
        return response.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        cleaned = [t.replace("\n", " ").strip() for t in texts]
        response = await get_openai_client().embeddings.create(
            input=cleaned,
            model=self.model_name,
        )
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]

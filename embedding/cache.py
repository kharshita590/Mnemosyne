from __future__ import annotations

import hashlib

from embedding.provider import EmbeddingProvider
from storage.redis import cache_embedding, get_cached_embedding


class CachedEmbedder(EmbeddingProvider):
    """Wraps any EmbeddingProvider with a Redis cache keyed by SHA-256 of the text."""

    def __init__(self, provider: EmbeddingProvider, ttl_hours: int = 24):
        self.provider = provider
        self.model_name = provider.model_name
        self.dimensions = provider.dimensions
        self.ttl_hours = ttl_hours

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    async def embed(self, text: str) -> list[float]:
        key = self._hash(text)
        cached = await get_cached_embedding(key)
        if cached is not None:
            return cached
        embedding = await self.provider.embed(text)
        await cache_embedding(key, embedding, self.ttl_hours)
        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float] | None] = [None] * len(texts)
        missing_indices: list[int] = []

        for i, text in enumerate(texts):
            cached = await get_cached_embedding(self._hash(text))
            if cached is not None:
                results[i] = cached
            else:
                missing_indices.append(i)

        if missing_indices:
            missing_texts = [texts[i] for i in missing_indices]
            new_embeddings = await self.provider.embed_batch(missing_texts)
            for idx, embedding in zip(missing_indices, new_embeddings):
                results[idx] = embedding
                await cache_embedding(self._hash(texts[idx]), embedding, self.ttl_hours)

        return results  # type: ignore[return-value]

from __future__ import annotations

import asyncio
from functools import lru_cache

from embedding.provider import EmbeddingProvider


@lru_cache(maxsize=1)
def _load_model(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


class LocalEmbedder(EmbeddingProvider):
    """
    CPU/GPU local embedder using sentence-transformers.
    Good models: BAAI/bge-m3 (1024 dims), intfloat/multilingual-e5-large (1024 dims)
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", dimensions: int = 384):
        self.model_name = model_name
        self.dimensions = dimensions

    async def embed(self, text: str) -> list[float]:
        loop = asyncio.get_event_loop()
        model = _load_model(self.model_name)
        embedding = await loop.run_in_executor(
            None, lambda: model.encode(text, normalize_embeddings=True).tolist()
        )
        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        loop = asyncio.get_event_loop()
        model = _load_model(self.model_name)
        embeddings = await loop.run_in_executor(
            None, lambda: model.encode(texts, normalize_embeddings=True, batch_size=32).tolist()
        )
        return embeddings

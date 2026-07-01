from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    model_name: str
    dimensions: int

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Embed a single string. Returns a float list of length self.dimensions."""
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Default: sequential. Override for true batching."""
        results = []
        for t in texts:
            results.append(await self.embed(t))
        return results

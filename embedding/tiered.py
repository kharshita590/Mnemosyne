from __future__ import annotations

from config.settings import settings
from embedding.cache import CachedEmbedder
from embedding.local import LocalEmbedder
from embedding.openai import OpenAIEmbedder
from embedding.provider import EmbeddingProvider
from storage.models import MemoryTier


def get_embedder_for_tier(tier: MemoryTier) -> EmbeddingProvider:
    """
    Routing logic:
    - local provider: always uses sentence-transformers (no API key required)
    - openai provider, working/episodic: small fast model (cost matters)
    - openai provider, long_term/semantic: large accurate model (quality matters)
    """
    if settings.embedding_provider == "local":
        base: EmbeddingProvider = LocalEmbedder(
            model_name=settings.local_embedding_model,
            dimensions=settings.local_embedding_dimensions,
        )
    elif tier in (MemoryTier.WORKING, MemoryTier.EPISODIC):
        base = OpenAIEmbedder(model=settings.embedding_model, dimensions=settings.embedding_dimensions)
    else:
        base = OpenAIEmbedder(model="text-embedding-3-large", dimensions=3072)
    return CachedEmbedder(base)

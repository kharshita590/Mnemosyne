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
        # ponytail: OpenAI embeddings match table schema vector dimensions; resize db column if 3072 dims needed
        base = OpenAIEmbedder(model=settings.embedding_model, dimensions=settings.embedding_dimensions)
    return CachedEmbedder(base)

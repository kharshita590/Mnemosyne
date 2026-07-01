from __future__ import annotations

import uuid

from storage.models import Memory, MemoryTier
from storage.pg import search_by_vector


async def dense_search(
    user_id: uuid.UUID,
    query_embedding: list[float],
    tier: MemoryTier | None = None,
    limit: int = 20,
) -> list[tuple[Memory, float]]:
    """
    Cosine similarity search via pgvector.
    Returns (memory, similarity_score) pairs, score in [0, 1].
    """
    return await search_by_vector(
        user_id=user_id,
        query_embedding=query_embedding,
        tier=tier,
        limit=limit,
    )

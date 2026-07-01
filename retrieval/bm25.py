from __future__ import annotations

import uuid

from storage.models import Memory, MemoryTier
from storage.pg import search_by_text


async def bm25_search(
    user_id: uuid.UUID,
    query: str,
    tier: MemoryTier | None = None,
    limit: int = 20,
) -> list[tuple[Memory, float]]:
    """
    BM25-style full-text search via PostgreSQL tsvector.
    Returns (memory, ts_rank) pairs.
    """
    return await search_by_text(
        user_id=user_id,
        query=query,
        tier=tier,
        limit=limit,
    )

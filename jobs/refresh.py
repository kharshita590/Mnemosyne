from __future__ import annotations

import asyncio

from sqlalchemy import select, update

from config.logging import logger
from embedding.tiered import get_embedder_for_tier
from storage.models import Memory, MemoryTier
from storage.pg import AsyncSessionLocal


async def re_embed_tier(tier: MemoryTier, new_model: str) -> None:
    """
    Re-embed all memories in a tier using a new embedding model.
    Run after updating the embedding model in settings.
    """
    embedder = get_embedder_for_tier(tier)
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Memory).where(
                Memory.tier == tier.value,
                Memory.embedding_model != new_model,
            ).limit(1000)
        )
        memories = list(result.scalars().all())

    logger.info("re_embed_start", tier=tier.value, count=len(memories), model=new_model)
    for mem in memories:
        new_embedding = await embedder.embed(mem.content)
        async with AsyncSessionLocal() as session:
            await session.execute(
                update(Memory)
                .where(Memory.id == mem.id)
                .values(embedding=new_embedding, embedding_model=new_model)
            )
            await session.commit()

    logger.info("re_embed_done", tier=tier.value, count=len(memories))


if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else "text-embedding-3-small"
    asyncio.run(re_embed_tier(MemoryTier.EPISODIC, model))

from __future__ import annotations

import asyncio

from sqlalchemy import text

from storage.pg import engine


async def rebuild_indexes() -> None:
    """Rebuild HNSW and GIN indexes. Run during maintenance window."""
    async with engine.begin() as conn:
        await conn.execute(text("REINDEX INDEX CONCURRENTLY ix_memories_embedding_hnsw"))
        await conn.execute(text("REINDEX INDEX CONCURRENTLY ix_memories_content_tsv"))


if __name__ == "__main__":
    asyncio.run(rebuild_indexes())

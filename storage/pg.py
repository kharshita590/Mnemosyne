from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from config.settings import settings
from storage.models import Base, IngestJob, IngestJobStatus, Memory, MemoryTier, User

engine = create_async_engine(
    settings.database_url,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,   # validates connections before use; auto-reconnects after Postgres restart
    echo=False,
)

AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def create_tables() -> None:
    """Only for tests. Use Alembic migrations in production."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def upsert_user(external_id: str) -> User:
    async with get_session() as session:
        result = await session.execute(
            select(User).where(User.external_id == external_id)
        )
        user = result.scalar_one_or_none()
        if user is None:
            user = User(external_id=external_id)
            session.add(user)
            await session.flush()
        return user


async def insert_memory(
    *,
    user_id: uuid.UUID,
    content: str,
    embedding: list[float],
    tier: MemoryTier,
    source_conversation_id: str | None = None,
    embedding_model: str = "text-embedding-3-small",
    chunk_strategy: str | None = None,
    extra: dict | None = None,
) -> Memory:
    async with get_session() as session:
        memory = Memory(
            user_id=user_id,
            content=content,
            embedding=embedding,
            tier=tier.value,
            source_conversation_id=source_conversation_id,
            embedding_model=embedding_model,
            chunk_strategy=chunk_strategy,
            extra=extra or {},
        )
        session.add(memory)
        await session.flush()
        return memory


async def search_by_vector(
    *,
    user_id: uuid.UUID,
    query_embedding: list[float],
    tier: MemoryTier | None = None,
    limit: int = 20,
) -> list[tuple[Memory, float]]:
    """Returns (memory, cosine_similarity) pairs sorted by similarity descending."""
    async with get_session() as session:
        distance_col = Memory.embedding.cosine_distance(query_embedding).label("distance")
        stmt = (
            select(Memory, distance_col)
            .where(Memory.user_id == user_id)
            .where(Memory.embedding.is_not(None))
            .order_by(distance_col)
            .limit(limit)
        )
        if tier is not None:
            stmt = stmt.where(Memory.tier == tier.value)

        rows = (await session.execute(stmt)).all()
        return [(row.Memory, 1.0 - float(row.distance)) for row in rows]


async def search_by_text(
    *,
    user_id: uuid.UUID,
    query: str,
    tier: MemoryTier | None = None,
    limit: int = 20,
) -> list[tuple[Memory, float]]:
    """BM25-style full-text search via PostgreSQL tsvector."""
    async with get_session() as session:
        ts_query = func.plainto_tsquery("english", query)
        rank_col = func.ts_rank_cd(Memory.content_tsv, ts_query).label("rank")
        stmt = (
            select(Memory, rank_col)
            .where(Memory.user_id == user_id)
            .where(Memory.content_tsv.op("@@")(ts_query))
            .order_by(rank_col.desc())
            .limit(limit)
        )
        if tier is not None:
            stmt = stmt.where(Memory.tier == tier.value)

        rows = (await session.execute(stmt)).all()
        return [(row.Memory, float(row.rank)) for row in rows]


async def get_memory_by_id(memory_id: uuid.UUID) -> Memory | None:
    async with get_session() as session:
        result = await session.execute(
            select(Memory).where(Memory.id == memory_id)
        )
        return result.scalar_one_or_none()


async def update_access_stats(memory_id: uuid.UUID) -> None:
    async with get_session() as session:
        await session.execute(
            update(Memory)
            .where(Memory.id == memory_id)
            .values(
                access_count=Memory.access_count + 1,
                last_accessed_at=datetime.now(timezone.utc),
            )
        )


async def update_decay_weight(memory_id: uuid.UUID, new_weight: float) -> None:
    async with get_session() as session:
        await session.execute(
            update(Memory)
            .where(Memory.id == memory_id)
            .values(decay_weight=new_weight)
        )


async def get_memories_for_decay_sweep(
    tier: MemoryTier,
    batch_size: int = 500,
    sweep_interval_hours: int = 24,
) -> list[Memory]:
    """Return memories whose decay weight may have drifted since last sweep.

    Only returns memories not accessed within sweep_interval_hours, ordered
    by oldest-accessed first so the most stale memories are prioritised.
    Memories accessed recently will be recalculated on next retrieval via
    record_access(), so they don't need the sweep.
    """
    async with get_session() as session:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=sweep_interval_hours)
        stmt = (
            select(Memory)
            .where(Memory.tier == tier.value)
            # Only sweep memories not recently accessed (avoids rescanning active ones)
            .where(
                (Memory.last_accessed_at < cutoff) | Memory.last_accessed_at.is_(None)
            )
            .order_by(Memory.last_accessed_at.asc().nulls_first())
            .limit(batch_size)
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())


async def delete_memory(memory_id: uuid.UUID) -> None:
    async with get_session() as session:
        memory = await session.get(Memory, memory_id)
        if memory:
            await session.delete(memory)


async def update_memory_content(memory_id: uuid.UUID, new_content: str, new_embedding: list[float]) -> None:
    async with get_session() as session:
        await session.execute(
            update(Memory)
            .where(Memory.id == memory_id)
            .values(content=new_content, embedding=new_embedding)
        )


async def find_similar_memories(
    *,
    user_id: uuid.UUID,
    embedding: list[float],
    threshold: float,
    limit: int,
    exclude_tiers: list[str] | None = None,
) -> list[tuple[Memory, float]]:
    """Returns memories with cosine similarity >= threshold, sorted by similarity desc.

    Uses the native <=> operator in the WHERE clause so pgvector's HNSW index
    can apply the distance filter during the index scan rather than post-fetch.
    """
    async with get_session() as session:
        max_distance = 1.0 - threshold
        distance_col = Memory.embedding.cosine_distance(embedding).label("distance")
        stmt = (
            select(Memory, distance_col)
            .where(Memory.user_id == user_id)
            .where(Memory.embedding.is_not(None))
            .where(Memory.embedding.cosine_distance(embedding) <= max_distance)
            .order_by(distance_col)
            .limit(limit)
        )
        if exclude_tiers:
            stmt = stmt.where(Memory.tier.notin_(exclude_tiers))
        rows = (await session.execute(stmt)).all()
        return [(row.Memory, 1.0 - float(row.distance)) for row in rows]


async def delete_memories_bulk(
    *,
    user_id: uuid.UUID,
    tier: str | None = None,
    before_date: datetime | None = None,
    memory_ids: list[uuid.UUID] | None = None,
) -> int:
    """Bulk delete memories by filters. Returns count of deleted rows.

    When memory_ids is provided (from a topic search), tier and before_date
    act as additional narrowing filters ON THOSE IDs — not as independent
    filters across all memories. This makes multi-filter behaviour predictable.
    """
    async with get_session() as session:
        stmt = delete(Memory).where(Memory.user_id == user_id)

        if memory_ids is not None:
            # Topic search already resolved to specific IDs; apply optional narrowing
            stmt = stmt.where(Memory.id.in_(memory_ids))
            if tier:
                stmt = stmt.where(Memory.tier == tier)
            if before_date:
                stmt = stmt.where(Memory.created_at < before_date)
        else:
            # No topic — tier and before_date are the primary filters
            if tier:
                stmt = stmt.where(Memory.tier == tier)
            if before_date:
                stmt = stmt.where(Memory.created_at < before_date)

        result = await session.execute(stmt)
        return result.rowcount


async def create_ingest_job(
    *, user_external_id: str, content: str, conversation_id: str | None
) -> IngestJob:
    """Persist a pending job *before* the background ingestion task is scheduled,
    so a server crash/restart leaves a record instead of silently losing the write."""
    async with get_session() as session:
        job = IngestJob(
            user_external_id=user_external_id,
            content=content,
            conversation_id=conversation_id or None,
            status=IngestJobStatus.PENDING.value,
        )
        session.add(job)
        await session.flush()
        return job


async def mark_ingest_job(job_id: uuid.UUID, *, status: IngestJobStatus, error: str | None = None) -> None:
    async with get_session() as session:
        await session.execute(
            update(IngestJob)
            .where(IngestJob.id == job_id)
            .values(status=status.value, error=error)
        )


async def get_stuck_ingest_jobs(*, older_than_minutes: int = 15, limit: int = 200) -> list[IngestJob]:
    """Jobs still 'pending' past a reasonable ingestion time — the process that
    scheduled them likely crashed or was restarted before the task finished."""
    async with get_session() as session:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=older_than_minutes)
        stmt = (
            select(IngestJob)
            .where(IngestJob.status == IngestJobStatus.PENDING.value)
            .where(IngestJob.created_at < cutoff)
            .order_by(IngestJob.created_at.asc())
            .limit(limit)
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())


async def search_memories_by_topic(
    *,
    user_id: uuid.UUID,
    topic: str,
    limit: int = 100,
) -> list[Memory]:
    """Full-text search for bulk topic-based deletion."""
    async with get_session() as session:
        ts_query = func.plainto_tsquery("english", topic)
        stmt = (
            select(Memory)
            .where(Memory.user_id == user_id)
            .where(Memory.content_tsv.op("@@")(ts_query))
            .limit(limit)
        )
        rows = (await session.execute(stmt)).all()
        return [row[0] for row in rows]

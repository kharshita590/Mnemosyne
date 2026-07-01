from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class MemoryTier(str, Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    LONG_TERM = "long_term"
    SEMANTIC = "semantic"


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    external_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    memories: Mapped[list[Memory]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Memory(Base):
    __tablename__ = "memories"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    tier: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Dense vector — dims depend on embedding provider (384 for local, 1536 for OpenAI)
    embedding: Mapped[list[float]] = mapped_column(Vector(384), nullable=True)

    # BM25 — populated automatically by Postgres trigger, never set by app code
    content_tsv: Mapped[any] = mapped_column(TSVECTOR, nullable=True)

    source_conversation_id: Mapped[str | None] = mapped_column(String(255))
    embedding_model: Mapped[str | None] = mapped_column(String(100))
    chunk_strategy: Mapped[str | None] = mapped_column(String(50))
    extra: Mapped[dict | None] = mapped_column(JSONB, default=dict)

    access_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    decay_weight: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    last_accessed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    user: Mapped[User] = relationship(back_populates="memories")

    __table_args__ = (
        Index(
            "ix_memories_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_memories_content_tsv", "content_tsv", postgresql_using="gin"),
        Index("ix_memories_user_tier", "user_id", "tier"),
        Index("ix_memories_user_created", "user_id", "created_at"),
    )

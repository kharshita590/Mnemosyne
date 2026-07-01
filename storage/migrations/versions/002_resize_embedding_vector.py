"""resize embedding vector to 384 dims for local embedder

Revision ID: 002
Revises: 001
Create Date: 2026-06-28 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Null out existing embeddings before resizing — a 1536-dim vector cannot
    # be cast to 384-dim and would cause ALTER TABLE to fail on non-empty tables.
    op.execute("UPDATE memories SET embedding = NULL WHERE embedding IS NOT NULL")

    op.drop_index("ix_memories_embedding_hnsw", table_name="memories")
    op.alter_column(
        "memories",
        "embedding",
        type_=Vector(384),
        nullable=True,
    )
    op.create_index(
        "ix_memories_embedding_hnsw",
        "memories",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_index("ix_memories_embedding_hnsw", table_name="memories")
    op.alter_column(
        "memories",
        "embedding",
        type_=Vector(1536),
        postgresql_using="embedding::text::vector(1536)",
    )
    op.create_index(
        "ix_memories_embedding_hnsw",
        "memories",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )

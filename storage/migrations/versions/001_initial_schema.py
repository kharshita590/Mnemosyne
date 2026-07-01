"""initial schema

Revision ID: 001
Revises:
Create Date: 2026-06-24 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("external_id", sa.String(255), unique=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_table(
        "memories",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("tier", sa.String(20), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("embedding", Vector(1536), nullable=True),
        sa.Column("content_tsv", postgresql.TSVECTOR, nullable=True),
        sa.Column("source_conversation_id", sa.String(255), nullable=True),
        sa.Column("embedding_model", sa.String(100), nullable=True),
        sa.Column("chunk_strategy", sa.String(50), nullable=True),
        sa.Column("extra", postgresql.JSONB, nullable=True),
        sa.Column("access_count", sa.Integer, default=0, nullable=False),
        sa.Column("decay_weight", sa.Float, default=1.0, nullable=False),
        sa.Column("last_accessed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_index(
        "ix_memories_embedding_hnsw",
        "memories",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.create_index(
        "ix_memories_content_tsv",
        "memories",
        ["content_tsv"],
        postgresql_using="gin",
    )
    op.create_index("ix_memories_user_tier", "memories", ["user_id", "tier"])
    op.create_index("ix_memories_user_created", "memories", ["user_id", "created_at"])

    op.execute("""
        CREATE OR REPLACE FUNCTION memories_tsv_update() RETURNS trigger AS $$
        BEGIN
            NEW.content_tsv := to_tsvector('english', NEW.content);
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)
    op.execute("""
        CREATE TRIGGER memories_tsv_trigger
        BEFORE INSERT OR UPDATE ON memories
        FOR EACH ROW EXECUTE FUNCTION memories_tsv_update();
    """)


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS memories_tsv_trigger ON memories")
    op.execute("DROP FUNCTION IF EXISTS memories_tsv_update")
    op.drop_table("memories")
    op.drop_table("users")
    op.execute("DROP EXTENSION IF EXISTS vector")

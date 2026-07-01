from __future__ import annotations

import pytest

from agents.router import Intent, route
from storage.models import MemoryTier
from storage.pg import create_tables, search_by_vector, upsert_user

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
async def setup_db():
    await create_tables()


async def test_ingestion_stores_memories():
    user_id = "integration-test-user-001"
    content = "I prefer PostgreSQL with pgvector for vector similarity search."

    state = await route(
        Intent.INGEST,
        user_id=user_id,
        content=content,
        conversation_id="conv-001",
        tier=MemoryTier.EPISODIC,
    )

    assert state.error is None
    assert len(state.stored_ids) > 0


async def test_ingested_memory_is_retrievable():
    user_id = "integration-test-user-002"
    content = "The user loves LangGraph for building agentic workflows."

    await route(
        Intent.INGEST,
        user_id=user_id,
        content=content,
        conversation_id="conv-002",
        tier=MemoryTier.EPISODIC,
    )

    state = await route(
        Intent.RETRIEVE,
        user_id=user_id,
        query="What framework does the user prefer for agents?",
    )

    assert state.error is None
    assert len(state.final_memories) > 0
    contents = [m.content for m in state.final_memories]
    assert any("LangGraph" in c or "langgraph" in c.lower() for c in contents)

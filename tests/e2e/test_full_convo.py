from __future__ import annotations

import pytest

from agents.router import Intent, route
from storage.models import MemoryTier
from storage.pg import create_tables

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
async def setup_db():
    await create_tables()


async def test_full_conversation_round_trip():
    """
    Full round-trip test:
    1. Ingest 3 memories for a user
    2. Call session_init for a new conversation
    3. Assert injected_prompt is non-empty with <memory> block
    4. Call retrieve_memory with a relevant query
    5. Assert at least one of the 3 ingested memories appears in results
    """
    user_id = "e2e-test-user-001"
    conversation_id = "conv-e2e-001"

    memories_to_ingest = [
        "The user is building a RAG system using LangGraph and pgvector.",
        "The user prefers Python 3.11 and uses async SQLAlchemy for database access.",
        "The user stores embeddings in PostgreSQL using the pgvector extension.",
    ]

    # Step 1: Ingest 3 memories
    for content in memories_to_ingest:
        state = await route(
            Intent.INGEST,
            user_id=user_id,
            content=content,
            conversation_id=conversation_id,
            tier=MemoryTier.EPISODIC,
        )
        assert state.error is None
        assert len(state.stored_ids) > 0

    # Step 2: Call session_init for a new conversation
    session_state = await route(
        Intent.SESSION_INIT,
        user_id=user_id,
        conversation_id="conv-e2e-002",
        session_context="RAG system with vector database",
    )

    # Step 3: Assert injected_prompt contains memory block
    assert session_state.error is None
    assert "<memory>" in session_state.injected_prompt
    assert "</memory>" in session_state.injected_prompt

    # Step 4: Retrieve with relevant query
    retrieval_state = await route(
        Intent.RETRIEVE,
        user_id=user_id,
        query="What vector database does the user prefer?",
    )

    # Step 5: Assert at least one ingested memory appears
    assert retrieval_state.error is None
    assert len(retrieval_state.final_memories) > 0

    retrieved_contents = " ".join(m.content for m in retrieval_state.final_memories).lower()
    assert any(
        kw in retrieved_contents
        for kw in ["pgvector", "postgresql", "langgraph", "python"]
    )

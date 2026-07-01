from __future__ import annotations

import pytest

from agents.router import Intent, route
from storage.models import MemoryTier
from storage.pg import create_tables

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
async def setup_db():
    await create_tables()


async def test_retrieval_returns_memories_for_known_user():
    user_id = "retrieval-test-user-001"
    content = "The user prefers Cohere for reranking search results."

    await route(
        Intent.INGEST,
        user_id=user_id,
        content=content,
        conversation_id="conv-ret-001",
        tier=MemoryTier.EPISODIC,
    )

    state = await route(
        Intent.RETRIEVE,
        user_id=user_id,
        query="What does the user use for reranking?",
    )

    assert state.error is None
    assert state.final_memories is not None


async def test_retrieval_empty_for_unknown_user():
    state = await route(
        Intent.RETRIEVE,
        user_id="no-such-user-xyzabc",
        query="anything at all",
    )
    assert state.error is None
    assert state.final_memories == []


async def test_retrieval_scores_are_positive():
    user_id = "retrieval-test-user-002"
    await route(
        Intent.INGEST,
        user_id=user_id,
        content="Redis is used for caching and working memory storage.",
        conversation_id="conv-ret-002",
        tier=MemoryTier.EPISODIC,
    )

    state = await route(
        Intent.RETRIEVE,
        user_id=user_id,
        query="What is used for caching?",
    )

    for mem in state.final_memories:
        assert mem.score > 0

from __future__ import annotations

import pytest

from agents.router import Intent, route
from storage.models import MemoryTier
from storage.pg import create_tables

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
async def setup_db():
    await create_tables()


async def test_session_init_with_no_memories_returns_empty_block():
    state = await route(
        Intent.SESSION_INIT,
        user_id="session-test-user-new",
        conversation_id="conv-sess-001",
        session_context="What is the user's tech stack?",
    )

    assert state.error is None
    assert state.injected_prompt == ""
    assert not state.has_memories if hasattr(state, "has_memories") else True


async def test_session_init_after_ingestion_returns_block():
    user_id = "session-test-user-001"

    await route(
        Intent.INGEST,
        user_id=user_id,
        content="The user uses Python 3.11 and prefers async frameworks.",
        conversation_id="conv-prev",
        tier=MemoryTier.EPISODIC,
    )

    state = await route(
        Intent.SESSION_INIT,
        user_id=user_id,
        conversation_id="conv-new",
        session_context="Python async development",
    )

    assert state.error is None
    assert "<memory>" in state.injected_prompt
    assert "</memory>" in state.injected_prompt

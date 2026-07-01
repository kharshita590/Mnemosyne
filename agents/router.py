from __future__ import annotations

from enum import Enum

from agents.ingestion.graph import ingestion_graph
from agents.ingestion.state import IngestionState
from agents.retrieval.graph import retrieval_graph
from agents.retrieval.state import RetrievalState
from agents.session.graph import session_graph
from agents.session.state import SessionState
from storage.models import MemoryTier


class Intent(str, Enum):
    INGEST = "ingest"
    RETRIEVE = "retrieve"
    SESSION_INIT = "session_init"
    CHAT = "chat"


async def route(
    intent: Intent,
    user_id: str,
    *,
    content: str | None = None,
    query: str | None = None,
    user_message: str | None = None,
    conversation_id: str = "",
    tier: MemoryTier = MemoryTier.EPISODIC,
    session_context: str = "",
):
    """
    Entry point for all agent calls. Routes to the correct subgraph
    based on intent and returns the final state.

    Intent.CHAT combines SESSION_INIT + generation in one call:
    pass user_message to get a memory-augmented LLM response.
    """
    if intent == Intent.INGEST:
        assert content is not None
        return await ingestion_graph.ainvoke(
            IngestionState(
                user_id=user_id,
                content=content,
                conversation_id=conversation_id,
                tier=tier,
            )
        )

    if intent == Intent.RETRIEVE:
        assert query is not None
        return await retrieval_graph.ainvoke(
            RetrievalState(user_id=user_id, query=query)
        )

    if intent in (Intent.SESSION_INIT, Intent.CHAT):
        msg = user_message or ""
        return await session_graph.ainvoke(
            SessionState(
                user_id=user_id,
                conversation_id=conversation_id,
                session_context=session_context or msg,
                user_message=msg,
            )
        )

    raise ValueError(f"Unknown intent: {intent}")

from __future__ import annotations

from agents.retrieval.state import RetrievalState
from agents.retrieval.graph import retrieval_graph
from agents.session.state import SessionState
from config.logging import logger
from llm.factory import get_llm
from memory.types import MemoryItem
from storage.models import MemoryTier
from storage.redis import append_to_working_memory, get_working_memory

_BASE_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to the user's personal memory. "
    "Use the memory context provided to give accurate, personalized responses. "
    "Only reference memories that are relevant to the current question."
)


async def load_working_memory_node(state: SessionState) -> SessionState:
    """Pull in-session context from Redis."""
    raw = await get_working_memory(state.user_id, state.conversation_id)
    state.working_memories = [
        MemoryItem(
            id=item.get("id", ""),
            content=item["content"],
            tier=MemoryTier.WORKING,
            score=1.0,
        )
        for item in raw
    ]
    return state


async def retrieve_episodic_node(state: SessionState) -> dict:
    """Retrieve recent episodic memories relevant to session context."""
    query = state.user_message or state.session_context
    if not query:
        return {}
    # retrieval_graph.ainvoke returns a dict (LangGraph's compiled runtime
    # representation), not a RetrievalState instance — index it, don't attribute-access it.
    result = await retrieval_graph.ainvoke(
        RetrievalState(
            user_id=state.user_id,
            query=query,
            tier=MemoryTier.EPISODIC,
        )
    )
    # Only return the field this node owns — load_working/retrieve_episodic/
    # retrieve_long_term run as parallel branches in the same graph superstep,
    # so returning the full state here would collide with the other branch's
    # update and raise INVALID_CONCURRENT_GRAPH_UPDATE.
    return {"episodic_memories": result.get("final_memories", [])}


async def retrieve_long_term_node(state: SessionState) -> dict:
    """Retrieve long-term stable facts relevant to session context."""
    query = state.user_message or state.session_context
    if not query:
        return {}
    result = await retrieval_graph.ainvoke(
        RetrievalState(
            user_id=state.user_id,
            query=query,
            tier=MemoryTier.LONG_TERM,
        )
    )
    return {"long_term_memories": result.get("final_memories", [])}


async def inject_node(state: SessionState) -> SessionState:
    """
    Assemble all retrieved memories into a structured memory block
    for prepending to the system prompt.

    Layer order: working (most recent) -> episodic -> long-term
    """
    lines = ["<memory>"]

    for m in state.working_memories:
        lines.append(f"  [working] {m.content}")

    for m in state.episodic_memories:
        lines.append(f"  [episodic] {m.content}")

    for m in state.long_term_memories:
        lines.append(f"  [long-term] {m.content}")

    lines.append("</memory>")

    if len(lines) > 2:  # more than just the tags
        state.injected_prompt = "\n".join(lines)
    else:
        state.injected_prompt = ""

    return state


async def generate_node(state: SessionState) -> SessionState:
    """
    Call the configured LLM with memory context + user message.
    Skipped silently if no LLM is configured or no user_message is provided.
    The injected_prompt is always available for callers that handle generation
    themselves (e.g., passing it as a system prompt prefix to their own LLM).
    """
    if not state.user_message:
        return state

    llm = get_llm()
    if llm is None:
        return state

    system = _BASE_SYSTEM_PROMPT
    if state.injected_prompt:
        system = state.injected_prompt + "\n\n" + system

    try:
        state.response = await llm.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": state.user_message},
            ]
        )
        logger.info("generated_response", chars=len(state.response))
    except Exception as e:
        state.error = f"generation failed: {e}"
        logger.error("generate_failed", error=str(e))

    return state


async def append_working_memory_node(state: SessionState) -> dict:
    """
    Persist this turn to Redis working memory so a later session_init in the
    same conversation_id has something to load, and end_session_tool has
    something to summarize/promote. Previously nothing in the codebase ever
    called append_to_working_memory, so working memory was always empty —
    this is the write side of that pipeline.

    Skipped when there's no user_message (a bare session_init call has
    nothing to log yet).
    """
    if not state.user_message:
        return {}
    await append_to_working_memory(
        state.user_id, state.conversation_id, {"role": "user", "content": state.user_message}
    )
    if state.response:
        await append_to_working_memory(
            state.user_id, state.conversation_id, {"role": "assistant", "content": state.response}
        )
    return {}

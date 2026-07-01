from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastmcp import FastMCP

from config.logging import configure_logging
from config.settings import settings
from memory.promote import promote_working_memory
from mcp_local.tools import forget_memories, forget_memory, ingest_memory, retrieve_memory, session_init

configure_logging()

mcp = FastMCP("Mnemosyne Memory Server")

# ---------------------------------------------------------------------------
# Auth guard
# ---------------------------------------------------------------------------

def _check_auth(api_key: str) -> dict | None:
    """Return an error dict if the key is wrong, else None."""
    if settings.mcp_api_key and settings.mcp_api_key != "changeme":
        if api_key != settings.mcp_api_key:
            return {"success": False, "error": "unauthorized"}
    return None


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def ingest_memory_tool(
    user_id: str,
    content: str,
    conversation_id: str = "",
    api_key: str = "",
) -> dict:
    """
    Store information from a conversation into memory.
    The tier (working/episodic/long_term/semantic) is automatically classified by the LLM.
    Ingestion is async — returns a job_id immediately.
    """
    if err := _check_auth(api_key):
        return err
    return await ingest_memory(user_id, content, conversation_id)


@mcp.tool()
async def retrieve_memory_tool(
    user_id: str,
    query: str,
    conversation_id: str = "",
    api_key: str = "",
) -> dict:
    """Retrieve memories relevant to the given query for a user."""
    if err := _check_auth(api_key):
        return err
    return await retrieve_memory(user_id, query, conversation_id)


@mcp.tool()
async def session_init_tool(
    user_id: str,
    conversation_id: str,
    session_context: str = "",
    api_key: str = "",
) -> dict:
    """
    Initialize a new conversation session.
    Returns a memory_block string to prepend to the system prompt.
    """
    if err := _check_auth(api_key):
        return err
    return await session_init(user_id, conversation_id, session_context)


@mcp.tool()
async def end_session_tool(
    user_id: str,
    conversation_id: str,
    api_key: str = "",
) -> dict:
    """
    Call at the end of a conversation to promote in-session working memory
    to long-term episodic storage. The conversation is summarised by the LLM
    and stored so future sessions can recall it. Clears the Redis working memory.
    """
    if err := _check_auth(api_key):
        return err
    return await promote_working_memory(user_id, conversation_id)


@mcp.tool()
async def forget_memory_tool(
    user_id: str,
    memory_id: str,
    api_key: str = "",
) -> dict:
    """Delete a specific memory by ID. User can only delete their own memories."""
    if err := _check_auth(api_key):
        return err
    return await forget_memory(user_id, memory_id)


@mcp.tool()
async def forget_memories_tool(
    user_id: str,
    topic: str = "",
    tier: str = "",
    before_date: str = "",
    confirm: bool = False,
    api_key: str = "",
) -> dict:
    """
    Bulk delete memories with privacy controls. At least one filter is required.

    Filters:
    - topic: delete all memories about this topic (e.g. "location", "health")
    - tier: delete all memories of a tier (working/episodic/long_term/semantic)
    - before_date: delete memories created before this date (ISO format: "2025-01-01")

    Always call first without confirm=True to preview what will be deleted.
    Then call again with confirm=True to actually delete.
    """
    if err := _check_auth(api_key):
        return err
    return await forget_memories(
        user_id=user_id,
        topic=topic or None,
        tier=tier or None,
        before_date=before_date or None,
        confirm=confirm,
    )


if __name__ == "__main__":
    mcp.run()

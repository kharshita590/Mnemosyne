from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from config.logging import configure_logging
from config.settings import settings
from memory.promote import promote_working_memory
from observability.health import check_health
from mcp_local.tools import (
    chat, forget_memories, forget_memory, ingest_memory, remember_turn, retrieve_memory,
    session_init,
)

configure_logging()

mcp = FastMCP("Mnemosyne Memory Server")


@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request) -> JSONResponse:
    result = await check_health()
    status_code = 200 if result["status"] == "healthy" else 503
    return JSONResponse(result, status_code=status_code)

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
async def chat_tool(
    user_id: str,
    conversation_id: str,
    user_message: str,
    api_key: str = "",
) -> dict:
    """
    Memory-augmented chat. Loads relevant memories, injects them into the
    system prompt, and generates a response with the server's configured LLM.
    Requires LLM_PROVIDER to be set — returns an error if no provider is configured.
    """
    if err := _check_auth(api_key):
        return err
    return await chat(user_id, conversation_id, user_message)


@mcp.tool()
async def remember_turn_tool(
    user_id: str,
    conversation_id: str,
    role: str,
    content: str,
    api_key: str = "",
) -> dict:
    """
    Log one conversation turn ('user' or 'assistant') into working memory.
    Call this once per turn if you generate responses yourself (rather than
    using chat_tool) so working memory accumulates and end_session_tool has
    something to promote to permanent storage.
    """
    if err := _check_auth(api_key):
        return err
    return await remember_turn(user_id, conversation_id, role, content)


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
    port = int(os.environ.get("PORT", 8080))
    mcp.run(transport="http", host="0.0.0.0", port=port)

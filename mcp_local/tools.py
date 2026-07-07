from __future__ import annotations

import asyncio
import uuid as _uuid
from datetime import datetime, timezone

from agents.router import Intent, route
from config.logging import logger
from mcp_local.sanitize import sanitize_input, sanitize_output
from storage.models import IngestJobStatus
from storage.pg import (
    create_ingest_job,
    delete_memories_bulk,
    delete_memory,
    get_memory_by_id,
    mark_ingest_job,
    search_memories_by_topic,
    upsert_user,
)
from storage.redis import append_to_working_memory


async def _run_ingest(job_id: _uuid.UUID, user_id: str, content: str, conversation_id: str) -> None:
    """
    Background task — runs the full ingestion pipeline without blocking the caller.
    The job row was already persisted as 'pending' before this was scheduled, so if
    the process dies here the job is recoverable via jobs/retry_stuck_ingests.py
    instead of silently vanishing.
    """
    try:
        await route(Intent.INGEST, user_id=user_id, content=content, conversation_id=conversation_id)
        await mark_ingest_job(job_id, status=IngestJobStatus.SUCCESS)
        logger.info("async_ingest_done", job_id=str(job_id))
    except Exception as e:
        await mark_ingest_job(job_id, status=IngestJobStatus.FAILED, error=str(e))
        logger.error("async_ingest_failed", job_id=str(job_id), error=str(e))


async def ingest_memory(
    user_id: str,
    content: str,
    conversation_id: str = "",
) -> dict:
    """
    Store new information from the current conversation.
    Returns immediately with a job_id — ingestion runs in the background.
    Tier is automatically classified by the LLM during ingestion.

    The job is persisted to Postgres as 'pending' *before* scheduling the
    background task. If the server crashes mid-ingestion, the row survives
    and jobs/retry_stuck_ingests.py can find and replay it.
    """
    clean_content = sanitize_input(content)
    job = await create_ingest_job(
        user_external_id=user_id, content=clean_content, conversation_id=conversation_id
    )

    # Scheduled only after the durable record exists — a crash before this line
    # loses nothing; a crash after it leaves a recoverable 'pending' row.
    asyncio.create_task(
        _run_ingest(job.id, user_id, clean_content, conversation_id)
    )

    return {"success": True, "job_id": str(job.id), "status": "ingesting"}


async def retrieve_memory(
    user_id: str,
    query: str,
    conversation_id: str = "",
) -> dict:
    """
    Retrieve memories relevant to a query.
    Called by the LLM client when it needs context.
    """
    clean_query = sanitize_input(query)

    state = await route(
        Intent.RETRIEVE,
        user_id=user_id,
        query=clean_query,
        conversation_id=conversation_id,
    )

    if state.get("error"):
        return {"success": False, "error": state["error"]}

    memories = [
        {
            "id": m["id"] if isinstance(m, dict) else m.id,
            "content": sanitize_output(m["content"] if isinstance(m, dict) else m.content),
            "tier": m["tier"].value if isinstance(m, dict) else m.tier.value,
            "score": round(m["score"] if isinstance(m, dict) else m.score, 4),
        }
        for m in state.get("final_memories", [])
    ]
    return {"success": True, "memories": memories}


async def session_init(
    user_id: str,
    conversation_id: str,
    session_context: str = "",
) -> dict:
    """
    Called at the start of a new conversation. Returns a memory block
    formatted for injection into the system prompt.
    """
    state = await route(
        Intent.SESSION_INIT,
        user_id=user_id,
        conversation_id=conversation_id,
        session_context=sanitize_input(session_context) if session_context else "",
    )

    if state.get("error"):
        return {"success": False, "error": state["error"]}

    return {
        "success": True,
        "memory_block": state.get("injected_prompt", ""),
        "has_memories": bool(state.get("injected_prompt")),
    }


async def remember_turn(
    user_id: str,
    conversation_id: str,
    role: str,
    content: str,
) -> dict:
    """
    Log one turn (a user message or an assistant reply) into Redis working
    memory for this conversation. For clients that generate their own
    responses (Claude Desktop, Cursor) and use session_init/retrieve_memory
    rather than chat_tool — call this once per turn so working memory
    actually accumulates, and end_session_tool has something to summarize
    into a permanent episodic memory.
    """
    if role not in ("user", "assistant"):
        return {"success": False, "error": "role must be 'user' or 'assistant'"}
    clean_content = sanitize_input(content)
    await append_to_working_memory(
        user_id, conversation_id, {"role": role, "content": clean_content}
    )
    return {"success": True}


async def chat(
    user_id: str,
    conversation_id: str,
    user_message: str,
) -> dict:
    """
    Memory-augmented chat: loads working/episodic/long-term memory, injects it
    into the system prompt, and generates a response with the configured LLM
    (llm/factory.get_llm — provider chosen via LLM_PROVIDER env var).

    Unlike session_init (which only returns the memory block for the caller's
    own LLM to use), this runs generation server-side. Returns success=False
    if no LLM provider is configured, since there is nothing to generate with.
    """
    clean_message = sanitize_input(user_message)

    state = await route(
        Intent.CHAT,
        user_id=user_id,
        conversation_id=conversation_id,
        user_message=clean_message,
    )

    if state.get("error"):
        return {"success": False, "error": state["error"]}

    response = state.get("response", "")
    if not response:
        return {"success": False, "error": "no LLM provider configured — set LLM_PROVIDER"}

    return {
        "success": True,
        "response": sanitize_output(response),
        "memory_block": state.get("injected_prompt", ""),
    }


async def forget_memory(user_id: str, memory_id: str) -> dict:
    """Delete a specific memory. User can only delete their own memories."""
    import uuid
    try:
        mid = uuid.UUID(memory_id)
    except ValueError:
        return {"success": False, "error": "invalid memory_id"}

    memory = await get_memory_by_id(mid)
    if memory is None:
        return {"success": False, "error": "memory not found"}

    user = await upsert_user(user_id)
    if str(memory.user_id) != str(user.id):
        return {"success": False, "error": "forbidden"}

    await delete_memory(mid)
    return {"success": True}


async def forget_memories(
    user_id: str,
    topic: str | None = None,
    tier: str | None = None,
    before_date: str | None = None,
    confirm: bool = False,
) -> dict:
    """
    Bulk delete memories with privacy controls.

    Filters (at least one required):
    - topic: delete all memories matching this topic (full-text search)
    - tier: delete all memories of this tier (working/episodic/long_term/semantic)
    - before_date: delete memories created before this ISO date (e.g. "2025-01-01")

    confirm must be True to actually delete — call without confirm=True first
    to see a preview of what will be deleted.
    """
    if not any([topic, tier, before_date]):
        return {"success": False, "error": "at least one filter (topic, tier, before_date) is required"}

    user = await upsert_user(user_id)

    parsed_date: datetime | None = None
    if before_date:
        try:
            parsed_date = datetime.fromisoformat(before_date).replace(tzinfo=timezone.utc)
        except ValueError:
            return {"success": False, "error": f"invalid before_date format, use ISO 8601 e.g. '2025-01-01'"}

    # Resolve topic → memory IDs via full-text search
    topic_ids: list | None = None
    preview_contents: list[str] = []
    if topic:
        matched = await search_memories_by_topic(
            user_id=user.id, topic=sanitize_input(topic)
        )
        topic_ids = [m.id for m in matched]
        preview_contents = [m.content for m in matched]
        if not topic_ids:
            return {"success": True, "deleted": 0, "message": "no memories matched topic"}

    if not confirm:
        # Preview mode — show what would be deleted without deleting
        if topic_ids is not None:
            # Apply the same narrowing filters to the preview
            from storage.models import MemoryTier
            previewed = [
                m for m in (await search_memories_by_topic(user_id=user.id, topic=sanitize_input(topic)))
                if (not tier or m.tier == tier)
                and (not parsed_date or m.created_at < parsed_date)
            ]
            preview_contents = [m.content for m in previewed]
        return {
            "success": True,
            "preview": True,
            "message": "set confirm=true to delete these memories",
            "filters": {"topic": topic, "tier": tier, "before_date": before_date},
            "matched_memories": preview_contents[:20],
            "matched_count": len(preview_contents),
        }

    deleted = await delete_memories_bulk(
        user_id=user.id,
        tier=tier,
        before_date=parsed_date,
        memory_ids=topic_ids,
    )
    return {"success": True, "deleted": deleted}

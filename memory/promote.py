from __future__ import annotations

from agents.router import Intent, route
from config.logging import logger
from prompts.registry import load_prompt
from llm.factory import get_llm
from storage.redis import clear_working_memory, get_working_memory


async def promote_working_memory(user_id: str, conversation_id: str) -> dict:
    """
    Called at session end (explicit or on TTL expiry via a background job).
    Summarises the Redis working-memory blob using the LLM and ingests the
    summary as an episodic memory so nothing is permanently lost.

    Returns: {"promoted": bool, "summary": str | None}
    """
    items = await get_working_memory(user_id, conversation_id)
    if not items:
        return {"promoted": False, "summary": None}

    conversation_text = "\n".join(
        item.get("content", "") for item in items if item.get("content")
    )
    if not conversation_text.strip():
        return {"promoted": False, "summary": None}

    llm = get_llm()
    summary: str

    if llm is not None:
        try:
            prompt = load_prompt("session.yaml", "summarize_session").format(
                conversation=conversation_text
            )
            summary = await llm.chat([{"role": "user", "content": prompt}], max_tokens=512)
            summary = summary.strip()
        except Exception as e:
            logger.warning("promote_summarize_failed", error=str(e))
            # Fall back to raw concatenation so we don't lose the data
            summary = conversation_text[:2000]
    else:
        summary = conversation_text[:2000]

    if not summary:
        return {"promoted": False, "summary": None}

    try:
        await route(
            Intent.INGEST,
            user_id=user_id,
            content=summary,
            conversation_id=conversation_id,
            # classify_node will assign the right tier; episodic is the expected default
        )
        await clear_working_memory(user_id, conversation_id)
        logger.info("working_memory_promoted", user_id=user_id, conversation_id=conversation_id)
        return {"promoted": True, "summary": summary}
    except Exception as e:
        logger.error("promote_ingest_failed", error=str(e))
        return {"promoted": False, "summary": summary}

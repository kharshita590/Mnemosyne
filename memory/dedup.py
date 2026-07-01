from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from enum import Enum

from config.logging import logger
from config.settings import settings
from llm.factory import get_llm
from prompts.registry import load_prompt
from storage.models import Memory
from storage.pg import delete_memory, find_similar_memories, update_memory_content


class ConflictAction(str, Enum):
    KEEP = "KEEP"
    UPDATE = "UPDATE"
    DUPLICATE = "DUPLICATE"


@dataclass
class DeduplicationResult:
    should_store: bool          # False if the new memory is a duplicate
    updated_ids: list[str]      # IDs of existing memories that were updated
    deleted_ids: list[str]      # IDs of existing memories that were deleted (replaced)


async def resolve_dedup_and_conflicts(
    *,
    user_id: uuid.UUID,
    new_content: str,
    new_embedding: list[float],
) -> DeduplicationResult:
    """
    Check new content against existing similar memories.
    - If identical/near-identical → mark as duplicate, discard new memory.
    - If conflicting → ask LLM to resolve, update/delete old memory, store new one.
    - If genuinely new → store as-is.
    """
    candidates = await find_similar_memories(
        user_id=user_id,
        embedding=new_embedding,
        threshold=settings.dedup_similarity_threshold,
        limit=settings.dedup_candidates_limit,
        exclude_tiers=["working"],  # working memories are ephemeral, skip dedup
    )

    if not candidates:
        return DeduplicationResult(should_store=True, updated_ids=[], deleted_ids=[])

    llm = get_llm()
    if llm is None:
        # No LLM — fall back to pure similarity: treat highest match as duplicate
        top_mem, top_score = candidates[0]
        if top_score >= 0.98:
            logger.info("dedup_no_llm_duplicate", score=top_score, existing_id=str(top_mem.id))
            return DeduplicationResult(should_store=False, updated_ids=[], deleted_ids=[])
        return DeduplicationResult(should_store=True, updated_ids=[], deleted_ids=[])

    # Use ||| as separator so single | in content doesn't break parsing
    existing_ids_and_content = "\n".join(
        f"{mem.id}|||{mem.content}" for mem, _ in candidates
    )

    prompt = load_prompt("ingestion.yaml", "resolve_conflict").format(
        new_memory=new_content,
        existing_ids_and_content=existing_ids_and_content,
    )

    try:
        response = await llm.chat([{"role": "user", "content": prompt}], max_tokens=256)
    except Exception as e:
        logger.warning("conflict_resolution_llm_failed", error=str(e))
        return DeduplicationResult(should_store=True, updated_ids=[], deleted_ids=[])

    actions = _parse_conflict_response(response)
    logger.info("conflict_resolution", actions=actions)

    updated_ids: list[str] = []
    deleted_ids: list[str] = []
    is_duplicate = False

    candidate_map = {str(mem.id): mem for mem, _ in candidates}

    for mem_id_str, action in actions.items():
        mem = candidate_map.get(mem_id_str)
        if mem is None:
            continue

        if action == ConflictAction.DUPLICATE:
            is_duplicate = True
            logger.info("dedup_duplicate_discarded", existing_id=mem_id_str)

        elif action == ConflictAction.UPDATE:
            await update_memory_content(mem.id, new_content, new_embedding)
            updated_ids.append(mem_id_str)
            logger.info("dedup_updated_existing", existing_id=mem_id_str)
            # The new content is now in the existing row — no need to insert a new row
            is_duplicate = True

        # KEEP → do nothing, new memory will be stored normally

    return DeduplicationResult(
        should_store=not is_duplicate,
        updated_ids=updated_ids,
        deleted_ids=deleted_ids,
    )


_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
)
_VALID_ACTIONS = {a.value for a in ConflictAction}


def _parse_conflict_response(response: str) -> dict[str, ConflictAction]:
    """Parse LLM response lines of the form '<uuid>|||<ACTION>'.

    Defensive rules:
    - Accepts ||| separator (canonical) and falls back to scanning for UUID + action word.
    - Strips markdown backticks, bullets, numbering.
    - Ignores any line with no recognisable UUID.
    - Ignores any line with no recognisable action word.
    - Unknown action words → treat as KEEP (safest default).
    """
    actions: dict[str, ConflictAction] = {}

    for raw_line in response.strip().splitlines():
        line = raw_line.strip().lstrip("- •*`").strip()

        # Try canonical format first: uuid|||ACTION
        if "|||" in line:
            parts = line.split("|||", 1)
            mem_id_str = parts[0].strip().strip("`")
            action_str = parts[1].strip().upper().split()[0] if parts[1].strip() else ""
        else:
            # Fallback: find a UUID anywhere in the line, then look for action word
            match = _UUID_RE.search(line)
            if not match:
                continue
            mem_id_str = match.group(0)
            remainder = line[match.end():].upper()
            action_str = next((a for a in _VALID_ACTIONS if a in remainder), "")

        # Validate UUID
        try:
            uuid.UUID(mem_id_str)
        except ValueError:
            continue

        # Validate / coerce action
        if action_str in _VALID_ACTIONS:
            actions[mem_id_str] = ConflictAction(action_str)
        else:
            # Unrecognised action → safest choice is KEEP (store the new memory)
            logger.warning("conflict_unknown_action", line=raw_line, defaulting_to="KEEP")
            actions[mem_id_str] = ConflictAction.KEEP

    return actions

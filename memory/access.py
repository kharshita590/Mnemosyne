from __future__ import annotations

import uuid

from memory.decay import compute_decay_weight
from storage.models import Memory
from storage.pg import update_access_stats, update_decay_weight


async def record_access(memory: Memory) -> None:
    """
    Called every time a memory is surfaced to a user.
    Updates access_count, last_accessed_at, and recomputes decay_weight.
    """
    await update_access_stats(memory.id)

    new_weight = compute_decay_weight(
        created_at=memory.created_at,
        last_accessed_at=memory.last_accessed_at,
        access_count=memory.access_count + 1,  # +1 because update hasn't committed yet
    )
    await update_decay_weight(memory.id, new_weight)

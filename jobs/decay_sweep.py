from __future__ import annotations

import asyncio

from config.logging import logger
from memory.decay import compute_decay_weight
from storage.models import MemoryTier
from storage.pg import get_memories_for_decay_sweep, update_decay_weight


async def run_decay_sweep() -> None:
    """
    Cron job: recalculates decay_weight for episodic and long_term memories.
    Run nightly. In production, deploy as AKS CronJob pointing at this entrypoint.
    """
    for tier in (MemoryTier.EPISODIC, MemoryTier.LONG_TERM):
        memories = await get_memories_for_decay_sweep(tier, batch_size=500)
        updated = 0
        for mem in memories:
            new_weight = compute_decay_weight(
                created_at=mem.created_at,
                last_accessed_at=mem.last_accessed_at,
                access_count=mem.access_count,
            )
            if abs(new_weight - mem.decay_weight) > 0.001:
                await update_decay_weight(mem.id, new_weight)
                updated += 1
        logger.info("decay_sweep", tier=tier.value, total=len(memories), updated=updated)


if __name__ == "__main__":
    asyncio.run(run_decay_sweep())

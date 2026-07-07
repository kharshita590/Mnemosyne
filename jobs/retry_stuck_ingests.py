from __future__ import annotations

import asyncio

from agents.router import Intent, route
from config.logging import logger
from storage.models import IngestJobStatus
from storage.pg import get_stuck_ingest_jobs, mark_ingest_job


async def run_retry_stuck_ingests(older_than_minutes: int = 15) -> None:
    """
    Cron job: replays ingestion jobs left in 'pending' because the process that
    scheduled them (mcp_local/tools.py:ingest_memory) crashed or restarted before
    the background task finished. Run every few minutes.
    """
    jobs = await get_stuck_ingest_jobs(older_than_minutes=older_than_minutes)
    retried = 0
    failed = 0
    for job in jobs:
        try:
            await route(
                Intent.INGEST,
                user_id=job.user_external_id,
                content=job.content,
                conversation_id=job.conversation_id or "",
            )
            await mark_ingest_job(job.id, status=IngestJobStatus.SUCCESS)
            retried += 1
        except Exception as e:
            await mark_ingest_job(job.id, status=IngestJobStatus.FAILED, error=str(e))
            failed += 1
            logger.warning("retry_stuck_ingest_failed", job_id=str(job.id), error=str(e))

    logger.info("retry_stuck_ingests", found=len(jobs), retried=retried, failed=failed)


if __name__ == "__main__":
    asyncio.run(run_retry_stuck_ingests())

from __future__ import annotations

from storage.pg import engine
from storage.redis import ping as redis_ping


async def check_health() -> dict:
    """
    Returns health status of all downstream dependencies.
    Use as /health endpoint in production.
    """
    results: dict[str, str] = {}

    # PostgreSQL
    try:
        async with engine.connect() as conn:
            await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        results["postgres"] = "ok"
    except Exception as e:
        results["postgres"] = f"error: {e}"

    # Redis
    results["redis"] = "ok" if await redis_ping() else "error"

    results["status"] = "healthy" if all(v == "ok" for v in results.values()) else "degraded"
    return results

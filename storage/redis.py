from __future__ import annotations

import json
import time
from datetime import timedelta

import redis.asyncio as aioredis

from config.logging import logger
from config.settings import settings

_client: aioredis.Redis | None = None

# Circuit breaker state
_circuit_open = False          # True = Redis is considered down, skip calls
_circuit_open_until: float = 0  # epoch seconds until we retry
_CIRCUIT_COOLDOWN = 30          # seconds before retrying after failure


def get_client() -> aioredis.Redis:
    global _client
    if _client is None:
        _client = aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
    return _client


def _is_circuit_open() -> bool:
    global _circuit_open, _circuit_open_until
    if _circuit_open and time.monotonic() > _circuit_open_until:
        _circuit_open = False  # cooldown elapsed — allow one probe
    return _circuit_open


def _trip_circuit() -> None:
    global _circuit_open, _circuit_open_until
    if not _circuit_open:
        logger.warning("redis_circuit_tripped", retry_in_seconds=_CIRCUIT_COOLDOWN)
    _circuit_open = True
    _circuit_open_until = time.monotonic() + _CIRCUIT_COOLDOWN


def _reset_circuit() -> None:
    global _circuit_open
    if _circuit_open:
        logger.info("redis_circuit_reset")
    _circuit_open = False


def _key(user_id: str, conversation_id: str) -> str:
    return f"wm:{user_id}:{conversation_id}"


async def set_working_memory(
    user_id: str,
    conversation_id: str,
    memories: list[dict],
    ttl: timedelta | None = None,
) -> None:
    if _is_circuit_open():
        return
    if ttl is None:
        ttl = timedelta(seconds=settings.working_memory_ttl)
    try:
        client = get_client()
        await client.setex(
            _key(user_id, conversation_id),
            int(ttl.total_seconds()),
            json.dumps(memories),
        )
        _reset_circuit()
    except Exception as e:
        logger.warning("redis_set_failed", error=str(e))
        _trip_circuit()


async def get_working_memory(user_id: str, conversation_id: str) -> list[dict]:
    if _is_circuit_open():
        return []
    try:
        raw = await get_client().get(_key(user_id, conversation_id))
        _reset_circuit()
        if raw is None:
            return []
        return json.loads(raw)
    except Exception as e:
        logger.warning("redis_get_failed", error=str(e))
        _trip_circuit()
        return []


_APPEND_SCRIPT = """
local key = KEYS[1]
local new_item = ARGV[1]
local ttl = tonumber(ARGV[2])
local raw = redis.call('GET', key)
local list
if raw then
    list = cjson.decode(raw)
else
    list = {}
end
table.insert(list, cjson.decode(new_item))
local encoded = cjson.encode(list)
redis.call('SETEX', key, ttl, encoded)
return encoded
"""


async def append_to_working_memory(
    user_id: str,
    conversation_id: str,
    new_memory: dict,
    ttl: timedelta | None = None,
) -> None:
    """Atomically append one memory to the working-memory list via a Lua script.

    The Lua script runs as a single Redis command so concurrent appends
    for the same conversation cannot overwrite each other.
    """
    if _is_circuit_open():
        return
    if ttl is None:
        ttl = timedelta(seconds=settings.working_memory_ttl)
    try:
        client = get_client()
        await client.eval(
            _APPEND_SCRIPT,
            1,
            _key(user_id, conversation_id),
            json.dumps(new_memory),
            int(ttl.total_seconds()),
        )
        _reset_circuit()
    except Exception as e:
        logger.warning("redis_append_failed", error=str(e))
        _trip_circuit()


async def clear_working_memory(user_id: str, conversation_id: str) -> None:
    if _is_circuit_open():
        return
    try:
        await get_client().delete(_key(user_id, conversation_id))
        _reset_circuit()
    except Exception as e:
        logger.warning("redis_delete_failed", error=str(e))
        _trip_circuit()


async def cache_embedding(text_hash: str, embedding: list[float], ttl_hours: int = 24) -> None:
    """Cache an embedding vector keyed by hash of its source text."""
    if _is_circuit_open():
        return
    try:
        await get_client().setex(
            f"emb:{text_hash}",
            ttl_hours * 3600,
            json.dumps(embedding),
        )
        _reset_circuit()
    except Exception as e:
        logger.warning("redis_cache_embedding_failed", error=str(e))
        _trip_circuit()


async def get_cached_embedding(text_hash: str) -> list[float] | None:
    if _is_circuit_open():
        return None
    try:
        raw = await get_client().get(f"emb:{text_hash}")
        _reset_circuit()
        if raw is None:
            return None
        return json.loads(raw)
    except Exception as e:
        logger.warning("redis_get_embedding_failed", error=str(e))
        _trip_circuit()
        return None


async def ping() -> bool:
    try:
        result = await get_client().ping()
        _reset_circuit()
        return result
    except Exception:
        _trip_circuit()
        return False

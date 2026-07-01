from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from langfuse import Langfuse

from config.settings import settings

_langfuse: Langfuse | None = None


def get_langfuse() -> Langfuse:
    global _langfuse
    if _langfuse is None:
        _langfuse = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
            enabled=bool(settings.langfuse_public_key),
        )
    return _langfuse


def trace_node(node_name: str):
    """
    Decorator for LangGraph nodes. Creates a Langfuse span around the node
    and records the input state, output state, and any errors.
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(state: Any) -> Any:
            lf = get_langfuse()
            span = lf.span(name=node_name)
            try:
                result = await fn(state)
                span.end(output={"error": getattr(result, "error", None)})
                return result
            except Exception as e:
                span.end(level="ERROR", status_message=str(e))
                raise
        return wrapper
    return decorator

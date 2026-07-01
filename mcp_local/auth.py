from __future__ import annotations

from config.settings import settings


def verify_api_key(api_key: str) -> bool:
    """
    Simple static API key check. In production, replace with JWT validation
    or an OAuth2 token introspection call.
    """
    return api_key == settings.mcp_api_key


def extract_user_id(headers: dict) -> str | None:
    """Extract the caller's user identifier from MCP request headers."""
    return headers.get("x-user-id") or headers.get("X-User-Id")

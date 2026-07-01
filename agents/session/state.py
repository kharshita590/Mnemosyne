from __future__ import annotations

from dataclasses import dataclass, field

from memory.types import MemoryItem


@dataclass
class SessionState:
    user_id: str
    conversation_id: str
    # Brief description of what the user is doing in this session
    session_context: str = ""
    # The user's current message to answer
    user_message: str = ""
    working_memories: list[MemoryItem] = field(default_factory=list)
    episodic_memories: list[MemoryItem] = field(default_factory=list)
    long_term_memories: list[MemoryItem] = field(default_factory=list)
    injected_prompt: str = ""
    # LLM response; empty string if no LLM is configured
    response: str = ""
    error: str | None = None

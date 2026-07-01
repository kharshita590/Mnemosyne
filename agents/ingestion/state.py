from __future__ import annotations

from dataclasses import dataclass, field

from chunking.base import Chunk
from storage.models import MemoryTier


@dataclass
class IngestionState:
    user_id: str
    content: str
    conversation_id: str
    tier: MemoryTier = MemoryTier.EPISODIC
    chunks: list[Chunk] = field(default_factory=list)
    embeddings: list[list[float]] = field(default_factory=list)
    stored_ids: list[str] = field(default_factory=list)
    # LLM-extracted facts stored separately as long_term memories
    extracted_facts: list[str] = field(default_factory=list)
    # Structured entity metadata — stored in extra JSONB on every memory row
    entities: dict = field(default_factory=dict)
    error: str | None = None

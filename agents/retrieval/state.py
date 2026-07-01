from __future__ import annotations

from dataclasses import dataclass, field

from memory.types import MemoryItem
from storage.models import MemoryTier


@dataclass
class RetrievalState:
    user_id: str
    query: str
    # When set, restricts search to this tier (used by session graph for focused pulls).
    # For general retrieve_memory_tool calls, leave None to search all tiers unified.
    tier: MemoryTier | None = None
    # Expanded queries (original + paraphrases) for multi-query retrieval
    expanded_queries: list[str] = field(default_factory=list)
    query_embeddings: list[list[float]] = field(default_factory=list)
    # Keep single alias for backward compat with nodes that write to it
    query_embedding: list[float] = field(default_factory=list)
    dense_results: list[tuple] = field(default_factory=list)
    bm25_results: list[tuple] = field(default_factory=list)
    fused_results: list[tuple] = field(default_factory=list)
    reranked_results: list[tuple] = field(default_factory=list)
    final_memories: list[MemoryItem] = field(default_factory=list)
    error: str | None = None

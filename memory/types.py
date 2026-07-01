from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from storage.models import MemoryTier


@dataclass
class MemoryItem:
    """Runtime representation of a retrieved memory, used inside agent state."""
    id: str
    content: str
    tier: MemoryTier
    score: float
    created_at: datetime | None = None
    last_accessed_at: datetime | None = None
    decay_weight: float = 1.0
    metadata: dict = field(default_factory=dict)

    def to_context_line(self) -> str:
        tier_label = {
            MemoryTier.WORKING: "working",
            MemoryTier.EPISODIC: "episodic",
            MemoryTier.LONG_TERM: "long-term",
            MemoryTier.SEMANTIC: "semantic",
        }.get(self.tier, self.tier.value)
        return f"[{tier_label}] {self.content}"

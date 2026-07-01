from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class RetrievalMetrics:
    """Computed after each retrieval run. Log to Langfuse or stdout."""
    query: str
    dense_count: int
    bm25_count: int
    fused_count: int
    reranked_count: int
    final_count: int
    latency_ms: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def recall_at_k(self, relevant_ids: set[str], returned_ids: list[str], k: int = 5) -> float:
        """
        Fraction of relevant memories found in the top-k results.
        recall@k = |relevant ∩ top-k| / |relevant|
        """
        if not relevant_ids:
            return 0.0
        top_k = set(returned_ids[:k])
        return len(relevant_ids & top_k) / len(relevant_ids)

    def mrr(self, relevant_ids: set[str], returned_ids: list[str]) -> float:
        """Reciprocal rank of the first relevant result. 0.0 if none found."""
        for rank, mid in enumerate(returned_ids, start=1):
            if mid in relevant_ids:
                return 1.0 / rank
        return 0.0

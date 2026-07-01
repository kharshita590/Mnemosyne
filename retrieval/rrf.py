from __future__ import annotations

from storage.models import Memory

# Standard RRF constant — 60 is the empirically-derived default
RRF_K = 60


def reciprocal_rank_fusion(
    *ranked_lists: list[tuple[Memory, float]],
    k: int = RRF_K,
) -> list[tuple[Memory, float]]:
    """
    Fuse multiple ranked result lists into one using Reciprocal Rank Fusion.

    Formula: score(d) = sum(1 / (k + rank(d)))
    where rank is 1-based position in each list.

    Returns merged list sorted by RRF score descending.
    Score differences between lists don't matter — only rank positions do.
    This handles the scale mismatch between cosine similarity (0-1) and BM25 scores.
    """
    scores: dict[str, float] = {}
    memories: dict[str, Memory] = {}

    for ranked_list in ranked_lists:
        for rank, (memory, _) in enumerate(ranked_list, start=1):
            mem_id = str(memory.id)
            scores[mem_id] = scores.get(mem_id, 0.0) + 1.0 / (k + rank)
            memories[mem_id] = memory

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [(memories[mid], scores[mid]) for mid in sorted_ids]

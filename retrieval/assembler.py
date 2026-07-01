from __future__ import annotations

import tiktoken

from config.settings import settings
from storage.models import Memory


def _count_tokens(text: str, model: str = "cl100k_base") -> int:
    enc = tiktoken.get_encoding(model)
    return len(enc.encode(text))


def assemble_context(
    candidates: list[tuple[Memory, float]],
    token_budget: int | None = None,
    deduplicate: bool = True,
) -> list[tuple[Memory, float]]:
    """
    Final stage of the retrieval pipeline:
    1. Deduplicate near-identical memories (exact content match)
    2. Apply decay weighting: final_score = reranker_score * decay_weight
    3. Trim to token budget

    Returns ordered list of (memory, final_score) pairs.
    """
    budget = token_budget or settings.context_token_budget

    if deduplicate:
        seen_content: set[str] = set()
        deduped = []
        for mem, score in candidates:
            if mem.content not in seen_content:
                seen_content.add(mem.content)
                deduped.append((mem, score))
        candidates = deduped

    # Apply decay weighting
    weighted = [
        (mem, score * mem.decay_weight)
        for mem, score in candidates
    ]
    weighted.sort(key=lambda x: x[1], reverse=True)

    # Trim to token budget
    result = []
    tokens_used = 0
    for mem, score in weighted:
        mem_tokens = _count_tokens(mem.content)
        if tokens_used + mem_tokens > budget:
            break
        result.append((mem, score))
        tokens_used += mem_tokens

    return result

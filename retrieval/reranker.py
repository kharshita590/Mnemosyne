from __future__ import annotations

import cohere

from config.settings import settings
from storage.models import Memory

_cohere_client: cohere.AsyncClientV2 | None = None


def get_cohere_client() -> cohere.AsyncClientV2:
    global _cohere_client
    if _cohere_client is None:
        _cohere_client = cohere.AsyncClientV2(api_key=settings.cohere_api_key)
    return _cohere_client


async def rerank(
    query: str,
    candidates: list[tuple[Memory, float]],
    top_n: int | None = None,
) -> list[tuple[Memory, float]]:
    """
    Rerank candidates using Cohere's cross-encoder.
    Falls back to original order if Cohere is unavailable or key is missing.
    Returns top_n results (or all if top_n is None).
    """
    if not settings.cohere_api_key or not candidates:
        return candidates[:top_n] if top_n else candidates

    documents = [mem.content for mem, _ in candidates]
    try:
        response = await get_cohere_client().rerank(
            model="rerank-v3.5",
            query=query,
            documents=documents,
            top_n=top_n or len(candidates),
        )
        reranked = [
            (candidates[result.index][0], result.relevance_score)
            for result in response.results
        ]
        return reranked
    except Exception:
        # Graceful degradation — return RRF order if reranker fails
        return candidates[:top_n] if top_n else candidates

from __future__ import annotations

import asyncio

from agents.retrieval.state import RetrievalState
from config.logging import logger
from config.settings import settings
from embedding.tiered import get_embedder_for_tier
from llm.factory import get_llm
from memory.access import record_access
from memory.types import MemoryItem
from prompts.registry import load_prompt
from retrieval.assembler import assemble_context
from retrieval.bm25 import bm25_search
from retrieval.dense import dense_search
from retrieval.reranker import rerank
from retrieval.rrf import reciprocal_rank_fusion
from storage.models import MemoryTier
from storage.pg import upsert_user

# Tier score multipliers — higher tiers get a boost so they surface over
# weakly-matching episodic entries when all tiers are searched together.
_TIER_WEIGHT: dict[str, float] = {
    MemoryTier.LONG_TERM.value: 1.20,
    MemoryTier.SEMANTIC.value: 1.15,
    MemoryTier.EPISODIC.value: 1.00,
    MemoryTier.WORKING.value: 0.90,
}


async def expand_query_node(state: RetrievalState) -> RetrievalState:
    """
    Generate 2-3 alternative phrasings of the query using the LLM.
    Falls back to the original query only if LLM is unavailable.
    Multi-query retrieval meaningfully improves recall for short queries.
    """
    state.expanded_queries = [state.query]
    llm = get_llm()
    if llm is None:
        return state
    try:
        prompt = load_prompt("retrieval.yaml", "expand_query").format(query=state.query)
        response = await llm.chat([{"role": "user", "content": prompt}], max_tokens=128)
        extras = [line.strip() for line in response.splitlines() if line.strip()]
        # Cap at 3 expansions to avoid overloading the embed batch
        state.expanded_queries = [state.query] + extras[:3]
        logger.info("query_expanded", original=state.query, count=len(state.expanded_queries))
    except Exception as e:
        logger.warning("expand_query_failed", error=str(e))
    return state


async def embed_query_node(state: RetrievalState) -> RetrievalState:
    """Embed all expanded queries in one batched call."""
    try:
        # Tier hint for embedder selection — use EPISODIC as default (mid-range model)
        tier = state.tier or MemoryTier.EPISODIC
        embedder = get_embedder_for_tier(tier)
        queries = state.expanded_queries or [state.query]
        state.query_embeddings = await embedder.embed_batch(queries)
        # Keep backward-compat alias pointing at the primary embedding
        state.query_embedding = state.query_embeddings[0]
    except Exception as e:
        state.error = f"query embedding failed: {e}"
    return state


async def hybrid_search_node(state: RetrievalState) -> RetrievalState:
    """
    Run dense + BM25 search for every expanded query in parallel.
    When tier is None, searches across ALL tiers (unified retrieval).
    """
    if state.error:
        return state
    try:
        user = await upsert_user(state.user_id)
        uid = user.id
        limit = settings.retrieval_top_k

        # Fan out: one dense search per expanded query embedding, all in parallel
        dense_tasks = [
            dense_search(user_id=uid, query_embedding=emb, tier=state.tier, limit=limit)
            for emb in state.query_embeddings
        ]
        # BM25 on the original query only (expansions don't help keyword search)
        bm25_task = bm25_search(user_id=uid, query=state.query, tier=state.tier, limit=limit)

        results = await asyncio.gather(*dense_tasks, bm25_task)
        bm25_results = results[-1]
        dense_results_per_query = results[:-1]

        # Merge dense results from all query expansions via RRF, then keep top-k
        if len(dense_results_per_query) > 1:
            merged_dense = reciprocal_rank_fusion(*dense_results_per_query)
        else:
            merged_dense = dense_results_per_query[0]

        state.dense_results = merged_dense[:limit]
        state.bm25_results = bm25_results
    except Exception as e:
        state.error = f"search failed: {e}"
    return state


async def fuse_node(state: RetrievalState) -> RetrievalState:
    """RRF fusion of dense and BM25 results, then apply tier score multipliers."""
    if state.error:
        return state
    fused = reciprocal_rank_fusion(state.dense_results, state.bm25_results)

    # Apply tier weight so long_term/semantic facts naturally surface higher
    # when all tiers are searched together. When a single tier is requested
    # the multiplier is uniform (1.0 or the tier's own weight) — harmless.
    weighted = [
        (mem, score * _TIER_WEIGHT.get(mem.tier, 1.0))
        for mem, score in fused
    ]
    weighted.sort(key=lambda x: x[1], reverse=True)
    state.fused_results = weighted
    return state


async def rerank_node(state: RetrievalState) -> RetrievalState:
    if state.error:
        return state
    state.reranked_results = await rerank(
        query=state.query,
        candidates=state.fused_results,
        top_n=settings.rerank_top_n * 3,
    )
    return state


async def assemble_node(state: RetrievalState) -> RetrievalState:
    if state.error:
        return state
    assembled = assemble_context(state.reranked_results)

    memories = []
    for mem, score in assembled:
        await record_access(mem)
        memories.append(MemoryItem(
            id=str(mem.id),
            content=mem.content,
            tier=MemoryTier(mem.tier),
            score=score,
            created_at=mem.created_at,
            last_accessed_at=mem.last_accessed_at,
            decay_weight=mem.decay_weight,
        ))
    state.final_memories = memories
    return state

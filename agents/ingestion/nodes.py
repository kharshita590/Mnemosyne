from __future__ import annotations

import json

from agents.ingestion.state import IngestionState
from chunking.selector import chunk_text
from config.logging import logger
from embedding.tiered import get_embedder_for_tier
from llm.factory import get_llm
from memory.dedup import resolve_dedup_and_conflicts
from prompts.registry import load_prompt
from storage.models import MemoryTier
from storage.pg import insert_memory, upsert_user


async def extract_node(state: IngestionState) -> IngestionState:
    """
    Use LLM to extract key facts from raw content.
    Skipped silently if no LLM is configured.
    """
    llm = get_llm()
    if llm is None:
        return state
    try:
        prompt = load_prompt("ingestion.yaml", "extract_facts").format(content=state.content)
        response = await llm.chat([{"role": "user", "content": prompt}], max_tokens=1024)
        facts = [line.strip() for line in response.splitlines() if line.strip()]
        state.extracted_facts = facts
        logger.info("extracted_facts", count=len(facts))
    except Exception as e:
        logger.warning("extract_facts_failed", error=str(e))
    return state


_VALID_TIERS = {t.value for t in MemoryTier}
_ENTITY_KEYS = {"people", "places", "technologies", "preferences", "topics"}


async def extract_entities_node(state: IngestionState) -> IngestionState:
    """
    Extract structured named entities from content using the LLM.
    Stored in every memory row's extra JSONB column. Used for:
    - Entity-based dedup (same entity = likely conflict, not just similar text)
    - Topic-based forgetting (delete all memories mentioning 'location')
    - Future graph-style traversal
    """
    llm = get_llm()
    if llm is None:
        return state
    try:
        prompt = load_prompt("ingestion.yaml", "extract_entities").format(content=state.content)
        response = await llm.chat([{"role": "user", "content": prompt}], max_tokens=256)
        # Strip markdown fences if present
        clean = response.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(clean)
        # Keep only expected keys, ignore hallucinated keys
        state.entities = {k: v for k, v in parsed.items() if k in _ENTITY_KEYS}
        logger.info("extracted_entities", keys=list(state.entities.keys()))
    except Exception as e:
        logger.warning("extract_entities_failed", error=str(e))
        state.entities = {}
    return state


async def classify_node(state: IngestionState) -> IngestionState:
    """Use LLM to classify the appropriate memory tier for this content.

    Scans every word in the response for a valid tier name so phrasing like
    'This should be long_term' still works. Falls back to EPISODIC on failure.
    """
    llm = get_llm()
    if llm is None:
        return state
    try:
        prompt = load_prompt("ingestion.yaml", "classify_tier").format(content=state.content)
        response = await llm.chat([{"role": "user", "content": prompt}], max_tokens=32)
        words = response.strip().lower().split()
        classified = next((w for w in words if w in _VALID_TIERS), None)
        if classified:
            state.tier = MemoryTier(classified)
            logger.info("classified_tier", tier=state.tier.value)
        else:
            logger.warning("classify_tier_no_match", response=response, fallback="episodic")
    except Exception as e:
        logger.warning("classify_tier_failed", error=str(e))
    return state


async def chunk_node(state: IngestionState) -> IngestionState:
    """Splits content using the best strategy for the classified tier."""
    try:
        state.chunks = chunk_text(state.content, tier=state.tier)
        logger.info("chunked", count=len(state.chunks), strategy=state.chunks[0].strategy if state.chunks else "none")
    except Exception as e:
        state.error = f"chunking failed: {e}"
    return state


async def embed_node(state: IngestionState) -> IngestionState:
    """Embeds all chunks in a single batched call."""
    if state.error or not state.chunks:
        return state
    try:
        embedder = get_embedder_for_tier(state.tier)
        texts = [c.text for c in state.chunks]
        state.embeddings = await embedder.embed_batch(texts)
    except Exception as e:
        state.error = f"embedding failed: {e}"
    return state


async def store_node(state: IngestionState) -> IngestionState:
    """Persists chunks + embeddings to pgvector, with dedup and conflict resolution."""
    if state.error or not state.embeddings:
        return state
    try:
        user = await upsert_user(state.user_id)
        ids = []
        skipped = 0
        for chunk, embedding in zip(state.chunks, state.embeddings):
            dedup = await resolve_dedup_and_conflicts(
                user_id=user.id,
                new_content=chunk.text,
                new_embedding=embedding,
            )
            ids.extend(dedup.updated_ids)

            if not dedup.should_store:
                skipped += 1
                continue

            memory = await insert_memory(
                user_id=user.id,
                content=chunk.text,
                embedding=embedding,
                tier=state.tier,
                source_conversation_id=state.conversation_id,
                chunk_strategy=chunk.strategy,
                extra={"entities": state.entities} if state.entities else {},
            )
            ids.append(str(memory.id))

        state.stored_ids = ids
        logger.info("stored", count=len(ids), skipped=skipped, tier=state.tier.value)
    except Exception as e:
        state.error = f"store failed: {e}"
    return state


async def store_facts_node(state: IngestionState) -> IngestionState:
    """
    Persists LLM-extracted facts as long_term memories.
    Each fact is embedded individually and stored at LONG_TERM tier.
    Skipped if no facts were extracted.
    """
    if not state.extracted_facts:
        return state
    try:
        user = await upsert_user(state.user_id)
        embedder = get_embedder_for_tier(MemoryTier.LONG_TERM)
        embeddings = await embedder.embed_batch(state.extracted_facts)
        stored = 0
        for fact, embedding in zip(state.extracted_facts, embeddings):
            dedup = await resolve_dedup_and_conflicts(
                user_id=user.id,
                new_content=fact,
                new_embedding=embedding,
            )
            state.stored_ids.extend(dedup.updated_ids)

            if not dedup.should_store:
                continue

            memory = await insert_memory(
                user_id=user.id,
                content=fact,
                embedding=embedding,
                tier=MemoryTier.LONG_TERM,
                source_conversation_id=state.conversation_id,
                chunk_strategy="llm_extract",
                extra={"entities": state.entities} if state.entities else {},
            )
            state.stored_ids.append(str(memory.id))
            stored += 1
        logger.info("stored_facts", stored=stored, total=len(state.extracted_facts))
    except Exception as e:
        logger.warning("store_facts_failed", error=str(e))
    return state

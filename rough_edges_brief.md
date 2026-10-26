# Rough Edges Brief: Mnemosyne

## 1. Async Ingest Race
**Problem**: `ingest_memory_tool` returns immediately with a job_id while ingestion runs in the background. Querying right after ingest may miss the newly stored memory because the background task hasn't completed yet.

**Proposed Fix**:
- Add a synchronous variant `ingest_memory_sync_tool` that runs the same ingestion pipeline but waits for completion before returning (or times out after a short deadline, e.g., 2 seconds).
- Keep the existing async tool for fire‑and‑forget use cases.
- Implementation: reuse `_run_ingest` internally; call it directly instead of scheduling via `asyncio.create_task`.
- Trade‑off: slightly higher latency for the synchronous endpoint, but eliminates the race for clients that need immediate consistency.

**Files to touch**:
- `mcp_local/tools.py`: add `ingest_memory_sync` function.
- Optionally expose via MCP server if needed.

## 2. Low and Tightly Clustered Vector Scores (~0.016–0.02)
**Problem**: Cosine similarity scores returned by vector search are uniformly low and narrowly distributed, making it hard to distinguish relevance by absolute score (though ranking remains correct). This suggests the stored/query vectors are not normalized, so cosine similarity does not reflect pure directional similarity.

**Proposed Fix**:
- Modify both embedding providers (`LocalEmbedder` and `OpenAIEmbedder`) to L2‑normalize their output vectors before returning them.
- This ensures that stored and query vectors are unit length, so cosine similarity equals the dot product and scores will spread over a more meaningful range (typically [0,1] for positively correlated vectors).
- Implementation: in each embedder’s `embed_batch` method, compute the norm and divide each vector by its norm (adding a small epsilon to avoid division by zero).
- Trade‑off: negligible CPU overhead; requires re‑embedding existing memories after changing the code (or accept that old vectors remain unnormalized and scores for them will be off until they are overwritten).

**Files to touch**:
- `embedding/local.py`: modify `LocalEmbedder.embed_batch`.
- `embedding/openai.py`: modify `OpenAIEmbedder.embed_batch`.
- Consider adding a migration script to renormalize existing embeddings (optional).

## 3. Turn Recall Weakness (Remembered Turns Not Ranking in Pure Retrieval)
**Problem**: Working memory turns (stored in Redis) are only surfaced during retrieval when a `conversation_id` is explicitly provided. In pure retrieval calls (no `conversation_id`) they are ignored, so users must remember to pass the conversation ID to get turn‑level boosting.

**Proposed Fix**:
- Maintain a per‑user “last conversation ID” in Redis (updated whenever `chat_tool` or `session_init_tool` is called with a `conversation_id`).
- In `retrieve_memory_tool`, if no `conversation_id` is given, fall back to the user’s last conversation ID (if any) to fetch working memory turns and prepend them with a score of 1.0.
- This brings the working‑memory boost to retrieval without requiring the caller to track the conversation ID, while still keeping the working memory scoped to the most recent dialogue.
- Trade‑off: adds a small Redis read per retrieval; assumes the user’s intent aligns with their most recent conversation (reasonable for chat‑style usage).

**Files to touch**:
- `mcp_local/tools.py`:
  - In `chat` and `session_init`, after successful routing, store the used `conversation_id` in a Redis key like `ws:{user_id}:last_conv`.
  - In `retrieve_memory`, if `conversation_id` is empty, attempt to read that key and use it to fetch working memory.
- `storage/redis.py`: add helper functions `set_last_conversation_id` and `get_last_conversation_id` (or reuse existing Redis commands).

---
**Next Steps**:
1. Apply the embedding normalization fix and re‑ingest a small test set to verify score spread improves.
2. Add the synchronous ingest tool and test the race condition is eliminated.
3. Implement the last‑conversation fallback and verify turns appear in retrieval without explicit ID.

All changes are backward‑compatible and follow the lazy‑senior‑developer principle (minimal code, reuse existing dependencies).
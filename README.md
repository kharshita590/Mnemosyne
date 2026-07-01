# Mnemosyne

A persistent memory layer for LLM applications. Stores, retrieves, and injects relevant context from past conversations so your AI assistant remembers who you are and what you care about.

Exposes a [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server that any MCP-compatible client (Claude Desktop, Cursor, etc.) can connect to.

---

## How it works

```
User message
     │
     ▼
┌─────────────────────────────────────────────────────┐
│                   Session graph                     │
│  Load working memory (Redis)                        │
│       → Retrieve episodic memories (pgvector)       │
│       → Retrieve long-term facts (pgvector)         │
│       → Inject memory block into system prompt      │
│       → Generate LLM response with memory context  │
└─────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────┐
│                  Ingestion graph                    │
│  LLM extracts key facts from conversation           │
│       → Auto-select chunk strategy                  │
│       → Embed chunks (OpenAI text-embedding-3)      │
│       → Store in pgvector + extracted facts as      │
│         long-term memories                          │
└─────────────────────────────────────────────────────┘
```

### Memory tiers

| Tier | Storage | TTL | Purpose |
|---|---|---|---|
| `working` | Redis | 4 hours | In-session context (current conversation) |
| `episodic` | Postgres | Decays over time | Specific events and interactions |
| `long_term` | Postgres | Persistent | Stable facts, preferences, habits |
| `semantic` | Postgres | Persistent | General knowledge extracted across episodes |

### Retrieval pipeline

Every query runs through **hybrid search** (dense vector + BM25 keyword), **Reciprocal Rank Fusion**, optional **Cohere reranking**, and **token-budget trimming** before memories are returned.

---

## Prerequisites

- Docker and Docker Compose
- Python 3.11+
- `pip` or `uv`

---

## Setup

### 1. Clone and enter the directory

```bash
git clone <repo-url>
cd mnemosyne
```

### 2. Configure environment

```bash
cp .env.example .env
```

Open `.env` and fill in the required values:

```env
# Required: OpenAI key for embeddings (always needed)
OPENAI_API_KEY=sk-...

# Required: choose one LLM provider for generation
LLM_PROVIDER=gemini          # anthropic | openai | gemini | local
GEMINI_API_KEY=your-key      # or ANTHROPIC_API_KEY / OPENAI_API_KEY

# Optional but recommended: Cohere for reranking
COHERE_API_KEY=

# Optional: Langfuse for observability (get keys at cloud.langfuse.com)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...

# Optional: encryption at rest for stored memories
ENCRYPT_KEY=
```

**LLM provider options:**

| `LLM_PROVIDER` | Default model | Free tier |
|---|---|---|
| `gemini` | `gemini-1.5-pro` | Yes — set `LLM_MODEL=gemini-2.0-flash` for best free limits |
| `anthropic` | `claude-opus-4-8` | No |
| `openai` | `gpt-4o` | No |
| `local` | `llama3` via Ollama | Yes — requires Ollama running locally |

To override the model, set `LLM_MODEL=gemini-2.0-flash` (or any model ID).

### 3. Start infrastructure

```bash
make up
```

Starts Postgres (with pgvector) on port `5432` and Redis on port `6379` via Docker.

### 4. Install Python dependencies

```bash
pip install -e ".[dev]"
# or with uv:
uv pip install -e ".[dev]"
```

### 5. Run database migrations

```bash
make migrate
```

### 6. Start the MCP server

```bash
python mcp/server.py
```

---

## Using with Claude Desktop

Add Mnemosyne as an MCP server so Claude Desktop automatically gains persistent memory.

Open `~/Library/Application Support/Claude/claude_desktop_config.json` and add:

```json
{
  "mcpServers": {
    "mnemosyne": {
      "command": "python",
      "args": ["/absolute/path/to/mnemosyne/mcp/server.py"],
      "env": {
        "PYTHONPATH": "/absolute/path/to/mnemosyne"
      }
    }
  }
}
```

Restart Claude Desktop. A hammer icon will appear in the input bar with 4 memory tools available.

### What Claude can now do

Claude will call memory tools automatically. You can also prompt it directly:

```
"Remember that I prefer dark mode in all my projects."
"What do you know about my coding preferences?"
"Forget everything about my old job."
"What did we discuss last week about the auth system?"
```

### Available MCP tools

| Tool | What it does |
|---|---|
| `ingest_memory_tool` | Save information from the current conversation |
| `retrieve_memory_tool` | Search past memories by topic or query |
| `session_init_tool` | Load all relevant memories at the start of a conversation |
| `forget_memory_tool` | Delete a specific memory by ID |

---

## Using with Cursor

Add to `~/.cursor/mcp.json` (global) or `.cursor/mcp.json` (per project):

```json
{
  "mcpServers": {
    "mnemosyne": {
      "command": "python",
      "args": ["/absolute/path/to/mnemosyne/mcp/server.py"]
    }
  }
}
```

---

## Using the Python API directly

```python
import asyncio
from mnemosyne.agents.router import Intent, route

async def main():
    # Store a memory
    await route(
        Intent.INGEST,
        user_id="user-123",
        content="The user prefers Python over JavaScript and uses VS Code.",
        conversation_id="conv-abc",
        tier="long_term",
    )

    # Ask a question — memory is retrieved and injected automatically
    result = await route(
        Intent.CHAT,
        user_id="user-123",
        conversation_id="conv-abc",
        user_message="What editor should I set up for this project?",
    )
    print(result.response)
    # → "Based on your preference for VS Code, here's how to set it up..."

    # Retrieve raw memories without generating a response
    result = await route(
        Intent.RETRIEVE,
        user_id="user-123",
        query="programming language preferences",
    )
    for memory in result.final_memories:
        print(f"[{memory.tier.value}] {memory.content} (score: {memory.score:.3f})")

asyncio.run(main())
```

---

## Local LLM with Ollama (no API keys)

Run entirely offline:

```bash
# Install Ollama from https://ollama.com, then pull a model
ollama pull llama3
```

Set in `.env`:

```env
LLM_PROVIDER=local
LOCAL_LLM_BASE_URL=http://localhost:11434
LOCAL_LLM_MODEL=llama3
```

> Note: you still need `OPENAI_API_KEY` for embeddings. To remove that dependency, swap the embedding provider to a local model in [`embedding/tiered.py`](embedding/tiered.py) using the `LocalEmbedder` class.

---

## Background jobs

Run these on a schedule (cron or similar) to keep memories healthy:

```bash
# Decay old memories — reduces score of stale episodic/long-term memories
# Recommended: run nightly
python jobs/decay_sweep.py

# Rebuild BM25 keyword search index
# Recommended: run after bulk ingestion
python jobs/index_rebuild.py
```

---

## Development

```bash
make test    # run test suite
make lint    # ruff + mypy
make down    # stop Docker services
```

### Project structure

```
mnemosyne/
├── agents/          # LangGraph pipelines (ingestion, retrieval, session)
├── chunking/        # Auto-selecting text chunking strategies
├── config/          # Settings (pydantic-settings) and structured logging
├── embedding/       # OpenAI and local embedding providers, tiered routing
├── evals/           # Evaluation harness and golden set
├── infra/           # Dockerfile
├── jobs/            # Decay sweep and index rebuild cron jobs
├── llm/             # LLM providers: Anthropic, OpenAI, Gemini, Local (Ollama)
├── mcp/             # MCP server and tool definitions
├── memory/          # Memory types, decay, and access tracking
├── observability/   # Langfuse tracing, metrics, health check
├── prompts/         # YAML prompt templates
├── retrieval/       # Hybrid search, BM25, reranker, RRF, assembler
└── storage/         # Postgres (pgvector), Redis, S3, encryption, RLS
```

---

## Architecture notes

- **No LLM required to run** — if `LLM_PROVIDER` is unset or the API key is missing, ingestion still stores raw chunks and `session_init` still returns the memory block. You can pass the memory block to your own LLM.
- **Multi-user** — each call takes a `user_id`. Memories are isolated per user via Postgres row-level security.
- **Pluggable reranking** — Cohere reranking is optional. Without a `COHERE_API_KEY` the system falls back to RRF ordering.
- **Chunking is automatic** — the ingestion pipeline inspects content and picks the best strategy: fixed, recursive, sentence-based, or structural (for Markdown/code).
# Mnemosyne

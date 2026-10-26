# Mnemosyne

A production-ready persistent memory layer for LLM applications and coding agents. Stores, retrieves, and injects relevant context from past conversations so your AI assistants remember who you are, your architecture rules, habits, and preferences across sessions.

Exposes a standard [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server that connects natively to Claude Desktop, Cursor, OpenClaw, Windsurf, Cline, and any MCP client over HTTP or stdio.

GitHub Repository: [https://github.com/kharshita590/Mnemosyne](https://github.com/kharshita590/Mnemosyne)

---

## Features

- **Tiered Memory Architecture**:
  - `working`: Fast in-session scratchpad buffer via Redis (4h TTL).
  - `episodic`: Event-based interaction records in pgvector with temporal decay.
  - `long_term`: Persistent user habits, system facts, and preferences.
  - `semantic`: Cross-conversation distilled knowledge.
- **Hybrid Retrieval**: Dense vector search (pgvector HNSW) + BM25 keyword search (`tsvector`), combined via Reciprocal Rank Fusion (RRF) and optional Cohere cross-encoder reranking.
- **Zero-Config Local Embeddings**: Built-in 384-dimension `sentence-transformers` (`BAAI/bge-small-en-v1.5`) runs completely local by default without API costs or external keys.
- **Sync & Async Ingestion**: Background ingestion with durability tracking, plus synchronous ingestion (`sync=True`) to prevent read-after-write races.
- **Multi-Client MCP Protocol**: Supports both Server-Sent Events / HTTP (`http://localhost:8080/mcp`) and standard `stdio` transport.
- **Active Working Memory Bridging**: Automatically recalls active and recent conversation turns during pure retrieval queries.

---

## Architecture Flow

```
User Message / Tool Call
          │
          ▼
┌────────────────────────────────────────────────────────┐
│                   Session Pipeline                     │
│  1. Check Redis working memory (recent turns)          │
│  2. Hybrid search pgvector (episodic + long-term)      │
│  3. RRF score fusion + Cohere reranking                │
│  4. Assemble context window within token budget        │
└────────────────────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────────┐
│                  Ingestion Pipeline                    │
│  1. LLM extracts facts + structured entities           │
│  2. Classify memory tier (episodic / long_term)        │
│  3. Chunking (fixed, recursive, structural markdown)   │
│  4. Local / OpenAI embedding generation                │
│  5. Dedup, conflict resolution, store in Postgres      │
└────────────────────────────────────────────────────────┘
```

---

## Quickstart

### 1. Clone Repository

```bash
git clone https://github.com/kharshita590/Mnemosyne.git
cd Mnemosyne
```

### 2. Configure Environment

Create `.env` from template:

```bash
cp .env.example .env
```

Minimal `.env` for zero-cost local setup (using local embeddings + OpenAI or Claude for extraction):

```env
# Embedding provider (local requires no API key)
EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
LOCAL_EMBEDDING_DIMENSIONS=384

# LLM provider for fact extraction & chat (choose one)
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=your_openai_api_key_here

# MCP Auth (leave empty or changeme to disable in local dev)
MCP_API_KEY=
```

### 3. Run with Docker Compose (Recommended)

One command starts PostgreSQL (with pgvector), Redis, runs Alembic migrations, and boots the Mnemosyne MCP server on port `8080`:

```bash
docker compose up --build -d
```

Check health:

```bash
curl http://localhost:8080/health
```

Expected output:
```json
{"postgres": "ok", "redis": "ok", "status": "healthy"}
```

---

## Client Setup Guides

### 1. Claude Desktop

Edit your Claude Desktop configuration:
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

#### Option A: HTTP Remote / Docker Setup (Recommended)

When Mnemosyne is running via Docker on port 8080:

```json
{
  "mcpServers": {
    "mnemosyne": {
      "command": "npx",
      "args": ["-y", "mcp-proxy", "http://localhost:8080/mcp"]
    }
  }
}
```

#### Option B: Direct Python / Stdio Setup

```json
{
  "mcpServers": {
    "mnemosyne": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/absolute/path/to/Mnemosyne",
        "python",
        "-m",
        "mcp_local.server"
      ],
      "env": {
        "MCP_TRANSPORT": "stdio",
        "DATABASE_URL": "postgresql+asyncpg://mnemosyne:mnemosyne@localhost:5436/mnemosyne",
        "REDIS_URL": "redis://localhost:6379/0"
      }
    }
  }
}
```

---

### 2. Cursor

Add to your project's `.cursor/mcp.json` or global `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "mnemosyne": {
      "url": "http://localhost:8080/mcp"
    }
  }
}
```

---

### 3. OpenClaw

Add Mnemosyne to your OpenClaw configuration (`~/.openclaw/config.yaml` or workspace tools):

```yaml
tools:
  mcp:
    servers:
      mnemosyne:
        url: "http://localhost:8080/mcp"
        timeoutSeconds: 30
```

Or run via stdio execution profile:

```yaml
tools:
  mcp:
    servers:
      mnemosyne:
        command: "docker"
        args: ["compose", "exec", "-T", "mnemosyne", "python", "-m", "mcp_local.server"]
        env:
          MCP_TRANSPORT: "stdio"
```

---

### 4. Windsurf / Codeium

In `~/.codeium/windsurf/mcp_config.json`:

```json
{
  "mcpServers": {
    "mnemosyne": {
      "serverUrl": "http://localhost:8080/mcp"
    }
  }
}
```

---

### 5. Cline (VS Code Extension)

In Cline Settings → MCP Servers (`cline_mcp_settings.json`):

```json
{
  "mcpServers": {
    "mnemosyne": {
      "url": "http://localhost:8080/mcp",
      "disabled": false,
      "autoApprove": [
        "retrieve_memory_tool",
        "ingest_memory_tool",
        "remember_turn_tool"
      ]
    }
  }
}
```

---

## Available MCP Tools

| Tool | Parameters | Description |
|---|---|---|
| `ingest_memory_tool` | `user_id`, `content`, `conversation_id`, `sync` | Persist facts, decisions, rules, or preferences. Set `sync=True` to block until stored. |
| `retrieve_memory_tool` | `user_id`, `query`, `conversation_id` | Hybrid search (dense + BM25) across memories and active Redis dialogue turns. |
| `remember_turn_tool` | `user_id`, `conversation_id`, `role`, `content` | Store dialogue turn in working memory for session summaries. |
| `session_init_tool` | `user_id`, `conversation_id`, `session_context` | Generate pre-formatted system prompt memory block for new sessions. |
| `chat_tool` | `user_id`, `conversation_id`, `user_message` | Server-side memory-augmented chat generation. |
| `end_session_tool` | `user_id`, `conversation_id` | Summarize working memory, store as episodic memory, and flush Redis session. |
| `forget_memory_tool` | `user_id`, `memory_id` | Delete a single memory by ID. |
| `forget_memories_tool` | `user_id`, `topic`, `tier`, `before_date`, `confirm` | Preview or bulk-delete memories matching criteria. |

---

## Local Development (Without Docker)

If running outside Docker using `uv`:

```bash
# 1. Install dependencies
uv sync --frozen

# 2. Start backing services
docker run -d --name pg -p 5436:5432 -e POSTGRES_PASSWORD=mnemosyne -e POSTGRES_DB=mnemosyne pgvector/pgvector:pg16
docker run -d --name redis -p 6379:6379 redis:7-alpine

# 3. Run database migrations
uv run alembic upgrade head

# 4. Start HTTP MCP server
uv run python -m mcp_local.server
```

---

## Maintenance & Periodic Tasks

```bash
# Sweep decayed memories (run nightly)
python jobs/decay_sweep.py

# Retry stuck background ingests
python jobs/retry_stuck_ingests.py
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    database_url: str = "postgresql+asyncpg://mnemosyne:mnemosyne@localhost:5436/mnemosyne"
    redis_url: str = "redis://localhost:6379/0"

    openai_api_key: str = ""
    anthropic_api_key: str = ""
    gemini_api_key: str = ""

    # LLM provider: anthropic | openai | gemini | local
    llm_provider: str = "anthropic"
    # Override model for the selected provider; uses provider default if empty
    llm_model: str = ""

    # Local / Ollama settings (used when llm_provider = "local")
    local_llm_base_url: str = "http://localhost:11434"
    local_llm_model: str = "llama3"

    # Embedding provider: openai | local
    embedding_provider: str = "openai"

    # OpenAI embedding settings (used when embedding_provider = "openai")
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # Local / sentence-transformers embedding settings (used when embedding_provider = "local")
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"
    local_embedding_dimensions: int = 384

    cohere_api_key: str = ""

    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    s3_bucket: str = "mnemosyne-raw"

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    encrypt_key: str = ""
    mcp_api_key: str = "changeme"

    # Working memory TTL in seconds
    working_memory_ttl: int = 14400  # 4 hours

    # Retrieval
    retrieval_top_k: int = 20
    rerank_top_n: int = 5
    context_token_budget: int = 2000

    # Deduplication & conflict resolution
    dedup_similarity_threshold: float = 0.92  # cosine similarity above which we check for dup/conflict
    dedup_candidates_limit: int = 5           # how many similar memories to compare against


settings = Settings()

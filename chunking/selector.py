from __future__ import annotations

import re

from chunking.base import Chunk, ChunkStrategy
from chunking.fixed import FixedChunker
from chunking.recursive import RecursiveChunker
from chunking.semantic import SemanticChunker
from chunking.sentence import SentenceChunker
from chunking.structural import StructuralChunker
from storage.models import MemoryTier


def _has_markdown_headers(text: str) -> bool:
    return bool(re.search(r"^#{1,3} ", text, re.MULTILINE))


def _has_code_blocks(text: str) -> bool:
    return "```" in text or "    " * 2 in text


def _is_conversational(text: str) -> bool:
    """Short, dialogue-style text with no complex structure."""
    return len(text) < 800 and "\n\n" not in text


def _is_atomic_fact(text: str) -> bool:
    """Single short statement — no need to split further."""
    return len(text.split()) <= 40


def _make_semantic_chunker() -> SemanticChunker:
    """Build a SemanticChunker backed by the local sentence-transformers model."""
    from embedding.local import _load_model
    from config.settings import settings
    model = _load_model(settings.local_embedding_model)

    def embed_fn(text: str) -> list[float]:
        return model.encode(text, normalize_embeddings=True).tolist()

    return SemanticChunker(embed_fn=embed_fn, similarity_threshold=0.75)


def select_strategy(text: str, tier: MemoryTier | None = None) -> ChunkStrategy:
    """
    Inspect content and tier to return the best chunking strategy.

    Decision order:
    1. Atomic fact (<=40 words)          -> FixedChunker (no split needed)
    2. Markdown / structured doc         -> StructuralChunker
    3. Episodic / long_term prose        -> SemanticChunker (best boundary quality)
    4. Conversational short text         -> SentenceChunker
    5. Everything else                   -> RecursiveChunker
    """
    if _is_atomic_fact(text):
        return FixedChunker(chunk_size=512, overlap=0)
    if _has_markdown_headers(text) or _has_code_blocks(text):
        return StructuralChunker()
    # Use semantic chunking for the tiers where chunk quality matters most.
    # Working memories are ephemeral — fast fixed chunking is fine for them.
    if tier in (MemoryTier.EPISODIC, MemoryTier.LONG_TERM, MemoryTier.SEMANTIC):
        try:
            return _make_semantic_chunker()
        except Exception:
            # sentence-transformers not installed or model not loaded — fall through
            pass
    if _is_conversational(text):
        return SentenceChunker(sentences_per_chunk=3)
    return RecursiveChunker(chunk_size=800, overlap=100)


def chunk_text(text: str, strategy: ChunkStrategy | None = None, tier: MemoryTier | None = None) -> list[Chunk]:
    """Chunk text using the provided strategy or auto-select one based on content and tier."""
    s = strategy or select_strategy(text, tier=tier)
    return s.chunk(text)

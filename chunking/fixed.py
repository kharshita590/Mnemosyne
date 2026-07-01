from __future__ import annotations

import tiktoken

from chunking.base import Chunk, ChunkStrategy


class FixedChunker(ChunkStrategy):
    name = "fixed"

    def __init__(self, chunk_size: int = 512, overlap: int = 64, model: str = "cl100k_base"):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.enc = tiktoken.get_encoding(model)

    def chunk(self, text: str) -> list[Chunk]:
        tokens = self.enc.encode(text)
        chunks = []
        start = 0
        idx = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_text = self.enc.decode(tokens[start:end])
            chunks.append(Chunk(text=chunk_text, index=idx, strategy=self.name))
            idx += 1
            start += self.chunk_size - self.overlap
        return chunks

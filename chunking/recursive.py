from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from chunking.base import Chunk, ChunkStrategy


class RecursiveChunker(ChunkStrategy):
    name = "recursive"

    def __init__(self, chunk_size: int = 1000, overlap: int = 100):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk(self, text: str) -> list[Chunk]:
        parts = self._splitter.split_text(text)
        return [Chunk(text=p, index=i, strategy=self.name) for i, p in enumerate(parts)]

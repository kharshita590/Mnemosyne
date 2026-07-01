from __future__ import annotations

import re

from chunking.base import Chunk, ChunkStrategy


class SentenceChunker(ChunkStrategy):
    name = "sentence"

    def __init__(self, sentences_per_chunk: int = 4, overlap: int = 1):
        self.sentences_per_chunk = sentences_per_chunk
        self.overlap = overlap

    def _split_sentences(self, text: str) -> list[str]:
        raw = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in raw if s.strip()]

    def chunk(self, text: str) -> list[Chunk]:
        sentences = self._split_sentences(text)
        if not sentences:
            return [Chunk(text=text, index=0, strategy=self.name)]
        chunks = []
        idx = 0
        step = max(1, self.sentences_per_chunk - self.overlap)
        for start in range(0, len(sentences), step):
            group = sentences[start : start + self.sentences_per_chunk]
            chunks.append(Chunk(text=" ".join(group), index=idx, strategy=self.name))
            idx += 1
        return chunks

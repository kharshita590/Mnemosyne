from __future__ import annotations

import re

from chunking.base import Chunk, ChunkStrategy


class StructuralChunker(ChunkStrategy):
    """Splits on Markdown headers, code fences, and blank lines."""
    name = "structural"

    def chunk(self, text: str) -> list[Chunk]:
        # Split on h1/h2 markdown headers or double newlines before a header
        sections = re.split(r"(?=\n#{1,2} )", text)
        if len(sections) <= 1:
            # Fallback: split on triple newline (section break in plain text)
            sections = re.split(r"\n{3,}", text)
        sections = [s.strip() for s in sections if s.strip()]
        return [Chunk(text=s, index=i, strategy=self.name) for i, s in enumerate(sections)]

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    index: int
    strategy: str
    metadata: dict | None = None


class ChunkStrategy(ABC):
    name: str = "base"

    @abstractmethod
    def chunk(self, text: str) -> list[Chunk]:
        """Split text into chunks. Must return at least one chunk."""
        ...

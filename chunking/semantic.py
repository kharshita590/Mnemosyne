from __future__ import annotations

import numpy as np

from chunking.base import Chunk, ChunkStrategy
from chunking.sentence import SentenceChunker


class SemanticChunker(ChunkStrategy):
    """
    Groups sentences by semantic similarity. Starts a new chunk when cosine
    similarity to the previous sentence drops below a threshold.

    Requires a synchronous embedding callable (sentence -> list[float]).
    For production, use a locally-loaded model to avoid latency per sentence.
    """
    name = "semantic"

    def __init__(self, embed_fn, similarity_threshold: float = 0.75):
        self._embed = embed_fn
        self.threshold = similarity_threshold
        self._sentence_chunker = SentenceChunker(sentences_per_chunk=1)

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        na, nb = np.array(a), np.array(b)
        denom = (np.linalg.norm(na) * np.linalg.norm(nb))
        if denom == 0:
            return 0.0
        return float(np.dot(na, nb) / denom)

    def chunk(self, text: str) -> list[Chunk]:
        sentences = [c.text for c in self._sentence_chunker.chunk(text)]
        if not sentences:
            return [Chunk(text=text, index=0, strategy=self.name)]

        groups: list[list[str]] = [[sentences[0]]]
        prev_emb = self._embed(sentences[0])

        for sentence in sentences[1:]:
            emb = self._embed(sentence)
            sim = self._cosine(prev_emb, emb)
            if sim >= self.threshold:
                groups[-1].append(sentence)
            else:
                groups.append([sentence])
            prev_emb = emb

        return [
            Chunk(text=" ".join(g), index=i, strategy=self.name)
            for i, g in enumerate(groups)
        ]

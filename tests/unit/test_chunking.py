from __future__ import annotations

import pytest

from chunking.fixed import FixedChunker
from chunking.recursive import RecursiveChunker
from chunking.selector import select_strategy
from chunking.sentence import SentenceChunker
from chunking.structural import StructuralChunker


class TestFixedChunker:
    def test_single_chunk_for_short_text(self):
        chunker = FixedChunker(chunk_size=512, overlap=0)
        chunks = chunker.chunk("Hello world.")
        assert len(chunks) >= 1
        assert chunks[0].strategy == "fixed"

    def test_multiple_chunks_for_long_text(self):
        chunker = FixedChunker(chunk_size=10, overlap=2)
        text = " ".join(["word"] * 100)
        chunks = chunker.chunk(text)
        assert len(chunks) > 1

    def test_overlap_causes_more_chunks(self):
        text = " ".join(["word"] * 200)
        no_overlap = FixedChunker(chunk_size=50, overlap=0).chunk(text)
        with_overlap = FixedChunker(chunk_size=50, overlap=20).chunk(text)
        assert len(with_overlap) >= len(no_overlap)

    def test_chunk_indices_sequential(self):
        chunker = FixedChunker(chunk_size=10, overlap=0)
        chunks = chunker.chunk(" ".join(["x"] * 100))
        for i, chunk in enumerate(chunks):
            assert chunk.index == i


class TestRecursiveChunker:
    def test_splits_on_double_newline(self):
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunker = RecursiveChunker(chunk_size=50, overlap=0)
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2

    def test_returns_at_least_one_chunk(self):
        chunker = RecursiveChunker()
        chunks = chunker.chunk("Short text.")
        assert len(chunks) >= 1
        assert chunks[0].strategy == "recursive"


class TestSentenceChunker:
    def test_splits_on_sentence_boundaries(self):
        text = "This is sentence one. This is sentence two. This is sentence three. And four."
        chunker = SentenceChunker(sentences_per_chunk=2, overlap=0)
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2

    def test_single_sentence_returns_one_chunk(self):
        chunker = SentenceChunker(sentences_per_chunk=4)
        chunks = chunker.chunk("Just one sentence here.")
        assert len(chunks) == 1
        assert chunks[0].strategy == "sentence"

    def test_overlap_includes_shared_sentences(self):
        text = "A. B. C. D. E. F. G. H."
        chunker = SentenceChunker(sentences_per_chunk=3, overlap=1)
        chunks = chunker.chunk(text)
        assert len(chunks) > 1


class TestStructuralChunker:
    def test_splits_on_markdown_headers(self):
        text = "# Header One\n\nContent one.\n\n## Header Two\n\nContent two."
        chunker = StructuralChunker()
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2
        assert chunks[0].strategy == "structural"

    def test_fallback_on_plain_text(self):
        text = "No headers here.\n\n\n\nJust paragraphs separated by blank lines."
        chunker = StructuralChunker()
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1


class TestSelectStrategy:
    def test_atomic_fact_uses_fixed(self):
        text = "Hari uses pgvector for vector similarity search."
        strategy = select_strategy(text)
        assert strategy.name == "fixed"

    def test_markdown_uses_structural(self):
        text = "# Title\n\nSome content here.\n\n## Section\n\nMore content."
        strategy = select_strategy(text)
        assert strategy.name == "structural"

    def test_conversational_uses_sentence(self):
        text = "I was thinking about the bug. It was the fan-out issue causing duplicates."
        strategy = select_strategy(text)
        assert strategy.name == "sentence"

    def test_long_prose_uses_recursive(self):
        text = " ".join(["This is a long unstructured paragraph about the system."] * 30)
        strategy = select_strategy(text)
        assert strategy.name == "recursive"

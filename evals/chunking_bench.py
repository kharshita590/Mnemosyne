from __future__ import annotations

from chunking.selector import select_strategy

SAMPLES = {
    "atomic_fact": "Hari uses pgvector for vector similarity search.",
    "conversational": "I was thinking about the bug we fixed yesterday in the join query. It was the fan-out issue with the budget table causing duplicate rows.",
    "structured_doc": "# Architecture\n\n## Storage\nWe use PostgreSQL with pgvector.\n\n## Agents\nLangGraph handles orchestration.",
    "long_prose": " ".join(["This is a long unstructured paragraph about the system."] * 30),
}


def run_benchmark() -> None:
    for name, text in SAMPLES.items():
        strategy = select_strategy(text)
        chunks = strategy.chunk(text)
        print(f"{name}: strategy={strategy.name}, chunks={len(chunks)}, avg_len={sum(len(c.text) for c in chunks) // len(chunks)}")


if __name__ == "__main__":
    run_benchmark()

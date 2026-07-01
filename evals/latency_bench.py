"""
Latency Benchmark — P50 / P90 / P99 under concurrent load.

Fires N concurrent retrieval queries against pre-seeded memories and
measures end-to-end latency distribution.

Metrics (all in milliseconds):
  p50   median latency
  p90   90th percentile
  p99   99th percentile
  min / max / avg
  throughput  queries per second

Usage:
    python -m evals.latency_bench
    python -m evals.latency_bench --concurrency 50 --queries 200
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from agents.router import Intent, route
from config.logging import logger
from storage.pg import upsert_user


# ---------------------------------------------------------------------------
# Seed data — realistic memory content for retrieval
# ---------------------------------------------------------------------------

_SEED_MEMORIES = [
    "The user prefers Python for backend services and Go for CLI tooling.",
    "PostgreSQL with pgvector is the primary database for vector similarity search.",
    "The team uses LangGraph for multi-step agent orchestration.",
    "Redis is used for working memory and embedding caching with a 4-hour TTL.",
    "The user lives in Bangalore and works at a fintech startup in Koramangala.",
    "VS Code with Monokai Pro theme is the preferred editor, dark mode always on.",
    "The project Mnemosyne is a persistent memory layer for LLM applications.",
    "Cohere rerank-v3.5 is used for cross-encoder reranking of retrieval results.",
    "Daily standup happens at 10am IST. Sprint reviews are on Friday.",
    "The user is preparing for the AWS Solutions Architect Professional exam.",
    "FastMCP is the MCP framework used to expose Mnemosyne's 6 tools to Claude Desktop.",
    "sentence-transformers BAAI/bge-small-en-v1.5 is the local embedding model (384 dims).",
    "The payment module was shipped last week. Now working on the notification system.",
    "Max is a 3-year-old golden retriever. He had food poisoning last month but recovered.",
    "The team grew to 12 engineers after the Q4 hiring round.",
    "Atomic Habits by James Clear inspired a 6am exercise routine.",
    "Alembic is used for database migrations. Two migrations exist: initial schema and vector resize.",
    "Docker Compose runs Postgres 16 on port 5436 and Redis 7 on port 6379.",
    "The ingestion pipeline is fire-and-forget — returns job_id immediately.",
    "Decay weight uses exponential half-life of 30 days plus a frequency access boost.",
]

_QUERIES = [
    "What database does the user use?",
    "Where does the user live?",
    "What editor does the user prefer?",
    "What is Mnemosyne?",
    "How does the team do standups?",
    "What is the embedding model used?",
    "What happened to Max?",
    "What exam is the user preparing for?",
    "What is the team size?",
    "What was shipped last week?",
    "What is the decay formula?",
    "How does ingestion work?",
    "What ports does Docker Compose expose?",
    "What is the primary programming language?",
    "What is the MCP framework used?",
]


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

@dataclass
class LatencyRun:
    query: str
    latency_ms: float
    success: bool
    error: str = ""


@dataclass
class LatencyReport:
    runs: list[LatencyRun] = field(default_factory=list)

    def percentile(self, p: float) -> float:
        latencies = sorted(r.latency_ms for r in self.runs if r.success)
        if not latencies:
            return 0.0
        idx = max(0, int(len(latencies) * p / 100) - 1)
        return latencies[idx]

    def summarize(self) -> dict[str, Any]:
        successful = [r for r in self.runs if r.success]
        failed = [r for r in self.runs if not r.success]
        latencies = [r.latency_ms for r in successful]

        if not latencies:
            return {"error": "all queries failed", "failures": [r.error for r in failed]}

        return {
            "total_queries": len(self.runs),
            "successful": len(successful),
            "failed": len(failed),
            "latency_ms": {
                "p50": round(self.percentile(50), 1),
                "p90": round(self.percentile(90), 1),
                "p99": round(self.percentile(99), 1),
                "min": round(min(latencies), 1),
                "max": round(max(latencies), 1),
                "avg": round(statistics.mean(latencies), 1),
                "stdev": round(statistics.stdev(latencies) if len(latencies) > 1 else 0, 1),
            },
            "throughput_qps": round(
                len(successful) / (sum(latencies) / 1000 / len(successful))
                if latencies else 0,
                2,
            ),
            "failures": [{"query": r.query, "error": r.error} for r in failed],
        }


async def _seed_memories(user_id: str) -> None:
    """Ingest all seed memories for the benchmark user."""
    tasks = [
        route(Intent.INGEST, user_id=user_id, content=mem)
        for mem in _SEED_MEMORIES
    ]
    await asyncio.gather(*tasks, return_exceptions=True)
    # Brief wait for async ingestion flush
    await asyncio.sleep(3)


async def _single_query(user_id: str, query: str) -> LatencyRun:
    t0 = time.perf_counter()
    try:
        await route(Intent.RETRIEVE, user_id=user_id, query=query)
        latency_ms = (time.perf_counter() - t0) * 1000
        return LatencyRun(query=query, latency_ms=latency_ms, success=True)
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        return LatencyRun(query=query, latency_ms=latency_ms, success=False, error=str(exc))


async def run_latency_bench(
    concurrency: int = 20,
    total_queries: int = 100,
) -> dict[str, Any]:
    """
    Seed memories once, then fire total_queries retrievals with concurrency cap.

    Args:
        concurrency: max concurrent retrieval coroutines
        total_queries: total queries to fire across the run
    """
    user_id = f"latency-bench-{uuid.uuid4().hex[:8]}"
    await upsert_user(user_id)

    logger.info("latency_bench_seeding", user_id=user_id, memories=len(_SEED_MEMORIES))
    await _seed_memories(user_id)
    logger.info("latency_bench_seed_done", user_id=user_id)

    # Build query list (cycle through _QUERIES to reach total_queries)
    queries = [_QUERIES[i % len(_QUERIES)] for i in range(total_queries)]

    # Fire with semaphore-limited concurrency
    semaphore = asyncio.Semaphore(concurrency)

    async def _bounded(q: str) -> LatencyRun:
        async with semaphore:
            return await _single_query(user_id, q)

    report = LatencyReport()
    wall_start = time.perf_counter()

    runs = await asyncio.gather(*[_bounded(q) for q in queries])
    wall_elapsed = time.perf_counter() - wall_start

    report.runs = list(runs)
    summary = report.summarize()
    summary["wall_clock_seconds"] = round(wall_elapsed, 2)
    summary["effective_throughput_qps"] = round(total_queries / wall_elapsed, 2)
    summary["concurrency"] = concurrency

    logger.info(
        "latency_bench_complete",
        p50=summary["latency_ms"]["p50"],
        p90=summary["latency_ms"]["p90"],
        p99=summary["latency_ms"]["p99"],
        qps=summary["effective_throughput_qps"],
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latency benchmark for Mnemosyne retrieval")
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--queries", type=int, default=100)
    args = parser.parse_args()

    result = asyncio.run(run_latency_bench(concurrency=args.concurrency, total_queries=args.queries))
    print(json.dumps(result, indent=2))

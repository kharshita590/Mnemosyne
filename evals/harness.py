from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from agents.router import Intent, route
from config.logging import logger


async def run_eval(golden_set_path: str = "evals/golden_set.json") -> dict:
    """
    Eval harness. Loads golden set, runs retrieval for each case, and
    computes recall@5 and MRR across the full set.

    Golden set format:
    [
      {
        "user_id": "eval-user-001",
        "query": "what database does the user use for vector search?",
        "expected_content_keywords": ["pgvector", "postgresql"],
        "description": "optional human-readable label for this case"
      }
    ]

    keyword-based matching: a retrieved memory counts as relevant if it
    contains at least one expected keyword (case-insensitive).
    """
    path = Path(golden_set_path)
    if not path.exists():
        return {"error": f"golden set not found at {golden_set_path}"}

    cases = json.loads(path.read_text())
    recall_hits = 0
    mrr_sum = 0.0
    total_latency_ms = 0.0
    failures: list[dict] = []

    for case in cases:
        t0 = time.perf_counter()
        try:
            state = await route(
                Intent.RETRIEVE,
                user_id=case["user_id"],
                query=case["query"],
            )
        except Exception as e:
            failures.append({"query": case["query"], "error": str(e)})
            continue

        latency_ms = (time.perf_counter() - t0) * 1000
        total_latency_ms += latency_ms

        retrieved = [m.content.lower() for m in (state.get("final_memories") or [])]
        keywords = [k.lower() for k in case.get("expected_content_keywords", [])]

        # A result is relevant if it contains any expected keyword
        def is_relevant(content: str) -> bool:
            return any(kw in content for kw in keywords)

        # Recall@5
        top5 = retrieved[:5]
        hit = any(is_relevant(c) for c in top5)
        if hit:
            recall_hits += 1

        # MRR — reciprocal rank of first hit in top-5
        for rank, content in enumerate(top5, start=1):
            if is_relevant(content):
                mrr_sum += 1.0 / rank
                break

        if not hit:
            failures.append({
                "query": case["query"],
                "expected_keywords": keywords,
                "retrieved_top3": retrieved[:3],
                "latency_ms": round(latency_ms, 1),
            })

    n = len(cases)
    result = {
        "n": n,
        "recall_at_5": round(recall_hits / n, 4) if n else 0,
        "mrr": round(mrr_sum / n, 4) if n else 0,
        "avg_latency_ms": round(total_latency_ms / max(n, 1), 1),
        "failures": failures,
    }
    logger.info("eval_complete", **{k: v for k, v in result.items() if k != "failures"})
    return result


if __name__ == "__main__":
    results = asyncio.run(run_eval())
    print(json.dumps(results, indent=2))

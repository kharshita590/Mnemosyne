"""
Conflict Resolution Accuracy Benchmark.

Tests that when contradicting facts are ingested (old then new),
Mnemosyne returns the latest/correct value on retrieval — not the stale one.

Each case:
  1. Ingest an old fact (now outdated)
  2. Ingest a new contradicting fact
  3. Retrieve using a natural query
  4. Assert retrieved answer contains the NEW value, not the old one

Metrics:
  resolution_accuracy   — % of cases where new value is returned
  old_value_bleed_rate  — % of cases where old (stale) value appears first

Usage:
    python -m evals.conflict_bench
"""
from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from typing import Any

from agents.router import Intent, route
from config.logging import logger
from evals.llm_judge import keyword_score
from storage.pg import upsert_user


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

# Each case: (old_fact, new_fact, query, correct_value, stale_value)
CONFLICT_CASES = [
    {
        "id": "city_move",
        "description": "User moved cities — should return new city",
        "old_fact": "I live in Mumbai.",
        "new_fact": "I moved to Bangalore last month for a new job.",
        "query": "Where does the user live?",
        "correct_keywords": ["bangalore"],
        "stale_keywords": ["mumbai"],
    },
    {
        "id": "job_change",
        "description": "Job change — should return current employer",
        "old_fact": "I work at Flipkart as a senior engineer.",
        "new_fact": "I recently joined Anthropic as a research engineer.",
        "query": "Where does the user work?",
        "correct_keywords": ["anthropic"],
        "stale_keywords": ["flipkart"],
    },
    {
        "id": "language_switch",
        "description": "Primary language changed — should return new language",
        "old_fact": "My primary programming language is Java.",
        "new_fact": "I switched to Python as my main language six months ago.",
        "query": "What is the user's primary programming language?",
        "correct_keywords": ["python"],
        "stale_keywords": ["java"],
    },
    {
        "id": "diet_change",
        "description": "Dietary preference changed",
        "old_fact": "I eat meat regularly and love burgers.",
        "new_fact": "I went vegetarian three months ago.",
        "query": "What is the user's diet?",
        "correct_keywords": ["vegetarian"],
        "stale_keywords": ["meat", "burger"],
    },
    {
        "id": "editor_switch",
        "description": "Editor preference updated",
        "old_fact": "I use Vim as my primary code editor.",
        "new_fact": "I switched from Vim to VS Code after joining my new team.",
        "query": "Which code editor does the user prefer?",
        "correct_keywords": ["vs code", "vscode", "visual studio code"],
        "stale_keywords": ["vim"],
    },
    {
        "id": "db_migration",
        "description": "Database technology changed",
        "old_fact": "We use MySQL as our primary database.",
        "new_fact": "We migrated from MySQL to PostgreSQL last quarter.",
        "query": "What database does the team use?",
        "correct_keywords": ["postgresql", "postgres"],
        "stale_keywords": ["mysql"],
    },
    {
        "id": "team_size",
        "description": "Team size update",
        "old_fact": "My team has 5 engineers.",
        "new_fact": "Our team grew to 12 engineers after the last hiring round.",
        "query": "How many engineers are on the user's team?",
        "correct_keywords": ["12", "twelve"],
        "stale_keywords": ["5", "five"],
    },
    {
        "id": "project_status",
        "description": "Project completion status",
        "old_fact": "We are still building the payment module.",
        "new_fact": "We shipped the payment module last week and moved to the notification system.",
        "query": "What is the status of the payment module?",
        "correct_keywords": ["shipped", "complete", "notification"],
        "stale_keywords": ["still building", "building"],
    },
]


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

@dataclass
class ConflictCaseResult:
    case_id: str
    description: str
    query: str
    correct_keywords: list[str]
    stale_keywords: list[str]
    retrieved_context: str
    correct_found: bool
    stale_found: bool
    resolution: str     # "correct" | "stale" | "empty" | "both"


async def _run_conflict_case(case: dict) -> ConflictCaseResult:
    user_id = f"conflict-{case['id']}-{uuid.uuid4().hex[:6]}"
    await upsert_user(user_id)

    # Ingest old fact first, then new contradicting fact
    await route(Intent.INGEST, user_id=user_id, content=case["old_fact"])
    # Small delay so old fact is indexed before new arrives
    await asyncio.sleep(1)
    await route(Intent.INGEST, user_id=user_id, content=case["new_fact"])

    # Wait for async ingestion to complete
    await asyncio.sleep(2)

    state = await route(Intent.RETRIEVE, user_id=user_id, query=case["query"])
    memories = state.get("final_memories") or []
    retrieved = " ".join(m.content.lower() for m in memories[:5])

    correct_found = any(kw in retrieved for kw in case["correct_keywords"])
    stale_found = any(kw in retrieved for kw in case["stale_keywords"])

    if correct_found and not stale_found:
        resolution = "correct"
    elif stale_found and not correct_found:
        resolution = "stale"
    elif not retrieved.strip():
        resolution = "empty"
    else:
        resolution = "both"   # both found — ambiguous

    return ConflictCaseResult(
        case_id=case["id"],
        description=case["description"],
        query=case["query"],
        correct_keywords=case["correct_keywords"],
        stale_keywords=case["stale_keywords"],
        retrieved_context=retrieved[:300],
        correct_found=correct_found,
        stale_found=stale_found,
        resolution=resolution,
    )


async def run_conflict_bench() -> dict[str, Any]:
    """
    Run all conflict resolution cases. Returns accuracy metrics.
    """
    results: list[ConflictCaseResult] = []

    for case in CONFLICT_CASES:
        result = await _run_conflict_case(case)
        results.append(result)
        logger.info(
            "conflict_case",
            case_id=result.case_id,
            resolution=result.resolution,
            correct_found=result.correct_found,
            stale_found=result.stale_found,
        )

    n = len(results)
    correct_count = sum(1 for r in results if r.resolution == "correct")
    stale_count = sum(1 for r in results if r.resolution == "stale")
    both_count = sum(1 for r in results if r.resolution == "both")
    empty_count = sum(1 for r in results if r.resolution == "empty")

    summary = {
        "total_cases": n,
        "resolution_accuracy_pct": round(correct_count / max(n, 1) * 100, 1),
        "stale_bleed_rate_pct": round(stale_count / max(n, 1) * 100, 1),
        "ambiguous_pct": round(both_count / max(n, 1) * 100, 1),
        "empty_pct": round(empty_count / max(n, 1) * 100, 1),
        "by_resolution": {
            "correct": correct_count,
            "stale": stale_count,
            "both": both_count,
            "empty": empty_count,
        },
        "details": [
            {
                "case_id": r.case_id,
                "description": r.description,
                "resolution": r.resolution,
                "correct_found": r.correct_found,
                "stale_found": r.stale_found,
                "retrieved_snippet": r.retrieved_context[:150],
            }
            for r in results
        ],
        "failures": [
            {
                "case_id": r.case_id,
                "description": r.description,
                "resolution": r.resolution,
                "expected": "correct",
            }
            for r in results
            if r.resolution != "correct"
        ],
    }
    return summary


if __name__ == "__main__":
    result = asyncio.run(run_conflict_bench())
    print(json.dumps(result, indent=2))

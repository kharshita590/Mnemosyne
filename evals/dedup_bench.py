"""
Deduplication Accuracy Benchmark.

Tests that when the same fact is ingested multiple times with paraphrasing,
Mnemosyne stores it only once (or very few times) rather than N copies.

Metrics:
  compression_ratio   = unique stored / total ingested  (lower is better)
  dedup_accuracy      = cases where stored_count == 1  (higher is better)
  false_dedup_rate    = cases where distinct facts were merged  (lower is better)

Usage:
    python -m evals.dedup_bench
"""
from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from typing import Any

from agents.router import Intent, route
from config.logging import logger
from sqlalchemy import func, select
from storage.models import Memory
from storage.pg import AsyncSessionLocal, upsert_user


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

# Each case: one canonical fact expressed in N different phrasings.
# After ingesting all phrasings, stored_count should be 1 (or very low).
DUPLICATE_CASES = [
    {
        "id": "dark_mode",
        "description": "UI theme preference — identical semantic meaning",
        "phrasings": [
            "I prefer dark mode for all my apps.",
            "The user uses dark mode as their preferred theme.",
            "Dark mode is my go-to UI setting.",
            "I always switch apps to dark mode.",
            "My UI preference is dark mode, not light mode.",
        ],
        "expected_stored": 1,
    },
    {
        "id": "python_primary",
        "description": "Primary programming language — minor paraphrase",
        "phrasings": [
            "Python is my primary programming language.",
            "I mostly code in Python.",
            "My main language is Python.",
            "I use Python more than any other language.",
        ],
        "expected_stored": 1,
    },
    {
        "id": "vscode_editor",
        "description": "Editor preference — different wording",
        "phrasings": [
            "I use VS Code as my code editor.",
            "My preferred editor is Visual Studio Code.",
            "I write all my code in VS Code.",
            "VS Code is the editor I use daily.",
        ],
        "expected_stored": 1,
    },
    {
        "id": "bangalore_location",
        "description": "Location fact — near-exact repetition",
        "phrasings": [
            "I live in Bangalore.",
            "My city is Bangalore.",
            "I'm based in Bangalore, India.",
        ],
        "expected_stored": 1,
    },
    {
        "id": "morning_routine",
        "description": "Habit — moderate paraphrase",
        "phrasings": [
            "I wake up at 6am every day to exercise.",
            "My morning routine starts at 6am with a workout.",
            "I exercise every morning at 6 o'clock.",
        ],
        "expected_stored": 1,
    },
]

# Each case: two genuinely different facts that should NOT be merged.
DISTINCT_CASES = [
    {
        "id": "two_languages",
        "description": "Python vs Go — different languages, should stay separate",
        "facts": [
            "I use Python for backend development.",
            "I use Go for writing CLI tools and infrastructure code.",
        ],
        "expected_stored": 2,
    },
    {
        "id": "two_cities",
        "description": "Home city vs travel destination — distinct facts",
        "facts": [
            "I live in Bangalore.",
            "I'm visiting Mumbai for a conference next week.",
        ],
        "expected_stored": 2,
    },
    {
        "id": "two_tools",
        "description": "Different tools for different purposes",
        "facts": [
            "I use PostgreSQL as my primary database.",
            "I use Redis for caching and session storage.",
        ],
        "expected_stored": 2,
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _count_memories_for_user(user_id: str, keyword: str) -> int:
    """Count memories stored for a user whose content contains a keyword."""
    from storage.pg import upsert_user as _get_user
    user = await _get_user(user_id)
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(func.count(Memory.id)).where(
                Memory.user_id == user.id,
                Memory.content.ilike(f"%{keyword}%"),
            )
        )
        return result.scalar_one()


async def _count_all_memories_for_user(user_id: str) -> int:
    user = await upsert_user(user_id)
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(func.count(Memory.id)).where(Memory.user_id == user.id)
        )
        return result.scalar_one()


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

@dataclass
class DedupCaseResult:
    case_id: str
    description: str
    ingested: int
    stored: int
    expected: int
    passed: bool
    compression_ratio: float


async def _run_duplicate_case(case: dict) -> DedupCaseResult:
    user_id = f"dedup-dup-{case['id']}-{uuid.uuid4().hex[:6]}"
    await upsert_user(user_id)

    phrasings = case["phrasings"]
    for text in phrasings:
        await route(Intent.INGEST, user_id=user_id, content=text)

    # Wait briefly for async ingestion tasks to flush
    await asyncio.sleep(2)

    stored = await _count_all_memories_for_user(user_id)
    expected = case["expected_stored"]

    return DedupCaseResult(
        case_id=case["id"],
        description=case["description"],
        ingested=len(phrasings),
        stored=stored,
        expected=expected,
        passed=stored <= expected + 1,    # allow 1 extra (extracted facts may add a row)
        compression_ratio=round(stored / len(phrasings), 3),
    )


async def _run_distinct_case(case: dict) -> DedupCaseResult:
    user_id = f"dedup-dist-{case['id']}-{uuid.uuid4().hex[:6]}"
    await upsert_user(user_id)

    facts = case["facts"]
    for text in facts:
        await route(Intent.INGEST, user_id=user_id, content=text)

    await asyncio.sleep(2)

    stored = await _count_all_memories_for_user(user_id)
    expected = case["expected_stored"]

    return DedupCaseResult(
        case_id=case["id"],
        description=case["description"],
        ingested=len(facts),
        stored=stored,
        expected=expected,
        passed=stored >= expected,         # must NOT merge distinct facts
        compression_ratio=round(stored / len(facts), 3),
    )


async def run_dedup_bench() -> dict[str, Any]:
    """
    Run all duplicate + distinct cases. Returns accuracy metrics.
    """
    dup_results: list[DedupCaseResult] = []
    dist_results: list[DedupCaseResult] = []

    # Run duplicate cases (should compress to ~1 memory each)
    for case in DUPLICATE_CASES:
        result = await _run_duplicate_case(case)
        dup_results.append(result)
        logger.info(
            "dedup_case",
            case_id=result.case_id,
            ingested=result.ingested,
            stored=result.stored,
            passed=result.passed,
        )

    # Run distinct cases (should keep all N memories)
    for case in DISTINCT_CASES:
        result = await _run_distinct_case(case)
        dist_results.append(result)
        logger.info(
            "distinct_case",
            case_id=result.case_id,
            ingested=result.ingested,
            stored=result.stored,
            passed=result.passed,
        )

    dup_pass = sum(1 for r in dup_results if r.passed)
    dist_pass = sum(1 for r in dist_results if r.passed)
    avg_compression = sum(r.compression_ratio for r in dup_results) / max(len(dup_results), 1)

    failures = [
        {"case_id": r.case_id, "ingested": r.ingested, "stored": r.stored, "expected": r.expected}
        for r in (dup_results + dist_results)
        if not r.passed
    ]

    summary = {
        "dedup_accuracy": {
            "duplicate_cases": len(dup_results),
            "duplicate_passed": dup_pass,
            "duplicate_accuracy_pct": round(dup_pass / max(len(dup_results), 1) * 100, 1),
            "avg_compression_ratio": round(avg_compression, 3),
            "note": "compression_ratio: stored/ingested — lower means better dedup",
        },
        "false_dedup_prevention": {
            "distinct_cases": len(dist_results),
            "distinct_passed": dist_pass,
            "false_dedup_rate_pct": round((len(dist_results) - dist_pass) / max(len(dist_results), 1) * 100, 1),
            "note": "false_dedup_rate: % of distinct facts incorrectly merged",
        },
        "failures": failures,
        "details": [
            {
                "case_id": r.case_id,
                "type": "duplicate",
                "ingested": r.ingested,
                "stored": r.stored,
                "passed": r.passed,
                "compression_ratio": r.compression_ratio,
            }
            for r in dup_results
        ] + [
            {
                "case_id": r.case_id,
                "type": "distinct",
                "ingested": r.ingested,
                "stored": r.stored,
                "passed": r.passed,
            }
            for r in dist_results
        ],
    }
    return summary


if __name__ == "__main__":
    result = asyncio.run(run_dedup_bench())
    print(json.dumps(result, indent=2))

"""
LOCOMO Benchmark — apples-to-apples comparison with mem0's published scores.

Dataset: episodic-memory/locomo on HuggingFace
  - Each row has a multi-turn conversation + QA pairs
  - QA types: single_hop, multi_hop, open_domain, summarization

Mem0's published baseline (LOCOMO):
  Recall    ~26.9%
  Precision ~30.5%
  LLM-judge avg score ~1.8 / 3.0

Usage:
    python -m evals.locomo
    python -m evals.locomo --split test --max-dialogs 50 --use-llm-judge
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

from agents.router import Intent, route
from config.logging import logger
from evals.llm_judge import keyword_score, llm_judge_score
from storage.pg import upsert_user


# ---------------------------------------------------------------------------
# LOCOMO dataset loader
# ---------------------------------------------------------------------------

def _load_locomo_hf(split: str = "test", max_dialogs: int | None = None) -> list[dict]:
    """
    Load LOCOMO from HuggingFace datasets library.
    Falls back to a bundled mini-sample if the library / network is unavailable.
    """
    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset("episodic-memory/locomo", split=split, trust_remote_code=True)
        rows = list(ds)
        if max_dialogs:
            rows = rows[:max_dialogs]
        logger.info("locomo_loaded", source="huggingface", split=split, dialogs=len(rows))
        return rows
    except Exception as exc:
        logger.warning("locomo_hf_unavailable", error=str(exc), fallback="mini_sample")
        return _mini_sample()


def _mini_sample() -> list[dict]:
    """
    Bundled 5-dialog sample so the benchmark runs offline.
    Matches LOCOMO schema: dialog (list of speaker/text turns) + qa (list of question/answer/type).
    """
    return [
        {
            "dialog_id": "sample_001",
            "dialog": [
                {"speaker": "A", "text": "I just moved to Bangalore last month for my new job at a fintech startup."},
                {"speaker": "B", "text": "That's exciting! What kind of work do you do there?"},
                {"speaker": "A", "text": "I'm a backend engineer. We build payment infrastructure using Python and PostgreSQL."},
                {"speaker": "B", "text": "Do you enjoy the city?"},
                {"speaker": "A", "text": "Yes, though the traffic is intense. I live in Koramangala."},
            ],
            "qa": [
                {"question": "Where does the person live?", "answer": "Bangalore, specifically Koramangala", "type": "single_hop"},
                {"question": "What technology does the person use for work?", "answer": "Python and PostgreSQL", "type": "single_hop"},
                {"question": "What kind of company does the person work at and in which city?", "answer": "A fintech startup in Bangalore", "type": "multi_hop"},
            ],
        },
        {
            "dialog_id": "sample_002",
            "dialog": [
                {"speaker": "A", "text": "My dog Max got sick last Tuesday. We had to rush him to the vet."},
                {"speaker": "B", "text": "Oh no, is he okay?"},
                {"speaker": "A", "text": "He's recovering. The vet said it was food poisoning. He ate something from the garden."},
                {"speaker": "B", "text": "What kind of dog is Max?"},
                {"speaker": "A", "text": "He's a 3-year-old golden retriever."},
            ],
            "qa": [
                {"question": "What happened to Max?", "answer": "Max got food poisoning", "type": "single_hop"},
                {"question": "What breed is Max?", "answer": "Golden retriever", "type": "single_hop"},
                {"question": "How old is Max and what happened to him?", "answer": "3-year-old golden retriever who had food poisoning", "type": "multi_hop"},
            ],
        },
        {
            "dialog_id": "sample_003",
            "dialog": [
                {"speaker": "A", "text": "I finished reading Atomic Habits last week. Really changed how I think about routines."},
                {"speaker": "B", "text": "What was your biggest takeaway?"},
                {"speaker": "A", "text": "The idea that 1% improvements compound over time. I've started waking up at 6am to exercise."},
                {"speaker": "B", "text": "Have you read anything else by James Clear?"},
                {"speaker": "A", "text": "Not yet, but I want to read his newsletter next."},
            ],
            "qa": [
                {"question": "What book did the person recently finish?", "answer": "Atomic Habits by James Clear", "type": "single_hop"},
                {"question": "What new habit did they start after reading the book?", "answer": "Waking up at 6am to exercise", "type": "single_hop"},
                {"question": "Who wrote the book the person read and what habit did it inspire?", "answer": "James Clear wrote Atomic Habits; inspired 6am exercise routine", "type": "multi_hop"},
            ],
        },
        {
            "dialog_id": "sample_004",
            "dialog": [
                {"speaker": "A", "text": "I've been learning to cook Italian food. Made my first carbonara yesterday."},
                {"speaker": "B", "text": "How did it turn out?"},
                {"speaker": "A", "text": "Pretty good actually! My partner loved it. I used guanciale instead of bacon."},
                {"speaker": "B", "text": "Are you taking classes?"},
                {"speaker": "A", "text": "No, just watching Babish on YouTube and reading Marcella Hazan's cookbook."},
            ],
            "qa": [
                {"question": "What cuisine is the person learning to cook?", "answer": "Italian", "type": "single_hop"},
                {"question": "What resources are they using to learn cooking?", "answer": "Babish on YouTube and Marcella Hazan's cookbook", "type": "single_hop"},
                {"question": "What dish did they make and how did they learn?", "answer": "Carbonara using Babish YouTube and Marcella Hazan cookbook", "type": "multi_hop"},
            ],
        },
        {
            "dialog_id": "sample_005",
            "dialog": [
                {"speaker": "A", "text": "I'm preparing for the AWS Solutions Architect exam. Been studying for three months."},
                {"speaker": "B", "text": "When is your exam?"},
                {"speaker": "A", "text": "Next Friday. I'm most nervous about the networking section."},
                {"speaker": "B", "text": "What resources have you used?"},
                {"speaker": "A", "text": "Stephane Maarek's Udemy course and Adrian Cantrill's labs. Also the official practice exams."},
            ],
            "qa": [
                {"question": "What certification is the person studying for?", "answer": "AWS Solutions Architect", "type": "single_hop"},
                {"question": "Which part of the exam are they most nervous about?", "answer": "The networking section", "type": "single_hop"},
                {"question": "When is the exam and what courses are they using?", "answer": "Next Friday, using Stephane Maarek Udemy and Adrian Cantrill labs", "type": "multi_hop"},
            ],
        },
    ]


# ---------------------------------------------------------------------------
# Core benchmark logic
# ---------------------------------------------------------------------------

@dataclass
class LocomoResult:
    dialog_id: str
    qa_type: str
    question: str
    ground_truth: str
    retrieved_context: str
    score: int                          # 0-3
    latency_ms: float


@dataclass
class LocomoReport:
    total_qa: int = 0
    by_type: dict[str, list[int]] = field(default_factory=dict)
    avg_score: float = 0.0
    recall_at_1: float = 0.0           # score >= 2 counts as hit
    avg_latency_ms: float = 0.0
    results: list[LocomoResult] = field(default_factory=list)

    # mem0 published baseline for comparison
    MEM0_RECALL = 0.269
    MEM0_PRECISION = 0.305
    MEM0_AVG_SCORE = 1.8

    def summarize(self) -> dict[str, Any]:
        scores = [r.score for r in self.results]
        n = len(scores)
        if not n:
            return {"error": "no results"}

        hits = sum(1 for s in scores if s >= 2)
        avg = sum(scores) / n
        by_type_avg = {
            qt: round(sum(v) / len(v), 3)
            for qt, v in self.by_type.items()
        }

        return {
            "total_qa_pairs": n,
            "avg_judge_score": round(avg, 3),
            "recall_at_1_pct": round(hits / n * 100, 1),
            "by_type": by_type_avg,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "vs_mem0": {
                "our_avg_score": round(avg, 3),
                "mem0_avg_score": self.MEM0_AVG_SCORE,
                "delta": round(avg - self.MEM0_AVG_SCORE, 3),
                "our_recall_pct": round(hits / n * 100, 1),
                "mem0_recall_pct": round(self.MEM0_RECALL * 100, 1),
            },
        }


async def _ingest_dialog(dialog_id: str, turns: list[dict]) -> str:
    """Ingest all turns of a dialog as a single block. Returns user_id."""
    user_id = f"locomo-{dialog_id}"
    await upsert_user(user_id)

    # Concatenate the full conversation as context
    conversation = "\n".join(
        f"{t.get('speaker', 'User')}: {t.get('text', '')}"
        for t in turns
    )
    await route(
        Intent.INGEST,
        user_id=user_id,
        content=conversation,
        conversation_id=dialog_id,
    )
    return user_id


async def _run_qa(
    user_id: str,
    qa: dict,
    dialog_id: str,
    use_llm_judge: bool,
) -> LocomoResult:
    question = qa.get("question", "")
    ground_truth = qa.get("answer", "")
    qa_type = qa.get("type", "unknown")

    t0 = time.perf_counter()
    state = await route(Intent.RETRIEVE, user_id=user_id, query=question)
    latency_ms = (time.perf_counter() - t0) * 1000

    memories = state.get("final_memories") or []
    retrieved_context = "\n".join(m.content for m in memories[:5])

    if use_llm_judge:
        score = await llm_judge_score(question, ground_truth, retrieved_context)
    else:
        score = keyword_score(ground_truth, retrieved_context)

    return LocomoResult(
        dialog_id=dialog_id,
        qa_type=qa_type,
        question=question,
        ground_truth=ground_truth,
        retrieved_context=retrieved_context,
        score=score,
        latency_ms=latency_ms,
    )


async def run_locomo(
    split: str = "test",
    max_dialogs: int | None = 20,
    use_llm_judge: bool = False,
) -> dict[str, Any]:
    """
    Main entry point. Returns a summary dict comparable to mem0's published scores.

    Args:
        split: HuggingFace dataset split ("train" or "test")
        max_dialogs: Cap number of dialogs (None = full dataset)
        use_llm_judge: Use LLM scoring (slower but accurate); False = fast keyword match
    """
    rows = _load_locomo_hf(split=split, max_dialogs=max_dialogs)
    report = LocomoReport()
    total_latency = 0.0

    for row in rows:
        dialog_id = str(row.get("dialog_id", f"row_{id(row)}"))
        turns = row.get("dialog", [])
        qa_pairs = row.get("qa", [])

        if not turns or not qa_pairs:
            continue

        # Ingest entire conversation
        try:
            user_id = await _ingest_dialog(dialog_id, turns)
        except Exception as exc:
            logger.warning("locomo_ingest_failed", dialog_id=dialog_id, error=str(exc))
            continue

        # Run all QA pairs for this dialog
        for qa in qa_pairs:
            try:
                result = await _run_qa(user_id, qa, dialog_id, use_llm_judge)
                report.results.append(result)
                report.by_type.setdefault(result.qa_type, []).append(result.score)
                total_latency += result.latency_ms
            except Exception as exc:
                logger.warning("locomo_qa_failed", dialog_id=dialog_id, error=str(exc))

    n = len(report.results)
    report.avg_latency_ms = total_latency / max(n, 1)
    summary = report.summarize()
    logger.info("locomo_complete", **{k: v for k, v in summary.items() if k != "vs_mem0"})
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LOCOMO benchmark against Mnemosyne")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--max-dialogs", type=int, default=20)
    parser.add_argument("--use-llm-judge", action="store_true")
    args = parser.parse_args()

    result = asyncio.run(
        run_locomo(
            split=args.split,
            max_dialogs=args.max_dialogs,
            use_llm_judge=args.use_llm_judge,
        )
    )
    print(json.dumps(result, indent=2))

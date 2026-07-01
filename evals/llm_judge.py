"""
LLM-as-judge scorer shared across all benchmarks.

Uses the configured LLM provider to score a system's response against
a ground-truth answer on a 0-3 scale (same rubric as mem0's LOCOMO eval).
"""
from __future__ import annotations

import re

from llm.factory import get_llm_provider

_JUDGE_PROMPT = """\
You are evaluating a memory retrieval system. Given a question, the correct answer,
and the system's retrieved context, score the retrieval on this scale:

0 - The retrieved context does not contain the answer at all.
1 - The retrieved context vaguely hints at the answer but is mostly wrong or incomplete.
2 - The retrieved context contains the answer but with some noise or imprecision.
3 - The retrieved context clearly and directly contains the correct answer.

Question: {question}
Correct answer: {ground_truth}
Retrieved context: {retrieved_context}

Respond with ONLY a single integer (0, 1, 2, or 3). No explanation."""


async def llm_judge_score(
    question: str,
    ground_truth: str,
    retrieved_context: str,
) -> int:
    """
    Returns a score 0-3. Falls back to keyword match (0 or 3) if LLM unavailable.
    """
    try:
        llm = get_llm_provider()
        prompt = _JUDGE_PROMPT.format(
            question=question,
            ground_truth=ground_truth,
            retrieved_context=retrieved_context or "(empty)",
        )
        response = await llm.generate(prompt)
        match = re.search(r"[0-3]", response.strip())
        if match:
            return int(match.group())
        return 0
    except Exception:
        # Fallback: keyword presence check
        gt_lower = ground_truth.lower()
        ctx_lower = retrieved_context.lower()
        keywords = [w for w in gt_lower.split() if len(w) > 3]
        hits = sum(1 for kw in keywords if kw in ctx_lower)
        if not keywords:
            return 0
        ratio = hits / len(keywords)
        if ratio >= 0.7:
            return 3
        if ratio >= 0.3:
            return 1
        return 0


def keyword_score(ground_truth: str, retrieved_context: str) -> int:
    """Fast keyword-only fallback (no LLM call). Returns 0 or 3."""
    keywords = [w.lower() for w in ground_truth.split() if len(w) > 3]
    if not keywords:
        return 0
    hits = sum(1 for kw in keywords if kw in retrieved_context.lower())
    return 3 if hits / len(keywords) >= 0.5 else 0

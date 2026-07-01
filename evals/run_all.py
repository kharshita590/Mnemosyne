"""
Unified benchmark runner — executes all 5 benchmarks and prints a
consolidated report with Mnemosyne scores vs mem0 / supermemory baselines.

Usage:
    # Quick run (keyword scoring, small sample)
    python -m evals.run_all

    # Full run with LLM judge (slower, accurate)
    python -m evals.run_all --use-llm-judge --locomo-dialogs 50

    # Skip specific benchmarks
    python -m evals.run_all --skip locomo latency
"""
from __future__ import annotations

import argparse
import asyncio
import json
import textwrap
import time
from typing import Any


# ---------------------------------------------------------------------------
# Published baselines for comparison
# ---------------------------------------------------------------------------

BASELINES = {
    "mem0": {
        "locomo_avg_score": 1.8,
        "locomo_recall_pct": 26.9,
        "locomo_precision_pct": 30.5,
        "dedup_accuracy_pct": None,       # not published
        "conflict_resolution_pct": None,  # not published
        "p50_ms": None,
        "p99_ms": None,
    },
    "supermemory": {
        "locomo_avg_score": None,         # not publicly published
        "locomo_recall_pct": None,
        "locomo_precision_pct": None,
        "dedup_accuracy_pct": None,
        "conflict_resolution_pct": None,
        "p50_ms": None,
        "p99_ms": None,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section(title: str) -> None:
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def _row(label: str, value: Any, baseline: Any = None, higher_is_better: bool = True) -> str:
    val_str = f"{value}" if value is not None else "—"
    if baseline is not None and value is not None:
        try:
            delta = float(value) - float(baseline)
            sign = "+" if delta >= 0 else ""
            better = (delta >= 0) == higher_is_better
            arrow = "✓" if better else "✗"
            return f"  {label:<40} {val_str:>10}   (baseline: {baseline}, {sign}{delta:.1f}) {arrow}"
        except (TypeError, ValueError):
            pass
    return f"  {label:<40} {val_str:>10}"


def _print_report(results: dict[str, Any]) -> None:
    print("\n")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          MNEMOSYNE BENCHMARK REPORT                         ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    mem0 = BASELINES["mem0"]

    # ── LOCOMO ──────────────────────────────────────────────────────────────
    if "locomo" in results:
        _section("1. LOCOMO Benchmark  (vs mem0 published scores)")
        r = results["locomo"]
        if "error" in r:
            print(f"  ERROR: {r['error']}")
        else:
            vs = r.get("vs_mem0", {})
            print(_row("Avg judge score (0-3)", vs.get("our_avg_score"), mem0["locomo_avg_score"]))
            print(_row("Recall@1 %", vs.get("our_recall_pct"), mem0["locomo_recall_pct"]))
            print(_row("Total QA pairs evaluated", r.get("total_qa_pairs")))
            print(_row("Avg retrieval latency (ms)", r.get("avg_latency_ms"), higher_is_better=False))
            by_type = r.get("by_type", {})
            if by_type:
                print("\n  Breakdown by QA type:")
                for qt, score in by_type.items():
                    print(f"    {qt:<20} avg score: {score}")
    else:
        print("\n  [LOCOMO] skipped")

    # ── GOLDEN SET (Recall@5 + MRR) ─────────────────────────────────────────
    if "golden_set" in results:
        _section("2. Golden Set  (Recall@5 + MRR)")
        r = results["golden_set"]
        if "error" in r:
            print(f"  ERROR: {r['error']}")
        else:
            print(_row("Recall@5", r.get("recall_at_5")))
            print(_row("MRR", r.get("mrr")))
            print(_row("Total cases", r.get("n")))
            print(_row("Avg latency (ms)", r.get("avg_latency_ms"), higher_is_better=False))
            failures = r.get("failures", [])
            if failures:
                print(f"\n  Failed cases ({len(failures)}):")
                for f in failures[:5]:
                    print(f"    • {f['query'][:60]}")
                if len(failures) > 5:
                    print(f"    ... and {len(failures) - 5} more")
    else:
        print("\n  [Golden Set] skipped")

    # ── DEDUP ────────────────────────────────────────────────────────────────
    if "dedup" in results:
        _section("3. Deduplication Accuracy")
        r = results["dedup"]
        if "error" in r:
            print(f"  ERROR: {r['error']}")
        else:
            dd = r.get("dedup_accuracy", {})
            fd = r.get("false_dedup_prevention", {})
            print(_row("Duplicate compression accuracy %", dd.get("duplicate_accuracy_pct"), mem0["dedup_accuracy_pct"]))
            print(_row("Avg compression ratio (lower=better)", dd.get("avg_compression_ratio"), higher_is_better=False))
            print(_row("False-dedup rate %", fd.get("false_dedup_rate_pct"), higher_is_better=False))
            failures = r.get("failures", [])
            if failures:
                print(f"\n  Failed cases:")
                for f in failures:
                    print(f"    • [{f['case_id']}] stored={f['stored']}, expected={f['expected']}")
    else:
        print("\n  [Dedup] skipped")

    # ── CONFLICT RESOLUTION ──────────────────────────────────────────────────
    if "conflict" in results:
        _section("4. Conflict Resolution Accuracy")
        r = results["conflict"]
        if "error" in r:
            print(f"  ERROR: {r['error']}")
        else:
            print(_row("Resolution accuracy %", r.get("resolution_accuracy_pct"), mem0["conflict_resolution_pct"]))
            print(_row("Stale-value bleed rate %", r.get("stale_bleed_rate_pct"), higher_is_better=False))
            print(_row("Ambiguous (both values) %", r.get("ambiguous_pct"), higher_is_better=False))
            print(_row("Total conflict cases", r.get("total_cases")))
            failures = r.get("failures", [])
            if failures:
                print(f"\n  Failed cases:")
                for f in failures[:5]:
                    print(f"    • [{f['case_id']}] resolution={f['resolution']} — {f['description']}")
    else:
        print("\n  [Conflict] skipped")

    # ── LATENCY ──────────────────────────────────────────────────────────────
    if "latency" in results:
        _section("5. Latency Benchmark  (retrieval under concurrency)")
        r = results["latency"]
        if "error" in r:
            print(f"  ERROR: {r['error']}")
        else:
            lat = r.get("latency_ms", {})
            print(_row("P50 latency (ms)", lat.get("p50"), mem0["p50_ms"], higher_is_better=False))
            print(_row("P90 latency (ms)", lat.get("p90"), higher_is_better=False))
            print(_row("P99 latency (ms)", lat.get("p99"), mem0["p99_ms"], higher_is_better=False))
            print(_row("Avg latency (ms)", lat.get("avg"), higher_is_better=False))
            print(_row("StdDev (ms)", lat.get("stdev"), higher_is_better=False))
            print(_row("Effective throughput (QPS)", r.get("effective_throughput_qps")))
            print(_row("Concurrency level", r.get("concurrency")))
            print(_row("Total queries fired", r.get("total_queries")))
            failed = r.get("failed", 0)
            if failed:
                print(f"\n  {failed} queries failed — see 'failures' in JSON output")
    else:
        print("\n  [Latency] skipped")

    # ── Summary ───────────────────────────────────────────────────────────────
    _section("Summary")
    print(textwrap.dedent("""\
      mem0 published (LOCOMO):  avg score 1.8/3.0,  recall 26.9%
      supermemory:              scores not publicly published

      Run with --use-llm-judge for LLM-scored LOCOMO (most comparable to mem0).
      Run with --locomo-dialogs 100+ for statistically significant LOCOMO results.
    """))


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

async def run_all(
    skip: list[str],
    use_llm_judge: bool,
    locomo_dialogs: int,
    latency_concurrency: int,
    latency_queries: int,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    wall = time.perf_counter()

    if "golden_set" not in skip:
        print("[1/5] Running golden set (Recall@5 + MRR)...")
        from evals.harness import run_eval
        try:
            results["golden_set"] = await run_eval()
        except Exception as exc:
            results["golden_set"] = {"error": str(exc)}

    if "locomo" not in skip:
        print(f"[2/5] Running LOCOMO benchmark ({locomo_dialogs} dialogs, llm_judge={use_llm_judge})...")
        from evals.locomo import run_locomo
        try:
            results["locomo"] = await run_locomo(
                max_dialogs=locomo_dialogs,
                use_llm_judge=use_llm_judge,
            )
        except Exception as exc:
            results["locomo"] = {"error": str(exc)}

    if "dedup" not in skip:
        print("[3/5] Running deduplication accuracy benchmark...")
        from evals.dedup_bench import run_dedup_bench
        try:
            results["dedup"] = await run_dedup_bench()
        except Exception as exc:
            results["dedup"] = {"error": str(exc)}

    if "conflict" not in skip:
        print("[4/5] Running conflict resolution accuracy benchmark...")
        from evals.conflict_bench import run_conflict_bench
        try:
            results["conflict"] = await run_conflict_bench()
        except Exception as exc:
            results["conflict"] = {"error": str(exc)}

    if "latency" not in skip:
        print(f"[5/5] Running latency benchmark (concurrency={latency_concurrency}, queries={latency_queries})...")
        from evals.latency_bench import run_latency_bench
        try:
            results["latency"] = await run_latency_bench(
                concurrency=latency_concurrency,
                total_queries=latency_queries,
            )
        except Exception as exc:
            results["latency"] = {"error": str(exc)}

    results["wall_clock_seconds"] = round(time.perf_counter() - wall, 1)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all Mnemosyne benchmarks")
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=["golden_set", "locomo", "dedup", "conflict", "latency"],
        help="Benchmarks to skip",
    )
    parser.add_argument("--use-llm-judge", action="store_true", help="Use LLM for LOCOMO scoring")
    parser.add_argument("--locomo-dialogs", type=int, default=20, help="Number of LOCOMO dialogs")
    parser.add_argument("--latency-concurrency", type=int, default=20)
    parser.add_argument("--latency-queries", type=int, default=100)
    parser.add_argument("--json", action="store_true", help="Print raw JSON output only")
    args = parser.parse_args()

    results = asyncio.run(
        run_all(
            skip=args.skip,
            use_llm_judge=args.use_llm_judge,
            locomo_dialogs=args.locomo_dialogs,
            latency_concurrency=args.latency_concurrency,
            latency_queries=args.latency_queries,
        )
    )

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        _print_report(results)
        print(f"\nTotal wall clock: {results['wall_clock_seconds']}s\n")


if __name__ == "__main__":
    main()

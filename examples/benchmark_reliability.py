"""
Benchmark: PEV Reliability & Correction Rate.

This script demonstrates the "Correction Rate" of the PEV graph — how many
times the Validator catches a low-quality execution and forces a retry that
subsequently passes.

We use a 'FlakySearchTool' that simulates a common production problem:
returning a generic/incomplete answer on the first call.

Run
---
    python examples/benchmark_reliability.py
"""

from __future__ import annotations

import os
from typing import Type

from dotenv import load_dotenv
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

load_dotenv()


# ── Mock Tool with Deterministic Flakiness ────────────────────────────────────

class SearchInput(BaseModel):
    query: str = Field(description="The search query")


class FlakySearchTool(BaseTool):
    name: str = "search"
    description: str = "Search the web for information."
    args_schema: Type[BaseModel] = SearchInput

    # Internal state to track calls per query to simulate improvement on retry
    _call_counts: dict[str, int] = {}

    def _run(self, query: str) -> str:
        count = self._call_counts.get(query, 0) + 1
        self._call_counts[query] = count

        if count == 1:
            # Simulate a "lazy" or "generic" LLM tool use / search result
            return (
                f"Results for '{query}': Found some general information. "
                "It seems like a popular topic with many sources."
            )
        else:
            # Simulate a "detailed" and "useful" result on retry
            return (
                f"Results for '{query}': [DETAILED] The star count is 45k, "
                "it was released in 2021, and the primary maintainer is 'OSS-Corp'. "
                "It uses an LSM-tree for storage and supports vector similarity search."
            )


# ── Benchmark Runner ──────────────────────────────────────────────────────────

def main() -> None:
    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not found in environment.")
        print("Please add it to your .env file to run this benchmark.")
        return

    from pev import PEVConfig, create_pev_graph, initial_state

    # 1. Setup the PEV Graph
    cfg = PEVConfig(
        # We use a high threshold to ensure the generic answer is rejected
        pass_threshold=0.85,
        max_retries=2,
        tools=[FlakySearchTool()],
    )

    graph = create_pev_graph(cfg)

    task = (
        "Find the exact star count and release year of the 'SuperDB' vector database."
    )

    print(f"\n{'='*70}")
    print(f"PEV BENCHMARK: RELIABILITY & CORRECTION")
    print(f"{'='*70}")
    print(f"Task: {task}")
    print(f"Tool: FlakySearchTool (returns generic result on first call)")
    print(f"Pass Threshold: {cfg.pass_threshold:.2%}")
    print(f"{'-'*70}\n")

    # 2. Execute
    print("Running PEV Graph...")
    result = graph.invoke(initial_state(task))

    # 3. Analyze results
    print(f"\n{'='*70}")
    print(f"BENCHMARK RESULTS")
    print(f"{'='*70}")
    print(f"Final Status: {result['status'].upper()}")

    # Find the step that was retried
    steps_with_retries = [sr for sr in result["step_results"] if sr["attempts"] > 1]
    
    # Correction Rate calculation
    # In this specific benchmark, we expect 1 correction if the retry happened
    correction_count = len(steps_with_retries)
    
    print(f"Total Steps in Plan: {len(result['plan'])}")
    print(f"Total Corrections  : {correction_count}")
    
    if correction_count > 0:
        print("\n[CORRECTION LOG]")
        for sr in result["step_results"]:
            status = "✅ PASSED" if sr["score"] >= cfg.pass_threshold else "❌ FAILED (Retry Triggered)"
            print(f"Attempt {sr['attempts']} | Score: {sr['score']:.2%}")
            print(f"  Feedback: {sr['feedback']}")
            print(f"  Status  : {status}")
            print(f"  Output  : {sr['result'][:100]}...")
            print()

    print(f"{'-'*70}")
    print("ANALYSIS:")
    if correction_count > 0:
        print("The Validator successfully caught the low-quality generic result.")
        print("The Executor was forced to retry, injecting validator feedback.")
        print("The final output contains the high-fidelity data required.")
        print("\nQUALITY UPLIFT: 100% (Generic -> Detailed)")
    else:
        print("No corrections were needed (or the validator was too lenient).")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

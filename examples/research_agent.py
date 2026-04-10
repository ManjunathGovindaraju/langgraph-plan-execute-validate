"""Example: Research agent using Plan → Execute → Validate.

This example uses the Tavily search tool to research a topic,
validate each step's quality, and retry automatically if the
validator scores the result below the threshold.

Setup
-----
    pip install tavily-python
    export ANTHROPIC_API_KEY=sk-ant-...
    export TAVILY_API_KEY=tvly-...

Run
---
    python examples/research_agent.py
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    # Lazy import so the example file is importable without all deps installed
    from langchain_community.tools.tavily_search import TavilySearchResults

    from pev import PEVConfig, create_pev_graph, initial_state

    cfg = PEVConfig(
        executor_model="claude-sonnet-4-6",
        planner_model="claude-haiku-4-5-20251001",
        validator_model="claude-haiku-4-5-20251001",
        pass_threshold=0.80,
        max_retries=2,
        max_replans=1,
        tools=[TavilySearchResults(max_results=3)],
    )

    graph = create_pev_graph(cfg)

    task = (
        "Research the top 3 open-source vector databases by GitHub stars. "
        "For each one, provide: name, star count, primary use case, and "
        "one sentence on what makes it unique."
    )

    print(f"\n{'=' * 60}")
    print(f"Task: {task}")
    print(f"{'=' * 60}\n")

    result = graph.invoke(initial_state(task))

    # ── Print results ──────────────────────────────────────────────────────────
    print(f"Status : {result['status']}")
    print(f"Steps  : {len(result['plan'])}")
    print(f"Replans: {result['replan_count']}")
    print()

    for i, sr in enumerate(result["step_results"], 1):
        print(f"Step {i}: {sr['step']}")
        print(f"  Score   : {sr['score']:.0%}")
        print(f"  Attempts: {sr['attempts']}")
        print(f"  Result  : {sr['result'][:300]}...")
        print()

    if result["status"] == "failed":
        print(f"Error: {result['error']}")


if __name__ == "__main__":
    main()

"""Example: Code review agent with strict validation.

Uses a high pass_threshold (0.90) to demonstrate the retry mechanism.
The validator is strict about code review completeness — the executor
must produce a thorough review or it will be retried with feedback.

Setup
-----
    export ANTHROPIC_API_KEY=sk-ant-...

Run
---
    python examples/code_review_agent.py
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()


CODE_SNIPPET = '''
async def process_batch(items: list[dict]) -> list[dict]:
    results = []
    for item in items:
        conn = await asyncpg.connect(DATABASE_URL)
        row = await conn.fetchrow(
            f"SELECT * FROM records WHERE id = {item['id']}"
        )
        results.append(dict(row))
        await conn.close()
    return results
'''


def main() -> None:
    from pev import PEVConfig, create_pev_graph, initial_state

    cfg = PEVConfig(
        pass_threshold=0.90,
        max_retries=2,
        max_replans=1,
        tools=[],
    )

    graph = create_pev_graph(cfg)

    task = (
        f"Perform a thorough code review of this Python function:\n\n"
        f"```python{CODE_SNIPPET}```\n\n"
        "Cover: correctness, security vulnerabilities, performance issues, "
        "and provide a corrected version."
    )

    print(f"\nTask:\n{task[:200]}...\n{'='*60}\n")

    result = graph.invoke(initial_state(task))

    print(f"Status  : {result['status']}")
    print(f"Replans : {result['replan_count']}")
    total_attempts = sum(sr["attempts"] for sr in result["step_results"])
    print(f"Total attempts across all steps: {total_attempts}\n")

    for i, sr in enumerate(result["step_results"], 1):
        print(f"── Step {i} (attempt {sr['attempts']}): {sr['step']}")
        print(f"   Score: {sr['score']:.0%}")
        if sr["score"] < cfg.pass_threshold:
            print(f"   Feedback: {sr['feedback']}")
        print(f"   {sr['result'][:500]}\n")


if __name__ == "__main__":
    main()

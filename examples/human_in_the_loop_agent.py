"""Human-in-the-loop example — interrupt before replan and/or before each step.

Demonstrates two HITL modes:

  1. interrupt_before_replan — pause when the router decides to replan.
     Human can approve or inject additional guidance into the replan prompt.

  2. interrupt_before_step — pause before every executor step.
     Human can approve as-is or override the step text.

Requirements
------------
    uv sync
    cp .env.example .env  # add ANTHROPIC_API_KEY

Run
---
    python examples/human_in_the_loop_agent.py
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from pev import PEVConfig, create_pev_graph, initial_state

# ── 1. Interrupt before replan ────────────────────────────────────────────────


def demo_interrupt_before_replan() -> None:
    """Run with a strict threshold to force a replan, then intercept it."""
    print("\n" + "=" * 60)
    print("Demo: interrupt_before_replan")
    print("=" * 60)

    cfg = PEVConfig(
        pass_threshold=0.99,   # intentionally strict to force a replan quickly
        max_retries=0,
        max_replans=1,
        interrupt_before_replan=True,
    )

    graph = create_pev_graph(cfg, checkpointer=MemorySaver())
    thread = {"configurable": {"thread_id": "hitl-replan-demo"}}
    state = initial_state("Summarise the key differences between LangChain and LlamaIndex")

    print("\n[1] Starting graph run...")
    result = graph.invoke(state, config=thread)

    # Check whether the graph interrupted (it returns state, not a final result)
    graph_state = graph.get_state(thread)
    if graph_state.next:
        print(f"\n[INTERRUPTED] Graph paused before: {graph_state.next}")

        # Inspect what the router surfaced
        interrupted_values = graph_state.values
        print(f"  Failed step : {interrupted_values.get('plan', ['?'])[interrupted_values.get('current_step_idx', 0)]}")
        print(f"  Score       : {interrupted_values.get('validation_score', 0):.0%}")
        print(f"  Feedback    : {interrupted_values.get('validation_feedback', '')}")

        # Human decision: provide guidance and let it replan
        human_guidance = "Focus on practical use-case differences, not just API surface."
        print(f"\n[HUMAN] Approving replan with guidance: '{human_guidance}'")

        result = graph.invoke(Command(resume=human_guidance), config=thread)
        print(f"\n[DONE] Status: {result['status']}")
    else:
        print(f"\n[DONE] Status: {result['status']} (no interrupt triggered)")

    for sr in result.get("step_results", []):
        print(f"  [{sr['score']:.0%}] {sr['step'][:60]}  (attempts: {sr['attempts']})")


# ── 2. Interrupt before each step ─────────────────────────────────────────────


def demo_interrupt_before_step() -> None:
    """Pause before each step — human approves or modifies the step text."""
    print("\n" + "=" * 60)
    print("Demo: interrupt_before_step")
    print("=" * 60)

    cfg = PEVConfig(
        pass_threshold=0.75,
        interrupt_before_step=True,
    )

    graph = create_pev_graph(cfg, checkpointer=MemorySaver())
    thread = {"configurable": {"thread_id": "hitl-step-demo"}}
    state = initial_state("List the top 3 Python web frameworks with a one-line description each")

    print("\n[1] Starting graph run (will pause before each step)...")
    result = graph.invoke(state, config=thread)

    step_num = 0
    while True:
        graph_state = graph.get_state(thread)
        if not graph_state.next:
            break  # graph finished

        interrupted = graph_state.values
        current_idx = interrupted.get("current_step_idx", 0)
        plan = interrupted.get("plan", [])
        current_step = plan[current_idx] if current_idx < len(plan) else "?"

        step_num += 1
        print(f"\n[INTERRUPTED] Step {current_idx + 1}/{len(plan)}: '{current_step}'")

        # Simulate human reviewing and approving (or modifying) the step
        if step_num == 1:
            # Approve first step as-is
            print("[HUMAN] Approved as-is.")
            result = graph.invoke(Command(resume=None), config=thread)
        else:
            # Override subsequent steps with more specific instructions
            override = f"{current_step} — include GitHub star count and latest version"
            print(f"[HUMAN] Overriding with: '{override}'")
            result = graph.invoke(Command(resume=override), config=thread)

    print(f"\n[DONE] Status: {result['status']}")
    for sr in result.get("step_results", []):
        print(f"  [{sr['score']:.0%}] {sr['step'][:70]}  (attempts: {sr['attempts']})")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo_interrupt_before_replan()
    demo_interrupt_before_step()

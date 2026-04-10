"""PEV graph — wires the Plan → Execute → Validate StateGraph.

Graph topology
──────────────

    START
      │
    planner ◄──────────────────────────────────────────────┐
      │                                                     │ replan
    executor ◄─────────────────────────────────┐           │
      │                                         │ retry     │
    validator                                   │           │
      │                                         │           │
    _router ─── score ≥ threshold, more steps ──┘(next)    │
              ─── score ≥ threshold, last step  ──► END (complete)
              ─── score < threshold, retry left ──► executor
              ─── score < threshold, retry gone ──► planner (replan)
              ─── all limits exhausted           ──► END (failed)

Router node
───────────
LangGraph's conditional edges can only return routing strings — they
cannot update state.  We need to advance current_step_idx and reset
retry_count when a step passes.  The solution is a thin "_router" node
that does the state bookkeeping and then a single unconditional edge
from _router to "_dispatch", which reads state["_next"] to decide the
real next node.

Simpler alternative used here: use a conditional edge that reads from
a "_next" field written by a router node.

Human-in-the-loop
─────────────────
When cfg.interrupt_before_replan is True, the router calls interrupt()
before triggering a replan.  The graph pauses and the caller resumes via:

    graph.invoke(Command(resume="optional guidance"), config=thread_config)

When cfg.interrupt_before_step is True, the executor calls interrupt()
before running each step.  Requires a checkpointer passed to create_pev_graph().
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from pev.config import PEVConfig
from pev.nodes.executor import make_executor_node
from pev.nodes.planner import make_planner_node
from pev.nodes.validator import make_validator_node
from pev.state import PEVState

# ── Routing decision ───────────────────────────────────────────────────────────

RouteDecision = Literal["execute", "retry", "replan", "complete", "failed"]


def _make_router(cfg: PEVConfig):
    """Returns a router *node* that writes "_next" into state and also.

    Performs the step-advance / retry-reset bookkeeping.

    Why a node and not a conditional edge function?
    Conditional edge functions are read-only — they cannot update state.
    We need to increment current_step_idx and reset retry_count when a
    step passes.  A node gives us both routing control and state writes.
    """

    def router_node(state: PEVState) -> dict:  # type: ignore[type-arg]
        score = state.get("validation_score", 0.0)
        retry_count = state.get("retry_count", 0)
        replan_count = state.get("replan_count", 0)
        idx = state.get("current_step_idx", 0)
        plan = state.get("plan", [])

        if score >= cfg.pass_threshold:
            next_idx = idx + 1
            if next_idx >= len(plan):
                # All steps passed — we're done
                return {"status": "complete", "_next": "complete"}
            # Advance to the next step, reset retry counter
            return {
                "current_step_idx": next_idx,
                "retry_count": 0,
                "status": "executing",
                "_next": "execute",
            }

        # Step failed — decide how to recover
        if retry_count < cfg.max_retries:
            return {
                "retry_count": retry_count + 1,
                "status": "executing",
                "_next": "retry",
            }

        if replan_count < cfg.max_replans:
            # ── Human-in-the-loop: replan approval ──────────────────────────
            human_feedback: str | None = None
            if cfg.interrupt_before_replan:
                human_input: Any = interrupt(
                    {
                        "type": "replan_approval",
                        "failed_step": plan[idx] if idx < len(plan) else "",
                        "score": score,
                        "validator_feedback": state.get("validation_feedback", ""),
                        "message": (
                            f"Step '{plan[idx]}' failed (score: {score:.0%}). "
                            "Resume to approve replanning, or provide guidance as a string."
                        ),
                    }
                )
                if isinstance(human_input, str) and human_input.strip():
                    human_feedback = human_input.strip()

            return {
                "status": "planning",
                "_next": "replan",
                "human_feedback": human_feedback,
            }

        # All recovery options exhausted
        return {
            "status": "failed",
            "error": (
                f"Step '{plan[idx]}' failed after {retry_count} retries "
                f"and {replan_count} replans "
                f"(final score: {score:.2f})."
            ),
            "_next": "failed",
        }

    return router_node


def _dispatch(state: PEVState) -> str:
    """Conditional edge that reads state["_next"] written by the router node.

    Maps the decision string to an actual node name.
    """
    decision: str = state.get("_next", "failed")  # type: ignore[assignment]
    mapping = {
        "execute": "executor",
        "retry": "executor",
        "replan": "planner",
        "complete": END,
        "failed": END,
    }
    return mapping.get(decision, END)


# ── Graph construction ─────────────────────────────────────────────────────────


def create_pev_graph(cfg: PEVConfig | None = None, checkpointer: Any = None) -> Any:
    """Build and compile the Plan → Execute → Validate graph.

    Parameters
    ----------
    cfg:
        Runtime configuration.  Defaults to PEVConfig() if not provided.
    checkpointer:
        A LangGraph checkpointer (e.g. ``MemorySaver()``) required when
        ``cfg.interrupt_before_replan`` or ``cfg.interrupt_before_step`` is
        True.  For non-HITL runs, omit this parameter.

    Returns:
    -------
    A compiled LangGraph ``StateGraph`` ready to ``.invoke()`` or
    ``.ainvoke()``.

    Basic usage::

        from pev import create_pev_graph, PEVConfig

        graph = create_pev_graph(PEVConfig(pass_threshold=0.85))
        result = graph.invoke(initial_state("Research the top 3 vector databases"))

    With LangSmith tracing::

        cfg = PEVConfig(run_name="research-agent", run_tags=["prod"])
        graph = create_pev_graph(cfg)
        result = graph.invoke(state, config=cfg.run_config())

    With human-in-the-loop::

        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.types import Command

        cfg = PEVConfig(interrupt_before_replan=True)
        graph = create_pev_graph(cfg, checkpointer=MemorySaver())

        thread = {"configurable": {"thread_id": "run-1"}}
        result = graph.invoke(initial_state("..."), config=thread)
        # If interrupted:
        result = graph.invoke(Command(resume="try a different approach"), config=thread)
    """
    if cfg is None:
        cfg = PEVConfig()

    needs_checkpointer = cfg.interrupt_before_replan or cfg.interrupt_before_step
    if needs_checkpointer and checkpointer is None:
        warnings.warn(
            "interrupt_before_replan or interrupt_before_step is enabled but no "
            "checkpointer was provided. Human-in-the-loop requires a checkpointer. "
            "Pass checkpointer=MemorySaver() to create_pev_graph().",
            UserWarning,
            stacklevel=2,
        )

    builder = StateGraph(PEVState)

    # ── Register nodes ─────────────────────────────────────────────────────
    builder.add_node("planner", make_planner_node(cfg))
    builder.add_node("executor", make_executor_node(cfg))
    builder.add_node("validator", make_validator_node(cfg))
    builder.add_node("router", _make_router(cfg))

    # ── Edges ──────────────────────────────────────────────────────────────
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "executor")
    builder.add_edge("executor", "validator")
    builder.add_edge("validator", "router")

    # The router node writes "_next"; _dispatch reads it to choose the branch
    builder.add_conditional_edges("router", _dispatch)

    return builder.compile(checkpointer=checkpointer)


# ── Convenience: build initial state ──────────────────────────────────────────


def initial_state(task: str) -> PEVState:
    """Return a fully-initialised PEVState for *task*.

    Saves callers from spelling out every key manually.

    Example::

        state = initial_state("Summarise the latest LangGraph release notes")
        result = graph.invoke(state)
    """
    return PEVState(
        task=task,
        plan=[],
        current_step_idx=0,
        pending_result="",
        step_results=[],
        validation_score=0.0,
        validation_feedback="",
        retry_count=0,
        replan_count=0,
        human_feedback=None,
        status="planning",
        error=None,
        _next="",  # type: ignore[typeddict-unknown-key]
    )

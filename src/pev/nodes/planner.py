"""
Planner node — breaks the task into an ordered list of steps.

On the first call it plans from scratch.  On subsequent calls (replanning)
it receives the validator's failure feedback and the steps attempted so far,
and produces a revised plan.

Uses structured output so the LLM is forced to return valid JSON — no
fragile string parsing.
"""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from pev.config import PEVConfig
from pev.prompts import PLANNER_REPLAN_SUFFIX, PLANNER_SYSTEM
from pev.state import PEVState


class PlanOutput(BaseModel):
    """Structured output schema enforced on the planner LLM."""

    steps: list[str]


def make_planner_node(cfg: PEVConfig):
    """
    Factory that returns a planner node closed over *cfg*.

    Using a factory (rather than reading config from RunnableConfig) keeps
    the node signature simple and the config explicit at graph-build time.
    """
    llm = ChatAnthropic(model=cfg.planner_model).with_structured_output(PlanOutput)  # type: ignore[arg-type]

    def planner_node(state: PEVState) -> dict:  # type: ignore[type-arg]
        is_replan = bool(state.get("plan"))  # non-empty plan → this is a revision

        # Build the human message
        if is_replan:
            past_steps = "\n".join(
                f"  • {r['step']} → score {r['score']:.0%}"
                for r in state.get("step_results", [])
            ) or "  (none)"

            suffix = PLANNER_REPLAN_SUFFIX.format(
                feedback=state.get("validation_feedback", "No feedback provided."),
                past_steps=past_steps,
            )
            human_content = f"Task: {state['task']}\n{suffix}"
        else:
            human_content = f"Task: {state['task']}"

        messages = [
            SystemMessage(content=PLANNER_SYSTEM),
            HumanMessage(content=human_content),
        ]

        result: PlanOutput = llm.invoke(messages)

        return {
            "plan": result.steps,
            "current_step_idx": 0,
            "retry_count": 0,
            "replan_count": state.get("replan_count", 0) + (1 if is_replan else 0),
            "pending_result": "",
            "validation_score": 0.0,
            "validation_feedback": "",
            "status": "executing",
            "error": None,
        }

    return planner_node

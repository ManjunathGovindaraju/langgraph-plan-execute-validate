"""
Validator node — scores the executor's output for the current step.

Reads state["pending_result"], sends it to the validator LLM with the
step description and overall task for context, and receives a structured
score (0.0–1.0) + feedback string.

On completion it:
  - Appends a complete StepResult (with score + feedback) to step_results
  - Writes validation_score and validation_feedback to top-level state
  - Sets status to "validating" (the router decides what happens next)

Uses structured output so the score is always a valid float — no
fragile string parsing.
"""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, field_validator

from pev.config import PEVConfig
from pev.prompts import VALIDATOR_HUMAN, VALIDATOR_SYSTEM
from pev.state import PEVState, StepResult


class ValidationOutput(BaseModel):
    """Structured output schema enforced on the validator LLM."""

    score: float
    feedback: str

    @field_validator("score")
    @classmethod
    def clamp_score(cls, v: float) -> float:
        """Clamp to [0.0, 1.0] even if the LLM overshoots."""
        return max(0.0, min(1.0, v))


def make_validator_node(cfg: PEVConfig):
    """Factory that returns a validator node closed over *cfg*."""

    llm = ChatAnthropic(model=cfg.validator_model).with_structured_output(ValidationOutput)  # type: ignore[arg-type]

    def validator_node(state: PEVState) -> dict:  # type: ignore[type-arg]
        current_step = state["plan"][state["current_step_idx"]]
        pending = state.get("pending_result", "")

        messages = [
            SystemMessage(content=VALIDATOR_SYSTEM),
            HumanMessage(
                content=VALIDATOR_HUMAN.format(
                    task=state["task"],
                    step=current_step,
                    result=pending or "(no output produced)",
                )
            ),
        ]

        output: ValidationOutput = llm.invoke(messages)

        # Clamp in the node itself — Pydantic validator is bypassed by mocks
        score = max(0.0, min(1.0, output.score))

        # Build a complete StepResult and append it to the audit trail
        step_result: StepResult = {
            "step": current_step,
            "result": pending,
            "score": score,
            "feedback": output.feedback,
            "attempts": state.get("retry_count", 0) + 1,
        }

        return {
            "step_results": [step_result],  # operator.add appends this
            "pending_result": "",           # consumed — clear it
            "validation_score": score,
            "validation_feedback": output.feedback,
            "status": "validating",         # router decides next transition
        }

    return validator_node

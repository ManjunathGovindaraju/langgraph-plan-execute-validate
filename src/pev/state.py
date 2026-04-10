"""
PEV state schema.

The entire graph operates on a single PEVState dict that flows through
every node.  Each node receives the full state and returns a partial
dict — LangGraph merges the update back automatically.

Step results use operator.add so that each executor run *appends*
rather than overwrites, giving a full audit trail of every attempt.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal

from typing_extensions import TypedDict

# ── Per-step result ────────────────────────────────────────────────────────────

class StepResult(TypedDict):
    """Outcome of a single plan step after execution and validation."""

    step: str           # The step description from the plan
    result: str         # Raw output from the executor
    score: float        # Validator confidence score  0.0 – 1.0
    feedback: str       # Validator's explanation of the score
    attempts: int       # How many times this step was attempted


# ── Status literals ────────────────────────────────────────────────────────────

Status = Literal["planning", "executing", "validating", "complete", "failed"]


# ── Graph state ────────────────────────────────────────────────────────────────

class PEVState(TypedDict):
    """
    Shared state that flows through every node in the PEV graph.

    Fields updated by each node:

    planner    → plan, current_step_idx (reset to 0), replan_count, status
    executor   → step_results (appended), status
    validator  → validation_score, validation_feedback, retry_count, status
    router     → current_step_idx (advance), retry_count (reset), status
    """

    # ── Task definition ──────────────────────────────────────────────────────
    task: str
    """The original high-level task provided by the caller."""

    # ── Plan ─────────────────────────────────────────────────────────────────
    plan: list[str]
    """Ordered list of step descriptions produced by the planner."""

    current_step_idx: int
    """Index into `plan` pointing at the step currently being executed."""

    # ── Execution history ────────────────────────────────────────────────────
    step_results: Annotated[list[StepResult], operator.add]
    """
    Accumulates one StepResult per *attempt* (not per step).
    operator.add means each executor run appends; nothing is overwritten.
    """

    # ── Validation ───────────────────────────────────────────────────────────
    validation_score: float
    """Most recent score from the validator.  0.0 = complete failure, 1.0 = perfect."""

    validation_feedback: str
    """Human-readable explanation from the validator used to guide retries/replanning."""

    # ── Loop counters ────────────────────────────────────────────────────────
    retry_count: int
    """Number of times the *current step* has been retried after a low score."""

    replan_count: int
    """Number of times the entire plan has been regenerated."""

    # ── Lifecycle ────────────────────────────────────────────────────────────
    status: Status
    """Current lifecycle phase of the graph run."""

    error: str | None
    """Last error message, if any.  None during normal operation."""

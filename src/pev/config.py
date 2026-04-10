"""PEVConfig — runtime configuration for the Plan → Execute → Validate graph.

Design rationale
----------------
Three separate model slots (planner / executor / validator) let callers
assign a cheap model to the bookkeeping phases (planning & scoring) and
reserve the capable model for the executor where tool use and reasoning
quality actually matter.  In production this cuts per-run cost by ~60-70 %
compared to routing every phase through the same model.

Example usage
-------------
    from pev import PEVConfig

    # Defaults — Claude haiku for planner/validator, sonnet for executor
    config = PEVConfig()

    # Override everything
    config = PEVConfig(
        executor_model="claude-opus-4-5-20251001",
        pass_threshold=0.90,
        max_retries=3,
        max_replans=2,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.tools import BaseTool


@dataclass
class PEVConfig:
    """Configuration for the PEV graph models, thresholds, and limits."""

    # ── Model routing ────────────────────────────────────────────────────────
    planner_model: str = "claude-haiku-4-5-20251001"
    """Model used by the Planner node.
    Cheap model is fine — the planner only produces structured JSON output."""

    executor_model: str = "claude-sonnet-4-6"
    """Model used by the Executor node.
    Use your most capable model here — this is where tool calls and
    multi-step reasoning happen."""

    validator_model: str = "claude-haiku-4-5-20251001"
    """Model used by the Validator node.
    Cheap model is fine — the validator only scores and explains."""

    # ── Quality thresholds ───────────────────────────────────────────────────
    pass_threshold: float = 0.80
    """Minimum validation score [0.0 – 1.0] for a step to be considered passing.
    Steps scoring below this trigger a retry or replan."""

    # ── Loop guards ─────────────────────────────────────────────────────────
    max_retries: int = 2
    """Maximum retry attempts for a *single step* before escalating to replan."""

    max_replans: int = 1
    """Maximum full replanning cycles before the graph marks the run as failed."""

    # ── Tools ────────────────────────────────────────────────────────────────
    tools: list[BaseTool] = field(default_factory=list)
    """LangChain tools available to the Executor node.
    The Planner and Validator nodes never receive tools."""

    # ── Validation ───────────────────────────────────────────────────────────
    def __post_init__(self) -> None:
        """Validate configuration parameters after initialisation."""
        if not (0.0 < self.pass_threshold <= 1.0):
            raise ValueError(f"pass_threshold must be in (0.0, 1.0], got {self.pass_threshold}")
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")
        if self.max_replans < 0:
            raise ValueError(f"max_replans must be >= 0, got {self.max_replans}")

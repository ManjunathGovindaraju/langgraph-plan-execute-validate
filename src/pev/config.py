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

    # With LangSmith tracing metadata
    config = PEVConfig(
        run_name="research-agent",
        run_tags=["production", "v1"],
        run_metadata={"user_id": "u-123", "environment": "prod"},
    )
    result = graph.invoke(state, config=config.run_config())

    # With human-in-the-loop
    from langgraph.checkpoint.memory import MemorySaver
    config = PEVConfig(interrupt_before_replan=True)
    graph = create_pev_graph(config, checkpointer=MemorySaver())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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

    # ── LangSmith tracing ────────────────────────────────────────────────────
    run_name: str | None = None
    """Display name for this run in LangSmith.  Useful for grouping experiments."""

    run_tags: list[str] = field(default_factory=list)
    """Tags attached to every LangSmith trace for this run (e.g. ["prod", "v2"])."""

    run_metadata: dict[str, Any] = field(default_factory=dict)
    """Arbitrary key-value metadata attached to the LangSmith trace
    (e.g. {"user_id": "u-123", "environment": "prod"})."""

    # ── Human-in-the-loop ────────────────────────────────────────────────────
    interrupt_before_replan: bool = False
    """If True, pause execution before triggering a full replan and wait for
    human input.  Requires a checkpointer passed to create_pev_graph().

    Resume via:
        graph.invoke(Command(resume="optional guidance"), config=thread_config)

    The resume value (if a non-empty string) is injected into the replan prompt
    as additional human guidance for the Planner."""

    interrupt_before_step: bool = False
    """If True, pause before each executor step and wait for human approval.
    Requires a checkpointer passed to create_pev_graph().

    Resume via:
        graph.invoke(Command(resume=None), config=thread_config)        # approve as-is
        graph.invoke(Command(resume="revised step text"), config=...)   # override step

    If a non-empty string is returned, it replaces the planned step text."""

    # ── Validation ───────────────────────────────────────────────────────────
    def __post_init__(self) -> None:
        """Validate configuration parameters after initialisation."""
        if not (0.0 < self.pass_threshold <= 1.0):
            raise ValueError(f"pass_threshold must be in (0.0, 1.0], got {self.pass_threshold}")
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")
        if self.max_replans < 0:
            raise ValueError(f"max_replans must be >= 0, got {self.max_replans}")

    def run_config(self) -> dict[str, Any]:
        """Return a RunnableConfig dict to pass to graph.invoke() or graph.ainvoke().

        Populates LangSmith run_name, tags, and metadata when set.  Safe to call
        even when LangSmith is not configured — unused keys are ignored.

        Usage::

            result = graph.invoke(state, config=cfg.run_config())

        For human-in-the-loop runs, merge with your thread config::

            thread = {"configurable": {"thread_id": "run-1"}}
            config = {**cfg.run_config(), **thread}
            result = graph.invoke(state, config=config)
        """
        config: dict[str, Any] = {}
        if self.run_name:
            config["run_name"] = self.run_name
        if self.run_tags:
            config["tags"] = list(self.run_tags)
        if self.run_metadata:
            config["metadata"] = dict(self.run_metadata)
        return config

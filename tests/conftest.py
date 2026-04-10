"""Shared fixtures for the PEV test suite.

All unit tests use mock LLMs — zero API calls, zero cost, fast (~1s).
Integration tests (marked `slow`) use real APIs and are excluded from CI.

Mock strategy
─────────────
We patch ChatAnthropic at the pev.nodes.* import level so the factories
receive a fake LLM that returns predictable structured responses.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pev.config import PEVConfig
from pev.state import PEVState, StepResult

# ── State builders ─────────────────────────────────────────────────────────────


def make_state(**overrides) -> PEVState:
    """Return a fully-initialised PEVState with optional field overrides."""
    base: PEVState = {
        "task": "Write a haiku about async programming",
        "plan": ["Draft the haiku", "Review syllable count", "Polish final version"],
        "current_step_idx": 0,
        "pending_result": "",
        "step_results": [],
        "validation_score": 0.0,
        "validation_feedback": "",
        "retry_count": 0,
        "replan_count": 0,
        "status": "executing",
        "error": None,
        "_next": "",  # type: ignore[typeddict-unknown-key]
    }
    base.update(overrides)  # type: ignore[typeddict-item]
    return base


def make_step_result(**overrides) -> StepResult:
    """Return a StepResult with sensible defaults and optional overrides."""
    base: StepResult = {
        "step": "Draft the haiku",
        "result": "Async calls await / Futures bloom in parallel / Event loop dreams",
        "score": 0.9,
        "feedback": "Good syllable structure, vivid imagery.",
        "attempts": 1,
    }
    base.update(overrides)  # type: ignore[typeddict-item]
    return base


# ── Config fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def default_cfg() -> PEVConfig:
    return PEVConfig(pass_threshold=0.80, max_retries=2, max_replans=1)


@pytest.fixture
def strict_cfg() -> PEVConfig:
    """High threshold — most results fail, useful for testing retry/replan paths."""
    return PEVConfig(pass_threshold=0.95, max_retries=1, max_replans=1)


@pytest.fixture
def zero_retry_cfg() -> PEVConfig:
    """Zero retries — any failure immediately triggers replan."""
    return PEVConfig(pass_threshold=0.80, max_retries=0, max_replans=1)


@pytest.fixture
def zero_replan_cfg() -> PEVConfig:
    """Zero replans — any exhausted retry goes straight to failed."""
    return PEVConfig(pass_threshold=0.80, max_retries=0, max_replans=0)


# ── Mock LLM helpers ───────────────────────────────────────────────────────────


def mock_llm_returning(content: str) -> MagicMock:
    """Return a MagicMock that behaves like a bound ChatAnthropic LLM.

    Calling .invoke() returns an AIMessage-like object whose .content
    equals *content*.  Calling .with_structured_output() returns a mock
    whose .invoke() returns the object passed to it directly (simulating
    structured output by returning the Pydantic model instance).
    """
    response = MagicMock()
    response.content = content
    response.tool_calls = []

    llm = MagicMock()
    llm.invoke.return_value = response
    llm.bind_tools.return_value = llm

    # with_structured_output returns a mock that yields a Pydantic instance
    structured = MagicMock()
    llm.with_structured_output.return_value = structured

    return llm


# ── Planner mock ───────────────────────────────────────────────────────────────


@pytest.fixture
def mock_planner_llm():
    """Patches ChatAnthropic in the planner module to return a 3-step plan."""
    from pydantic import BaseModel

    class _PlanOutput(BaseModel):
        steps: list[str]

    plan_output = _PlanOutput(steps=["Step A", "Step B", "Step C"])

    with patch("pev.nodes.planner.ChatAnthropic") as mock_cls:
        llm = MagicMock()
        structured = MagicMock()
        structured.invoke.return_value = plan_output
        llm.with_structured_output.return_value = structured
        mock_cls.return_value = llm
        yield structured


# ── Executor mock ──────────────────────────────────────────────────────────────


@pytest.fixture
def mock_executor_llm():
    """Patches ChatAnthropic in the executor module to return a text result."""
    with patch("pev.nodes.executor.ChatAnthropic") as mock_cls:
        response = MagicMock()
        response.content = "Executor output: task completed successfully."
        response.tool_calls = []

        llm = MagicMock()
        llm.invoke.return_value = response
        llm.bind_tools.return_value = llm
        mock_cls.return_value = llm
        yield llm


# ── Validator mock ─────────────────────────────────────────────────────────────


def make_validator_mock(score: float, feedback: str = "Looks good."):
    """Context manager that patches ChatAnthropic in the validator module.

    Usage:
        with make_validator_mock(score=0.9) as mock:
            ...
    """
    from pydantic import BaseModel

    class _ValidationOutput(BaseModel):
        score: float
        feedback: str

    output = _ValidationOutput(score=score, feedback=feedback)

    mock_ctx = patch("pev.nodes.validator.ChatAnthropic")

    class _Ctx:
        def __enter__(self):
            mock_cls = mock_ctx.__enter__()
            llm = MagicMock()
            structured = MagicMock()
            structured.invoke.return_value = output
            llm.with_structured_output.return_value = structured
            mock_cls.return_value = llm
            return structured

        def __exit__(self, *args):
            mock_ctx.__exit__(*args)

    return _Ctx()

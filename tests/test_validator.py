"""
Unit tests for the validator node.

Tests cover:
- Score and feedback written to state
- StepResult appended to step_results with correct fields
- pending_result cleared after validation
- Score clamped to [0.0, 1.0] even if LLM overshoots
- Status set to "validating" (router decides next step)
- Attempt count carried from retry_count
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from pev.config import PEVConfig
from pev.nodes.validator import make_validator_node
from tests.conftest import make_state


class _ValidationOutput(BaseModel):
    score: float
    feedback: str


def _patch_validator(score: float, feedback: str = "Test feedback."):
    output = _ValidationOutput(score=score, feedback=feedback)
    mock_ctx = patch("pev.nodes.validator.ChatAnthropic")

    class _Ctx:
        def __enter__(self_inner):
            MockCls = mock_ctx.__enter__()
            llm = MagicMock()
            structured = MagicMock()
            structured.invoke.return_value = output
            llm.with_structured_output.return_value = structured
            MockCls.return_value = llm
            return structured

        def __exit__(self_inner, *args):
            mock_ctx.__exit__(*args)

    return _Ctx()


# ── Score and feedback ─────────────────────────────────────────────────────────

def test_validator_writes_score(default_cfg: PEVConfig):
    state = make_state(pending_result="some output")

    with _patch_validator(score=0.85):
        result = make_validator_node(default_cfg)(state)

    assert result["validation_score"] == pytest.approx(0.85)


def test_validator_writes_feedback(default_cfg: PEVConfig):
    state = make_state(pending_result="some output")

    with _patch_validator(score=0.7, feedback="Missing detail on X."):
        result = make_validator_node(default_cfg)(state)

    assert result["validation_feedback"] == "Missing detail on X."


def test_validator_clamps_score_above_1(default_cfg: PEVConfig):
    """Score > 1.0 from LLM must be clamped to 1.0."""
    state = make_state(pending_result="great output")

    with _patch_validator(score=1.5):
        result = make_validator_node(default_cfg)(state)

    assert result["validation_score"] == pytest.approx(1.0)


def test_validator_clamps_score_below_0(default_cfg: PEVConfig):
    """Score < 0.0 from LLM must be clamped to 0.0."""
    state = make_state(pending_result="terrible output")

    with _patch_validator(score=-0.3):
        result = make_validator_node(default_cfg)(state)

    assert result["validation_score"] == pytest.approx(0.0)


# ── StepResult audit trail ────────────────────────────────────────────────────

def test_validator_appends_step_result(default_cfg: PEVConfig):
    state = make_state(pending_result="executor output here")

    with _patch_validator(score=0.9, feedback="Well done."):
        result = make_validator_node(default_cfg)(state)

    assert len(result["step_results"]) == 1
    sr = result["step_results"][0]
    assert sr["result"] == "executor output here"
    assert sr["score"] == pytest.approx(0.9)
    assert sr["feedback"] == "Well done."


def test_validator_records_step_description(default_cfg: PEVConfig):
    state = make_state(
        plan=["Draft the haiku", "Review syllables"],
        current_step_idx=0,
        pending_result="output",
    )

    with _patch_validator(score=0.8):
        result = make_validator_node(default_cfg)(state)

    assert result["step_results"][0]["step"] == "Draft the haiku"


def test_validator_records_attempt_count(default_cfg: PEVConfig):
    """attempts = retry_count + 1."""
    state = make_state(pending_result="output", retry_count=2)

    with _patch_validator(score=0.8):
        result = make_validator_node(default_cfg)(state)

    assert result["step_results"][0]["attempts"] == 3


# ── Handoff cleanup ────────────────────────────────────────────────────────────

def test_validator_clears_pending_result(default_cfg: PEVConfig):
    state = make_state(pending_result="some pending content")

    with _patch_validator(score=0.9):
        result = make_validator_node(default_cfg)(state)

    assert result["pending_result"] == ""


def test_validator_sets_status_validating(default_cfg: PEVConfig):
    """Validator always sets status=validating; router decides what's next."""
    state = make_state(pending_result="output")

    with _patch_validator(score=0.9):
        result = make_validator_node(default_cfg)(state)

    assert result["status"] == "validating"

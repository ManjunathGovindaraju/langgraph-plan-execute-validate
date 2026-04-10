"""
Unit tests for the planner node.

Tests cover:
- Initial planning (empty plan) produces correct state updates
- Replanning (non-empty plan) increments replan_count and injects feedback
- Plan steps are stored correctly
- State fields are reset on (re)plan
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from pydantic import BaseModel

from pev.config import PEVConfig
from pev.nodes.planner import make_planner_node
from tests.conftest import make_state


class _PlanOutput(BaseModel):
    steps: list[str]


def _patch_planner(steps: list[str]):
    """Patch ChatAnthropic in planner module to return *steps*."""
    output = _PlanOutput(steps=steps)

    mock_ctx = patch("pev.nodes.planner.ChatAnthropic")

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


# ── Initial planning ───────────────────────────────────────────────────────────

def test_planner_sets_plan_on_first_call(default_cfg: PEVConfig):
    steps = ["Research topic", "Write draft", "Review"]
    state = make_state(plan=[])  # empty → first plan

    with _patch_planner(steps):
        node = make_planner_node(default_cfg)
        result = node(state)

    assert result["plan"] == steps


def test_planner_resets_idx_on_first_call(default_cfg: PEVConfig):
    state = make_state(plan=[], current_step_idx=2)  # simulated mid-run

    with _patch_planner(["A", "B"]):
        result = make_planner_node(default_cfg)(state)

    assert result["current_step_idx"] == 0


def test_planner_resets_retry_count_on_first_call(default_cfg: PEVConfig):
    state = make_state(plan=[], retry_count=2)

    with _patch_planner(["A"]):
        result = make_planner_node(default_cfg)(state)

    assert result["retry_count"] == 0


def test_planner_does_not_increment_replan_count_on_first_call(default_cfg: PEVConfig):
    state = make_state(plan=[], replan_count=0)

    with _patch_planner(["A"]):
        result = make_planner_node(default_cfg)(state)

    assert result["replan_count"] == 0


def test_planner_sets_status_executing(default_cfg: PEVConfig):
    state = make_state(plan=[])

    with _patch_planner(["A"]):
        result = make_planner_node(default_cfg)(state)

    assert result["status"] == "executing"


def test_planner_clears_error_field(default_cfg: PEVConfig):
    state = make_state(plan=[], error="previous error")

    with _patch_planner(["A"]):
        result = make_planner_node(default_cfg)(state)

    assert result["error"] is None


# ── Replanning ─────────────────────────────────────────────────────────────────

def test_replan_increments_replan_count(default_cfg: PEVConfig):
    # Non-empty plan → this is a replan
    state = make_state(plan=["Old step"], replan_count=0)

    with _patch_planner(["New step A", "New step B"]):
        result = make_planner_node(default_cfg)(state)

    assert result["replan_count"] == 1


def test_replan_replaces_plan(default_cfg: PEVConfig):
    state = make_state(plan=["Old step 1", "Old step 2"], replan_count=0)
    new_steps = ["Revised step 1", "Revised step 2", "Revised step 3"]

    with _patch_planner(new_steps):
        result = make_planner_node(default_cfg)(state)

    assert result["plan"] == new_steps


def test_replan_resets_step_idx(default_cfg: PEVConfig):
    state = make_state(plan=["Old step"], current_step_idx=1, replan_count=0)

    with _patch_planner(["New A", "New B"]):
        result = make_planner_node(default_cfg)(state)

    assert result["current_step_idx"] == 0


def test_replan_resets_retry_count(default_cfg: PEVConfig):
    state = make_state(plan=["Old step"], retry_count=2, replan_count=0)

    with _patch_planner(["New A"]):
        result = make_planner_node(default_cfg)(state)

    assert result["retry_count"] == 0

"""Unit tests for router node routing logic.

This is the most important test file — the router is the brain of the
retry/replan system.  Tests cover every branch of the decision tree:

  score ≥ threshold + more steps  → "execute"   (advance idx)
  score ≥ threshold + last step   → "complete"
  score < threshold + retry left  → "retry"      (increment retry_count)
  score < threshold + no retry    → "replan"     (if replans left)
  score < threshold + no retry    → "failed"     (if no replans left)
"""

from __future__ import annotations

from pev.config import PEVConfig
from pev.graph import _dispatch, _make_router
from tests.conftest import make_state

# ── Helpers ────────────────────────────────────────────────────────────────────


def route(state, cfg: PEVConfig) -> tuple[dict, str]:
    """Run the router node and the dispatch edge; return (state_update, node_name)."""
    router = _make_router(cfg)
    update = router(state)
    merged = {**state, **update}
    destination = _dispatch(merged)
    return update, destination


# ── Passing step — more steps remain ──────────────────────────────────────────


def test_passing_step_advances_to_next(default_cfg: PEVConfig):
    state = make_state(
        plan=["A", "B", "C"],
        current_step_idx=0,
        validation_score=0.9,
    )
    update, dest = route(state, default_cfg)

    assert update["current_step_idx"] == 1
    assert dest == "executor"


def test_passing_step_resets_retry_count(default_cfg: PEVConfig):
    state = make_state(
        plan=["A", "B"],
        current_step_idx=0,
        validation_score=0.9,
        retry_count=2,
    )
    update, _ = route(state, default_cfg)

    assert update["retry_count"] == 0


def test_passing_step_sets_status_executing(default_cfg: PEVConfig):
    state = make_state(
        plan=["A", "B"],
        current_step_idx=0,
        validation_score=0.9,
    )
    update, _ = route(state, default_cfg)

    assert update["status"] == "executing"


# ── Passing step — final step ──────────────────────────────────────────────────


def test_passing_final_step_routes_complete(default_cfg: PEVConfig):
    state = make_state(
        plan=["Only step"],
        current_step_idx=0,
        validation_score=0.9,
    )
    update, dest = route(state, default_cfg)

    assert update["status"] == "complete"
    assert dest == "__end__"


def test_passing_last_step_of_multi_plan_routes_complete(default_cfg: PEVConfig):
    state = make_state(
        plan=["A", "B", "C"],
        current_step_idx=2,  # last index
        validation_score=0.9,
    )
    update, dest = route(state, default_cfg)

    assert update["status"] == "complete"
    assert dest == "__end__"


# ── Failing step — retry available ────────────────────────────────────────────


def test_failing_step_retries_when_retries_available(default_cfg: PEVConfig):
    state = make_state(
        plan=["A", "B"],
        current_step_idx=0,
        validation_score=0.5,  # below 0.80 threshold
        retry_count=0,
    )
    update, dest = route(state, default_cfg)

    assert update["retry_count"] == 1
    assert dest == "executor"


def test_failing_step_increments_retry_count(default_cfg: PEVConfig):
    state = make_state(validation_score=0.3, retry_count=1)
    update, _ = route(state, default_cfg)

    assert update["retry_count"] == 2


def test_failing_step_sets_status_executing_on_retry(default_cfg: PEVConfig):
    state = make_state(validation_score=0.3, retry_count=0)
    update, _ = route(state, default_cfg)

    assert update["status"] == "executing"


# ── Failing step — retries exhausted, replan available ────────────────────────


def test_exhausted_retries_triggers_replan(default_cfg: PEVConfig):
    # default_cfg has max_retries=2
    state = make_state(
        validation_score=0.3,
        retry_count=2,  # at max
        replan_count=0,
    )
    update, dest = route(state, default_cfg)

    assert update["status"] == "planning"
    assert dest == "planner"


def test_zero_retry_cfg_immediately_replans(zero_retry_cfg: PEVConfig):
    state = make_state(
        validation_score=0.3,
        retry_count=0,
        replan_count=0,
    )
    update, dest = route(state, zero_retry_cfg)

    assert dest == "planner"


# ── Failing step — all limits exhausted → failed ──────────────────────────────


def test_all_limits_exhausted_routes_failed(default_cfg: PEVConfig):
    state = make_state(
        plan=["step X"],
        current_step_idx=0,
        validation_score=0.1,
        retry_count=2,  # at max_retries
        replan_count=1,  # at max_replans
    )
    update, dest = route(state, default_cfg)

    assert update["status"] == "failed"
    assert dest == "__end__"


def test_failed_state_includes_error_message(default_cfg: PEVConfig):
    state = make_state(
        plan=["step X"],
        current_step_idx=0,
        validation_score=0.1,
        retry_count=2,
        replan_count=1,
    )
    update, _ = route(state, default_cfg)

    assert update["error"] is not None
    assert "step X" in update["error"]
    assert "0.10" in update["error"]


def test_zero_retry_zero_replan_fails_immediately(zero_replan_cfg: PEVConfig):
    state = make_state(
        validation_score=0.1,
        retry_count=0,
        replan_count=0,
    )
    update, dest = route(state, zero_replan_cfg)

    assert update["status"] == "failed"
    assert dest == "__end__"


# ── Threshold boundary ─────────────────────────────────────────────────────────


def test_score_exactly_at_threshold_passes(default_cfg: PEVConfig):
    """Score == threshold should pass (≥ not >)."""
    state = make_state(
        plan=["A", "B"],
        current_step_idx=0,
        validation_score=0.80,  # exactly at default threshold
    )
    update, dest = route(state, default_cfg)

    assert dest == "executor"  # advance, not retry


def test_score_just_below_threshold_fails(default_cfg: PEVConfig):
    state = make_state(
        plan=["A", "B"],
        current_step_idx=0,
        validation_score=0.799,
        retry_count=0,
    )
    update, dest = route(state, default_cfg)

    assert update["retry_count"] == 1  # retry triggered

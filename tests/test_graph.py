"""Graph-level integration tests (mock LLM — no API calls).

Tests cover:
- Graph compiles with default and custom configs
- initial_state() produces a valid, fully-keyed state dict
- PEVConfig validation raises on bad inputs
- Graph structure: correct nodes and edges registered
"""

from __future__ import annotations

import pytest

from pev import PEVConfig, create_pev_graph, initial_state
from pev.graph import _dispatch

# ── Config validation ──────────────────────────────────────────────────────────

def test_config_rejects_threshold_above_1():
    with pytest.raises(ValueError, match="pass_threshold"):
        PEVConfig(pass_threshold=1.1)


def test_config_rejects_threshold_at_zero():
    with pytest.raises(ValueError, match="pass_threshold"):
        PEVConfig(pass_threshold=0.0)


def test_config_rejects_negative_max_retries():
    with pytest.raises(ValueError, match="max_retries"):
        PEVConfig(max_retries=-1)


def test_config_rejects_negative_max_replans():
    with pytest.raises(ValueError, match="max_replans"):
        PEVConfig(max_replans=-1)


def test_config_accepts_valid_values():
    cfg = PEVConfig(pass_threshold=0.75, max_retries=3, max_replans=2)
    assert cfg.pass_threshold == 0.75
    assert cfg.max_retries == 3
    assert cfg.max_replans == 2


# ── Graph compilation ──────────────────────────────────────────────────────────

def test_graph_compiles_with_defaults():
    graph = create_pev_graph()
    assert graph is not None


def test_graph_compiles_with_custom_config():
    cfg = PEVConfig(pass_threshold=0.9, max_retries=1, max_replans=0)
    graph = create_pev_graph(cfg)
    assert graph is not None


def test_graph_has_expected_nodes():
    graph = create_pev_graph()
    nodes = set(graph.get_graph().nodes.keys())
    assert {"planner", "executor", "validator", "router"}.issubset(nodes)


# ── initial_state ──────────────────────────────────────────────────────────────

def test_initial_state_sets_task():
    state = initial_state("My task")
    assert state["task"] == "My task"


def test_initial_state_has_all_required_keys():
    state = initial_state("test")
    required = {
        "task", "plan", "current_step_idx", "pending_result",
        "step_results", "validation_score", "validation_feedback",
        "retry_count", "replan_count", "status", "error",
    }
    assert required.issubset(set(state.keys()))


def test_initial_state_empty_plan():
    state = initial_state("test")
    assert state["plan"] == []
    assert state["step_results"] == []


def test_initial_state_zero_counters():
    state = initial_state("test")
    assert state["current_step_idx"] == 0
    assert state["retry_count"] == 0
    assert state["replan_count"] == 0
    assert state["validation_score"] == 0.0


def test_initial_state_status_planning():
    state = initial_state("test")
    assert state["status"] == "planning"


def test_initial_state_no_error():
    state = initial_state("test")
    assert state["error"] is None


# ── Dispatch edge ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("next_val,expected_dest", [
    ("execute",  "executor"),
    ("retry",    "executor"),
    ("replan",   "planner"),
    ("complete", "__end__"),
    ("failed",   "__end__"),
    ("unknown",  "__end__"),   # fallback
])
def test_dispatch_maps_correctly(next_val: str, expected_dest: str):
    from tests.conftest import make_state
    state = make_state(_next=next_val)  # type: ignore[call-arg]
    assert _dispatch(state) == expected_dest

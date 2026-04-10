"""
Unit tests for the executor node.

Tests cover:
- Result written to pending_result
- Status set to "validating"
- Retry feedback injected into prompt when retry_count > 0
- Context from passing steps injected when step_results exist
- Tool call loop executes tools and feeds results back
- Unknown tool name produces error message (not exception)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from pev.config import PEVConfig
from pev.nodes.executor import make_executor_node
from tests.conftest import make_state, make_step_result


def _patch_executor(response_content: str, tool_calls=None):
    """Patch ChatAnthropic in executor module."""
    mock_ctx = patch("pev.nodes.executor.ChatAnthropic")

    class _Ctx:
        def __enter__(self_inner):
            MockCls = mock_ctx.__enter__()
            response = MagicMock()
            response.content = response_content
            response.tool_calls = tool_calls or []

            llm = MagicMock()
            llm.invoke.return_value = response
            llm.bind_tools.return_value = llm
            MockCls.return_value = llm
            self_inner.llm = llm
            self_inner.response = response
            return self_inner

        def __exit__(self_inner, *args):
            mock_ctx.__exit__(*args)

    return _Ctx()


# ── Basic execution ────────────────────────────────────────────────────────────

def test_executor_writes_pending_result(default_cfg: PEVConfig):
    state = make_state()

    with _patch_executor("Step A output here."):
        result = make_executor_node(default_cfg)(state)

    assert result["pending_result"] == "Step A output here."


def test_executor_sets_status_validating(default_cfg: PEVConfig):
    state = make_state()

    with _patch_executor("done"):
        result = make_executor_node(default_cfg)(state)

    assert result["status"] == "validating"


def test_executor_uses_current_step(default_cfg: PEVConfig):
    """Executor should pick state['plan'][current_step_idx]."""
    state = make_state(
        plan=["Step 0", "Step 1", "Step 2"],
        current_step_idx=1,
    )

    with _patch_executor("output") as ctx:
        make_executor_node(default_cfg)(state)
        # Verify the human message content contains the right step
        invoke_call = ctx.llm.invoke.call_args
        messages = invoke_call[0][0]
        # messages = [SystemMessage, HumanMessage]; index 1 is the human prompt
        human_content = messages[1].content
        assert "Step 1" in human_content


# ── Context injection ──────────────────────────────────────────────────────────

def test_executor_injects_passing_context(default_cfg: PEVConfig):
    """Passing step results should appear in the executor's prompt."""
    passing = make_step_result(
        step="Step A",
        result="Step A was done well.",
        score=0.9,
    )
    state = make_state(
        plan=["Step A", "Step B"],
        current_step_idx=1,
        step_results=[passing],
    )

    with _patch_executor("output") as ctx:
        make_executor_node(default_cfg)(state)
        messages = ctx.llm.invoke.call_args[0][0]
        human_content = messages[1].content
        assert "Step A" in human_content


def test_executor_excludes_failing_context(default_cfg: PEVConfig):
    """Failed step results (score below threshold) should NOT feed forward."""
    failing = make_step_result(step="Step A", result="bad output", score=0.3)
    state = make_state(
        plan=["Step A", "Step B"],
        current_step_idx=1,
        step_results=[failing],
    )

    with _patch_executor("output") as ctx:
        make_executor_node(default_cfg)(state)
        messages = ctx.llm.invoke.call_args[0][0]
        human_content = messages[1].content
        assert "bad output" not in human_content


def test_executor_injects_retry_feedback(default_cfg: PEVConfig):
    """On retry, validator feedback must appear in the prompt."""
    state = make_state(
        retry_count=1,
        validation_score=0.5,
        validation_feedback="Missing the key metric.",
    )

    with _patch_executor("retry output") as ctx:
        make_executor_node(default_cfg)(state)
        messages = ctx.llm.invoke.call_args[0][0]
        human_content = messages[1].content
        assert "Missing the key metric." in human_content


# ── Tool calls ─────────────────────────────────────────────────────────────────

def test_executor_calls_registered_tool(default_cfg: PEVConfig):
    """If the LLM returns a tool call, the tool should be invoked."""
    tool = MagicMock()
    tool.name = "search"
    tool.invoke.return_value = "search results"
    cfg = PEVConfig(tools=[tool])

    tool_call = {"name": "search", "args": {"query": "python"}, "id": "tc_1"}

    with patch("pev.nodes.executor.ChatAnthropic") as MockCls:
        # First response has tool call; second response is final text
        first_response = MagicMock()
        first_response.content = ""
        first_response.tool_calls = [tool_call]

        final_response = MagicMock()
        final_response.content = "Based on results: python is great."
        final_response.tool_calls = []

        llm = MagicMock()
        llm.invoke.side_effect = [first_response, final_response]
        llm.bind_tools.return_value = llm
        MockCls.return_value = llm

        result = make_executor_node(cfg)(make_state())

    tool.invoke.assert_called_once_with({"query": "python"})
    assert "python is great" in result["pending_result"]


def test_executor_handles_unknown_tool_gracefully(default_cfg: PEVConfig):
    """An unregistered tool name should produce an error string, not an exception."""
    tool_call = {"name": "nonexistent_tool", "args": {}, "id": "tc_2"}

    with patch("pev.nodes.executor.ChatAnthropic") as MockCls:
        first = MagicMock()
        first.content = ""
        first.tool_calls = [tool_call]

        final = MagicMock()
        final.content = "Handled gracefully."
        final.tool_calls = []

        llm = MagicMock()
        llm.invoke.side_effect = [first, final]
        llm.bind_tools.return_value = llm
        MockCls.return_value = llm

        # Should not raise
        result = make_executor_node(default_cfg)(make_state())

    assert result["pending_result"] == "Handled gracefully."

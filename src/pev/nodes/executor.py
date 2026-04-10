"""Executor node — runs a single plan step using available tools.

The executor implements a tool-call loop:
  1. Send step + context to the LLM (with tools bound)
  2. If the LLM returns tool calls, execute them and feed results back
  3. Repeat until the LLM returns a final text response (no tool calls)

The raw text result is written to state["pending_result"].  The validator
reads it next and appends a scored StepResult to state["step_results"].

The tool-call loop is capped at MAX_TOOL_ROUNDS to prevent runaway agents.
"""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import interrupt

from pev.config import PEVConfig
from pev.prompts import (
    EXECUTOR_CONTEXT_HEADER,
    EXECUTOR_HUMAN,
    EXECUTOR_RETRY_NOTICE,
    EXECUTOR_SYSTEM,
)
from pev.state import PEVState

MAX_TOOL_ROUNDS = 10  # hard cap on tool-call iterations per step


def make_executor_node(cfg: PEVConfig):
    """Factory that returns an executor node closed over *cfg*."""
    tools_by_name: dict[str, BaseTool] = {t.name: t for t in cfg.tools}
    llm = ChatAnthropic(model=cfg.executor_model)
    llm_with_tools = llm.bind_tools(cfg.tools) if cfg.tools else llm  # type: ignore[arg-type]

    def executor_node(state: PEVState) -> dict:  # type: ignore[type-arg]
        current_step = state["plan"][state["current_step_idx"]]
        retry_count = state.get("retry_count", 0)

        # ── Human-in-the-loop: step approval ────────────────────────────────
        if cfg.interrupt_before_step:
            idx = state["current_step_idx"]
            total = len(state["plan"])
            human_input = interrupt({
                "type": "step_approval",
                "step": current_step,
                "step_number": idx + 1,
                "total_steps": total,
                "message": (
                    f"Step {idx + 1}/{total}: '{current_step}'\n"
                    "Resume with None to approve, or a string to override the step."
                ),
            })
            # Non-empty string → human overrode the step text
            if isinstance(human_input, str) and human_input.strip():
                current_step = human_input.strip()

        # ── Build context string ─────────────────────────────────────────────
        context = ""

        # Inject passing step results as context for later steps
        passing = [r for r in state.get("step_results", []) if r["score"] >= cfg.pass_threshold]
        if passing:
            completed_lines = "\n".join(
                f"  Step {i + 1}: {r['step']}\n  Result: {r['result'][:400]}"
                for i, r in enumerate(passing)
            )
            context += EXECUTOR_CONTEXT_HEADER.format(completed=completed_lines)

        # Inject validator feedback on retries
        if retry_count > 0 and state.get("validation_feedback"):
            context += EXECUTOR_RETRY_NOTICE.format(
                score=state.get("validation_score", 0.0),
                feedback=state["validation_feedback"],
            )

        human_content = EXECUTOR_HUMAN.format(context=context, step=current_step)

        messages = [
            SystemMessage(content=EXECUTOR_SYSTEM),
            HumanMessage(content=human_content),
        ]

        # ── Tool-call loop ───────────────────────────────────────────────────
        for _ in range(MAX_TOOL_ROUNDS):
            response = llm_with_tools.invoke(messages)
            messages.append(response)  # type: ignore[arg-type]

            if not getattr(response, "tool_calls", None):
                break  # LLM returned final answer — exit loop

            # Execute each tool call and feed results back
            for tool_call in response.tool_calls:  # type: ignore[union-attr]
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]

                if tool_name not in tools_by_name:
                    tool_output = f"Error: tool '{tool_name}' is not registered."
                else:
                    try:
                        tool_output = str(tools_by_name[tool_name].invoke(tool_args))
                    except Exception as exc:  # noqa: BLE001
                        tool_output = f"Error executing '{tool_name}': {exc}"

                messages.append(ToolMessage(content=tool_output, tool_call_id=tool_id))

        # ── Extract final text result ────────────────────────────────────────
        final = messages[-1]
        content = getattr(final, "content", "")
        if isinstance(content, list):
            # Anthropic sometimes returns content as a list of blocks
            result_text = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        else:
            result_text = str(content)

        return {
            "pending_result": result_text.strip(),
            "status": "validating",
        }

    return executor_node

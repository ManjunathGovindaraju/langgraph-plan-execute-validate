"""
Example: PEV + FastMCP — The "Full-Stack AI" Architecture.

This script demonstrates how to connect a PEV orchestrator (this repo) to
an MCP server built with your 'fastmcp-production-template'.

Architecture
------------
1. Orchestrator (PEV): Handles Planning, Execution Loop, and Validation.
2. Capability (MCP): Provides standardized tools (DB, API, Files) via FastMCP.

Setup
-----
    pip install langchain-mcp-adapters
    # Ensure your MCP server is reachable (e.g., via 'uv run server.py')

Run
---
    python examples/mcp_agent.py
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    # 1. Import MCP Adapter and PEV components
    from langchain_mcp_adapters.tools import load_mcp_tools

    from pev import PEVConfig, create_pev_graph, initial_state

    # 2. Connect to your MCP Server
    # Replace the path with your actual fastmcp-production-template server
    MCP_SERVER_PATH = "path/to/your/fastmcp-production-template/src/server/main.py"

    print(f"Connecting to MCP Server: {MCP_SERVER_PATH}...")

    try:
        # This loads all tools defined in your FastMCP server as LangChain BaseTools
        mcp_tools = load_mcp_tools(
            "uv",
            ["run", MCP_SERVER_PATH],
        )
        print(f"Successfully loaded {len(mcp_tools)} tools from MCP server.")
    except Exception as e:
        print(f"\n[Note] MCP Server not found at path: {MCP_SERVER_PATH}")
        print("This is expected if you haven't cloned the other repo locally yet.")
        print("To run this for real, update the MCP_SERVER_PATH variable.")
        mcp_tools = []

    # 3. Configure the PEV Graph with MCP Tools
    cfg = PEVConfig(
        planner_model="claude-haiku-4-5-20251001",
        executor_model="claude-sonnet-4-6",
        validator_model="claude-haiku-4-5-20251001",
        pass_threshold=0.85,  # Strict quality gate for MCP tool output
        tools=mcp_tools,
    )

    graph = create_pev_graph(cfg)

    # 4. Run a task that requires your MCP tools
    task = "Use the MCP tools to query the production database for active user counts."

    print(f"\nTask: {task}")
    print("PEV is now ready to Plan, Execute (via MCP), and Validate the results.")

    # result = graph.invoke(initial_state(task))
    # print(f"Final Status: {result['status']}")


if __name__ == "__main__":
    main()

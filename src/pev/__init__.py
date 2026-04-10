"""pev-langgraph — Plan → Execute → Validate for LangGraph.

Public API
----------
    from pev import create_pev_graph, initial_state, PEVConfig, PEVState

    graph = create_pev_graph(PEVConfig(pass_threshold=0.85))
    result = graph.invoke(initial_state("Your task here"))

Human-in-the-loop
-----------------
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    cfg = PEVConfig(interrupt_before_replan=True)
    graph = create_pev_graph(cfg, checkpointer=MemorySaver())

    thread = {"configurable": {"thread_id": "run-1"}}
    result = graph.invoke(initial_state("Your task"), config=thread)
    # If interrupted: graph.invoke(Command(resume="optional guidance"), config=thread)
"""

from pev.config import PEVConfig
from pev.graph import create_pev_graph, initial_state
from pev.state import PEVState, Status, StepResult

__version__ = "0.1.0"

__all__ = [
    "PEVConfig",
    "PEVState",
    "StepResult",
    "Status",
    "create_pev_graph",
    "initial_state",
    "__version__",
]

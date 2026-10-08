from langgraph.checkpoint.postgres.base import BasePostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.postgres.base import BasePostgresStore
from langgraph.types import RetryPolicy

from chatbot.graphs.state import Context, State
from chatbot.graphs.v1.nodes import call_model, call_tool, should_continue
from chatbot.settings import S


def build_agent(checkpointer: BasePostgresSaver, store: BasePostgresStore):
    """Construct and compile the agent graph."""

    # Define a new graph with state
    workflow = StateGraph(state_schema=State, context_schema=Context)

    # Define the nodes
    workflow.add_node("llm", call_model, retry_policy=RetryPolicy(max_attempts=S.MAX_ATTEMPTS, initial_interval=1.0))
    workflow.add_node("tools", call_tool, retry_policy=RetryPolicy(max_attempts=S.MAX_ATTEMPTS, initial_interval=1.0))

    # Define the edges
    workflow.add_edge(START, "llm")
    workflow.add_edge("tools", "llm")
    workflow.add_conditional_edges(
        "llm",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )

    return workflow.compile(name="agent", checkpointer=checkpointer, store=store)

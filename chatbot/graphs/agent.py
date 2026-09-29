from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from chatbot.graphs.nodes import call_model, call_tool, should_continue
from chatbot.graphs.state import Context, State
from chatbot.db.session import async_pool
from chatbot.settings import S


def create_agent_graph(max_attempts: int, checkpointer: BaseCheckpointSaver):
    """Construct and compile the agent graph."""

    # Define a new graph with state
    workflow = StateGraph(state_schema=State, context_schema=Context)

    # Define the nodes
    workflow.add_node("llm", call_model, retry_policy=RetryPolicy(max_attempts=max_attempts, initial_interval=1.0))
    workflow.add_node("tools", call_tool, retry_policy=RetryPolicy(max_attempts=max_attempts, initial_interval=1.0))

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

    return workflow.compile(name="agent", checkpointer=checkpointer)


postgres_checkpointer = AsyncPostgresSaver(async_pool)

# Create the graph instance.
GRAPH = create_agent_graph(max_attempts=S.MAX_ATTEMPTS, checkpointer=postgres_checkpointer)

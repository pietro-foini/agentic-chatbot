from typing import Awaitable, Callable

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelRequest,
    ModelResponse,
    ModelRetryMiddleware,
    ToolRetryMiddleware,
    wrap_model_call,
)
from langchain.tools import ToolRuntime
from langchain_core.tools import tool
from langgraph.checkpoint.postgres import BasePostgresSaver
from langgraph.store.postgres.base import BasePostgresStore

from chatbot.graphs.prompts import SUPERVISOR_PROMPT
from chatbot.graphs.state import Context, State
from chatbot.graphs.v3.subagents import agent_1, agent_2, agent_3, agent_4
from chatbot.llm.config import MODELS
from chatbot.settings import S


@tool
async def delegate_to_database_specialist(request: str, runtime: ToolRuntime[Context, State]) -> str:
    """Delegate database inspection, schema discovery, and SQL query execution tasks to the database specialist"""

    result = await agent_1.ainvoke(
        {"messages": [{"role": "user", "content": request}]},
        context=runtime.context,
    )
    return result["messages"][-1].content


@tool
async def delegate_to_research_specialist(request: str, runtime: ToolRuntime[Context, State]) -> str:
    """Delegate factual information retrieval, documentation lookup, and RAG knowledge search"""

    result = await agent_2.ainvoke(
        {"messages": [{"role": "user", "content": request}]},
        context=runtime.context,
    )
    return result["messages"][-1].content


@tool
async def delegate_to_document_specialist(request: str, runtime: ToolRuntime[Context, State]) -> str:
    """Delegate cloud storage file operations, specifically downloading and analyzing PDF documents from Google Cloud Storage"""

    result = await agent_3.ainvoke(
        {"messages": [{"role": "user", "content": request}]},
        context=runtime.context,
    )
    return result["messages"][-1].content


@tool
async def delegate_to_designer_specialist(request: str, runtime: ToolRuntime[Context, State]) -> str:
    """Delegate chart generation and data visualization (Seaborn/Matplotlib) based on an existing stored table"""

    result = await agent_4.ainvoke(
        {"messages": [{"role": "user", "content": request}]},
        context=runtime.context,
    )
    return result["messages"][-1].content


SUPERVISOR_TOOLS = [
    delegate_to_database_specialist,
    delegate_to_research_specialist,
    delegate_to_document_specialist,
    delegate_to_designer_specialist,
]


@wrap_model_call
async def dynamic_model(
    request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
) -> ModelResponse:
    """Dynamic model"""

    return await handler(request.override(model=request.runtime.context.model))


# Create the graph instance.
def build_agent(checkpointer: BasePostgresSaver, store: BasePostgresStore):
    """Construct an agent instance"""

    # noinspection PyTypeChecker
    return create_agent(
        model=MODELS[S.MODEL_NAME],
        tools=SUPERVISOR_TOOLS,
        state_schema=State,
        context_schema=Context,
        checkpointer=checkpointer,
        store=store,
        system_prompt=SUPERVISOR_PROMPT.format(
            user_name="Pietro",
            bot_name=S.CHATBOT_NAME,
        ),
        middleware=[
            dynamic_model,
            ModelRetryMiddleware(max_retries=S.MAX_ATTEMPTS),
            ToolRetryMiddleware(max_retries=S.MAX_ATTEMPTS),
        ],
        name="supervisor_agent",
    )

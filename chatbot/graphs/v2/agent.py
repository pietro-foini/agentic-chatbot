from typing import Awaitable, Callable

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    InterruptOnConfig,
    ModelRequest,
    ModelResponse,
    ModelRetryMiddleware,
    ToolRetryMiddleware,
    dynamic_prompt,
    wrap_model_call,
)
from langgraph.store.memory import InMemoryStore

from chatbot.graphs.config import ALL_TOOLS, SENSITIVE_TOOLS, postgres_checkpointer
from chatbot.graphs.state import Context, State
from chatbot.llm.config import get_model
from chatbot.settings import S

store = InMemoryStore()


@dynamic_prompt
async def inject_dynamic_prompt(request: ModelRequest) -> str:
    """Dynamic prompt"""

    return request.runtime.context.system_prompt


@wrap_model_call
async def inject_dynamic_model(
    request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
) -> ModelResponse:
    """Dynamic model"""

    return await handler(request.override(model=request.runtime.context.model))


# Create the graph instance.
# noinspection PyTypeChecker
GRAPH = create_agent(
    model=get_model(S.MODEL_NAME),
    tools=ALL_TOOLS,
    state_schema=State,
    context_schema=Context,
    checkpointer=postgres_checkpointer,
    store=store,
    middleware=[
        inject_dynamic_prompt,
        inject_dynamic_model,
        ModelRetryMiddleware(max_retries=S.MAX_ATTEMPTS),
        ToolRetryMiddleware(max_retries=S.MAX_ATTEMPTS),
    ]
    + [
        HumanInTheLoopMiddleware(
            interrupt_on={tool: InterruptOnConfig(allowed_decisions=["approve", "reject"]) for tool in SENSITIVE_TOOLS}
        ),
    ],
    name=S.CHATBOT_NAME,
)

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    InterruptOnConfig,
    ModelRequest,
    ModelResponse,
    ModelRetryMiddleware,
    ToolRetryMiddleware,
    wrap_model_call,
)
from langchain.messages import SystemMessage
from langgraph.store.memory import InMemoryStore

from chatbot.graphs.config import ALL_TOOLS, SENSITIVE_TOOLS, postgres_checkpointer
from chatbot.graphs.state import Context, State
from chatbot.llm.google import llm
from chatbot.settings import S

store = InMemoryStore()


@wrap_model_call
async def inject_dynamic_prompt(request: ModelRequest, handler) -> ModelResponse:
    """Dynamic prompt"""

    # Overwrite or add system message.
    new_system_message = SystemMessage(content=[{"type": "text", "text": request.runtime.context.system_prompt}])

    return await handler(request.override(system_message=new_system_message))


# Create the graph instance.
# noinspection PyTypeChecker
GRAPH = create_agent(
    model=llm,
    tools=ALL_TOOLS,
    state_schema=State,
    context_schema=Context,
    checkpointer=postgres_checkpointer,
    store=store,
    middleware=[
        inject_dynamic_prompt,
        ModelRetryMiddleware(max_retries=S.MAX_ATTEMPTS),
        ToolRetryMiddleware(max_retries=S.MAX_ATTEMPTS),
    ]
    + [
        HumanInTheLoopMiddleware(interrupt_on={tool: InterruptOnConfig(allowed_decisions=["approve", "reject"])})
        for tool in SENSITIVE_TOOLS
    ],
    name=S.CHATBOT_NAME,
)

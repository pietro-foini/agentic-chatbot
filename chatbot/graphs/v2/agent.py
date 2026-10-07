import time
from typing import Awaitable, Callable

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ExtendedModelResponse,
    HumanInTheLoopMiddleware,
    InterruptOnConfig,
    ModelRequest,
    ModelResponse,
    ModelRetryMiddleware,
    ToolRetryMiddleware,
    dynamic_prompt,
    wrap_model_call,
    wrap_tool_call,
)
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from chatbot.db.config import checkpointer, store
from chatbot.graphs.config import ALL_TOOLS, SENSITIVE_TOOLS
from chatbot.graphs.state import Context, State
from chatbot.llm.config import get_model
from chatbot.settings import S


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


@wrap_model_call
async def track_model_metrics(
    request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]
) -> ExtendedModelResponse:
    """Track model metrics"""

    start_time = time.time()
    response = await handler(request)
    end_time = time.time()

    elapsed = round(end_time - start_time, 2)

    total_input_tokens, total_output_tokens = 0, 0

    for msg in response.result:
        msg.response_metadata = {
            **(getattr(msg, "response_metadata", {}) or {}),
            "elapsed_time": elapsed,
        }
        if hasattr(msg, "usage_metadata") and msg.usage_metadata:
            total_input_tokens += msg.usage_metadata.get("input_tokens", 0)
            total_output_tokens += msg.usage_metadata.get("output_tokens", 0)

    return ExtendedModelResponse(
        model_response=response,
        command=Command(
            update={
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
            }
        ),
    )


@wrap_tool_call
async def track_tool_metrics(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
) -> ToolMessage | Command:
    """Track tool metrics"""

    start_time = time.time()
    result = await handler(request)
    end_time = time.time()

    elapsed = round(end_time - start_time, 2)

    if isinstance(result, ToolMessage):
        result.response_metadata = {
            **(result.response_metadata or {}),
            "elapsed_time": elapsed,
        }
    elif isinstance(result, Command) and isinstance(result.update, dict):
        messages = result.update.get("messages", [])
        for msg in messages:
            if isinstance(msg, ToolMessage):
                msg.response_metadata = {
                    **(msg.response_metadata or {}),
                    "elapsed_time": elapsed,
                }

    return result


# Create the graph instance.
# noinspection PyTypeChecker
LangGraphAgent = create_agent(
    model=get_model(S.MODEL_NAME),
    tools=ALL_TOOLS,
    state_schema=State,
    context_schema=Context,
    checkpointer=checkpointer,
    store=store,
    middleware=[
        inject_dynamic_prompt,
        inject_dynamic_model,
        track_model_metrics,
        track_tool_metrics,
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

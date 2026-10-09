import base64
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
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.postgres.base import BasePostgresSaver
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.store.postgres.base import BasePostgresStore
from langgraph.types import Command

from chatbot.graphs.config import ALL_TOOLS, SENSITIVE_TOOLS, download_and_analyze_pdf_from_storage
from chatbot.graphs.state import Context, State
from chatbot.graphs.utils import extract_model_name
from chatbot.llm.config import get_model
from chatbot.llm.pricing import compute_cost
from chatbot.settings import CloudProvider, S


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

    total_cost = 0.0
    for msg in response.result:
        in_tokens, out_tokens = 0, 0

        # Get used tokens.
        if hasattr(msg, "usage_metadata") and msg.usage_metadata:
            in_tokens = msg.usage_metadata.get("input_tokens", 0)
            out_tokens = msg.usage_metadata.get("output_tokens", 0)

        # Compute cost.
        model_name = extract_model_name(request, msg)
        msg_cost = round(compute_cost(model_name, in_tokens, out_tokens), 6)

        # Save retrieved information into the message.
        msg.response_metadata = {
            **(getattr(msg, "response_metadata", {}) or {}),
            "elapsed_time": elapsed,
            "cost": msg_cost,
        }

        # Accumulate costs for state.
        total_cost += msg_cost

    return ExtendedModelResponse(
        model_response=response,
        command=Command(
            update={
                "total_cost": total_cost,
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


@wrap_tool_call
async def inject_aws_pdf_middleware(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
) -> ToolMessage | Command:
    """Inject PDF S3 URI as a HumanMessage for AWS"""

    result = await handler(request)

    if S.CLOUD_PROVIDER != CloudProvider.AWS or request.tool.name != download_and_analyze_pdf_from_storage.name:
        return result

    if isinstance(result, ToolMessage) and result.artifact:
        file_id = result.artifact.get("file_id", "")
        file_uri = result.artifact.get("file_uri")

        if file_uri:
            safe_id = file_id.replace("-", "")[:8]

            pdf_msg = HumanMessage(
                content=[
                    {
                        "type": "document",
                        "document": {
                            "format": "pdf",
                            "name": f"attachment_{safe_id}",
                            "source": {"s3Location": {"uri": file_uri}},
                        },
                    },
                    {"text": "Here is the requested PDF document. Please analyze it."},
                ]
            )

            return Command(update={"messages": [result, pdf_msg]})

    return result


# Create the graph instance.
def build_agent(checkpointer: BasePostgresSaver, store: BasePostgresStore):
    """Construct an agent instance"""

    # noinspection PyTypeChecker
    return create_agent(
        model=get_model(),
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
            inject_aws_pdf_middleware,
            ModelRetryMiddleware(max_retries=S.MAX_ATTEMPTS),
            ToolRetryMiddleware(max_retries=S.MAX_ATTEMPTS),
        ]
        + [
            HumanInTheLoopMiddleware(
                interrupt_on={
                    tool: InterruptOnConfig(allowed_decisions=["approve", "reject"]) for tool in SENSITIVE_TOOLS
                }
            ),
        ],
        name=S.CHATBOT_NAME,
    )

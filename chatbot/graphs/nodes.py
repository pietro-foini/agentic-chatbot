import time
from datetime import datetime
from zoneinfo import ZoneInfo

from langchain.tools import ToolRuntime
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt

from chatbot.graphs.security import public_tool_call
from chatbot.graphs.state import State

SENSITIVE_TOOLS = {"download_and_analyze_pdf_from_gcs"}


async def _run_tool_calls(
    tool_calls: list[dict], state: State, config: RunnableConfig, runtime: Runtime
) -> list[ToolMessage]:
    """Execute tool calls, skipping unknown names instead of crashing the graph."""

    tools_by_name = {tool.name: tool for tool in runtime.context.tools}
    outputs = []

    for tool_call in tool_calls:
        tool = tools_by_name.get(tool_call["name"])
        if tool is None:
            outputs.append(
                ToolMessage(
                    content=f"Unknown tool '{tool_call['name']}' was not executed.",
                    name=tool_call["name"],
                    status="error",
                    tool_call_id=tool_call["id"],
                )
            )
            continue

        tool_runtime = ToolRuntime(
            state=state,
            context=runtime.context,
            store=runtime.store,
            stream_writer=runtime.stream_writer,
            config=config,
            tool_call_id=tool_call["id"],
        )

        start_time = time.time()
        tool_response = await tool.ainvoke({**tool_call["args"], "runtime": tool_runtime})
        end_time = time.time()

        outputs.append(
            ToolMessage(
                content=tool_response["content"],
                name=tool_call["name"],
                status=tool_response["status"],
                tool_call_id=tool_call["id"],
                artifact=tool_response.get("artifact"),
                response_metadata={
                    "elapsed_time": round(end_time - start_time, 2),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
        )

    return outputs


async def call_tool(state: State, config: RunnableConfig, runtime: Runtime):
    """Handles tool calls with a Human-in-the-Loop Guardrail check."""

    tool_calls = list(state["messages"][-1].tool_calls or [])
    sensitive_calls = [tc for tc in tool_calls if tc["name"] in SENSITIVE_TOOLS]
    safe_calls = [tc for tc in tool_calls if tc["name"] not in SENSITIVE_TOOLS]

    if sensitive_calls:
        tool_names = ", ".join(tc["name"] for tc in sensitive_calls)
        review_request = {
            "action": "human_approval_required",
            "sensitive_tool_calls": [public_tool_call(tc) for tc in sensitive_calls],
            "safe_tool_names": [tc["name"] for tc in safe_calls],
            "message": f"The assistant requires authorization to run: {tool_names}.",
        }
        human_decision = interrupt(review_request)

        if not human_decision.get("approved", False):
            rejection_reason = human_decision.get("reason", "Action rejected by user.")
            outputs = [
                ToolMessage(
                    content=f"Action canceled by user: {rejection_reason}",
                    name=tc["name"],
                    status="error",
                    tool_call_id=tc["id"],
                )
                for tc in sensitive_calls
            ]
            if safe_calls:
                outputs.extend(await _run_tool_calls(safe_calls, state, config, runtime))
            return Command(update={"messages": outputs})

    outputs = await _run_tool_calls(tool_calls, state, config, runtime)
    return Command(update={"messages": outputs})


async def call_model(state: State, config: RunnableConfig, runtime: Runtime):
    """Invoke the LLM with the current state"""

    model = runtime.context.model
    if tools := runtime.context.tools:
        model = model.bind_tools(tools)

    start_time = time.time()
    model_response = await model.ainvoke(
        [SystemMessage(content=runtime.context.system_prompt)] + state["messages"], config
    )
    end_time = time.time()

    model_response.response_metadata["elapsed_time"] = round(end_time - start_time, 2)
    model_response.response_metadata["timestamp"] = datetime.now(ZoneInfo("Europe/Berlin")).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    model_response.name = runtime.context.chatbot_name

    return Command(update={"messages": [model_response]})


async def should_continue(state: State):
    """Determine whether to continue with tools or end"""

    messages = state["messages"]
    last_message = messages[-1]

    # https://github.com/langchain-ai/langgraph/issues/6574.
    finish_reason = (getattr(last_message, "response_metadata", {}) or {}).get("finish_reason")
    if finish_reason == "MALFORMED_FUNCTION_CALL":
        raise ValueError("LLM returned malformed function call")

    if not last_message.tool_calls:
        return "end"

    return "continue"

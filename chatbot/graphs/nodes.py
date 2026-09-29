import time
from datetime import datetime
from zoneinfo import ZoneInfo

from langchain.tools import ToolRuntime
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from langgraph.types import Command

from chatbot.graphs.state import State


async def call_tool(state: State, config: RunnableConfig, runtime: Runtime):
    """Handles agent reasoning and potentially emits tool calls"""

    tools_by_name = {tool.name: tool for tool in runtime.context.tools}

    outputs = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        tool_args = tool_call["args"]

        tool_runtime = ToolRuntime(
            state=state,
            context=runtime.context,
            store=runtime.store,
            stream_writer=runtime.stream_writer,
            config=config,
            tool_call_id=tool_call["id"],
        )

        start_time = time.time()
        tool_response = await tool.ainvoke({**tool_args, "runtime": tool_runtime})
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

    return Command(update={"messages": outputs})


async def call_model(state: State, config: RunnableConfig, runtime: Runtime):
    """Invoke the LLM with the current state"""

    model = runtime.context.model
    if tools := runtime.context.tools:
        model = model.bind_tools(tools)

    start_time = time.time()
    model_response = await model.ainvoke([SystemMessage(content=runtime.context.system_prompt)] + state["messages"], config)
    end_time = time.time()

    model_response.response_metadata["elapsed_time"] = round(end_time - start_time, 2)
    model_response.response_metadata["timestamp"] = datetime.now(ZoneInfo("Europe/Berlin")).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    model_response.name = runtime.context.chatbot_name

    return Command(update={"messages": [model_response]})


def should_continue(state: State):
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

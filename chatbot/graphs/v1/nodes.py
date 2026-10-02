import time
from datetime import datetime
from zoneinfo import ZoneInfo

from langchain.tools import ToolRuntime
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt

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

        call_payload = {
            "name": tool_call["name"],
            "args": {**tool_call["args"], "runtime": tool_runtime},
            "id": tool_call["id"],
            "type": "tool_call",
        }

        tool_message = await tool.ainvoke(call_payload, config=config)
        end_time = time.time()

        if not isinstance(tool_message, ToolMessage):
            tool_message = ToolMessage(
                content=tool_message,
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )

        tool_message.response_metadata = {
            **(tool_message.response_metadata or {}),
            "elapsed_time": round(end_time - start_time, 2),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        outputs.append(tool_message)

    return outputs


async def call_tool(state: State, config: RunnableConfig, runtime: Runtime):
    """Handles tool calls with a Human-in-the-Loop Guardrail check."""

    tool_calls = list(state["messages"][-1].tool_calls or [])
    sensitive_calls = [tc for tc in tool_calls if tc["name"] in SENSITIVE_TOOLS]
    safe_calls = [tc for tc in tool_calls if tc["name"] not in SENSITIVE_TOOLS]

    if sensitive_calls:
        action_requests = [
            {
                "name": tc["name"],
                "args": tc["args"],
                "description": f"Tool execution requires approval\n\nTool: {tc['name']}\nArgs: {tc['args']}",
            }
            for tc in sensitive_calls
        ]

        review_configs = [
            {
                "action_name": tc["name"],
                "allowed_decisions": ["approve", "reject"],
            }
            for tc in sensitive_calls
        ]

        review_request = {
            "action_requests": action_requests,
            "review_configs": review_configs,
        }

        # Interrupt execution.
        human_decision = interrupt(review_request)

        if isinstance(human_decision, dict) and "decisions" in human_decision:
            decisions = human_decision["decisions"]
        elif isinstance(human_decision, list):
            decisions = human_decision
        else:
            decisions = [human_decision]

        decision = decisions[0] if decisions else {}
        decision_type = decision.get("type") or decision.get("action")

        if decision_type != "approve":
            rejection_reason = decision.get("message") or "Action rejected by user."
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

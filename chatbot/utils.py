import uuid

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from more_itertools import split_before

from chatbot.data import AssistantResponse, Attachment, ThinkingStep, UserResponse


def parse_messages(messages: list[HumanMessage | ToolMessage | AIMessage]) -> list[UserResponse | AssistantResponse]:
    """
    Parse the conversation sequence.

    :param messages: the input conversation messages
    :type messages: list[HumanMessage | ToolMessage | AIMessage]
    :return: the parsed conversation messages
    :rtype: list[UserResponse | AssistantResponse]
    """

    parsed_messages = []

    for group in split_before(messages, lambda m: isinstance(m, HumanMessage)):
        user_response = extract_user_response(group)
        bot_response = extract_assistant_response(group)
        parsed_messages.extend((user_response, bot_response))

    return parsed_messages


def extract_user_response(messages: list[HumanMessage | ToolMessage | AIMessage]) -> UserResponse:
    """
    Extract the user response.

    :param messages: the input conversation messages
    :type messages: list[HumanMessage | ToolMessage | AIMessage]
    :return: the user response
    :rtype: UserResponse
    """

    response = UserResponse(id=str(uuid.uuid4()), name="User", content="")

    for msg in messages:
        if isinstance(msg, HumanMessage):
            if msg.id:
                response.id = msg.id
            if msg.name:
                response.name = msg.name
            if msg.content:
                if isinstance(msg.content, str):
                    response.content = msg.content
                elif isinstance(msg.content, list):
                    text_parts = []
                    attachment_data = None

                    # Retrieve optional metadata file.
                    file_metadata = getattr(msg, "additional_kwargs", {}).get("file_metadata", {})

                    for item in msg.content:
                        if isinstance(item, str):
                            text_parts.append(item)
                        elif isinstance(item, dict):
                            item_type = item.get("type")
                            if item_type == "text" and isinstance(item.get("text"), str):
                                text_parts.append(item["text"])
                            elif item_type in ("media", "image", "file"):
                                attachment_data = Attachment(
                                    id=file_metadata["id"],
                                    file_uri=item["file_uri"],
                                    type=item["mime_type"],
                                    size=file_metadata["size"],
                                )

                    response.content = "\n".join(text_parts)
                    response.attachment = attachment_data

    return response


def extract_assistant_response(messages: list[HumanMessage | ToolMessage | AIMessage]) -> AssistantResponse:
    """
    Extract the assistant response.

    :param messages: the input conversation messages
    :type messages: list[HumanMessage | ToolMessage | AIMessage]
    :return: the assistant response
    :rtype: AssistantResponse
    """

    response = AssistantResponse(id=str(uuid.uuid4()), name="AI Assistant", content="")

    is_tool_used = any(isinstance(msg, ToolMessage) for msg in messages)

    for msg in messages:
        if isinstance(msg, ToolMessage):
            elapsed = msg.response_metadata.get("elapsed_time", 0)
            response.thinking.append(ThinkingStep(step_name=f"Tool Response ({elapsed}s)", details=str(msg.content)))
            if msg.artifact and msg.artifact.get("type") == "image":
                response.image = msg.artifact["base64"]
            if msg.artifact and msg.artifact.get("type") == "table":
                response.table = msg.artifact["base64"]
            if msg.artifact and msg.artifact.get("type") == "pdf":
                response.pdf = msg.artifact["base64"]

        elif isinstance(msg, AIMessage):
            if msg.id:
                response.id = msg.id
            if msg.name:
                response.name = msg.name
            if msg.tool_calls:
                args = msg.tool_calls[0].get("args", {})
                tool = msg.tool_calls[0].get("name", "Unknown")
                elapsed = msg.response_metadata.get("elapsed_time", 0)
                response.thinking.append(ThinkingStep(step_name=f"Tool Call: {tool} ({elapsed}s)", details=args))
            elif msg.content:
                if isinstance(msg.content, str):
                    response.content = msg.content
                elif isinstance(msg.content, list) and isinstance(msg.content[0], str):
                    response.content = msg.content[0]
                elif isinstance(msg.content, list) and isinstance(msg.content[0].get("text"), str):
                    response.content = msg.content[0].get("text", "")
                elif (
                    isinstance(msg.content, list)
                    and isinstance(msg.content[0].get("thinking"), str)
                    and isinstance(msg.content[1].get("text"), str)
                ):
                    response.content = msg.content[1].get("text", "")
                    response.thinking.append(
                        ThinkingStep(step_name="AI Thinking", details=msg.content[0].get("thinking", ""))
                    )
                else:
                    response.content = ""

                if is_tool_used:
                    elapsed = msg.response_metadata.get("elapsed_time", 0)
                    response.thinking.append(
                        ThinkingStep(step_name=f"AI Response ({elapsed}s)", details=f"{str(response.content)[:10]}...")
                    )

    return response

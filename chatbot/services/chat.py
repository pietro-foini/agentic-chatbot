import uuid

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.store.postgres import AsyncPostgresStore
from langgraph.types import Command
from more_itertools import split_before

from chatbot.db.models import UserModel
from chatbot.schemas.chat import AssistantResponse, Attachment, ThinkingStep, UserResponse
from chatbot.settings import CloudProvider, S


def get_last_turn_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Messages starting from the latest HumanMessage"""

    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return messages[i:]

    return []


def build_human_message(user: UserModel, text: str, attachment: Attachment | None) -> HumanMessage:
    """Build a HumanMessage from the provided data"""

    content: list[dict] = [{"type": "text", "text": text}]
    additional_kwargs: dict = {}

    if attachment:
        additional_kwargs["file_metadata"] = {"id": attachment.id, "size": attachment.size}
        if S.CLOUD_PROVIDER == CloudProvider.GCP:
            content.append({"type": "file", "mime_type": attachment.type, "file_id": attachment.file_uri})
        if S.CLOUD_PROVIDER == CloudProvider.AWS:
            safe_id = attachment.id.replace("-", "")[:8]
            content.append(
                {
                    "type": "document",
                    "document": {
                        "format": "pdf",
                        "name": f"attachment_{safe_id}",
                        "source": {"s3Location": {"uri": attachment.file_uri}},
                    },
                }
            )

    return HumanMessage(
        id=str(uuid.uuid4()),
        name=user.username or "user",
        content=content,
        additional_kwargs=additional_kwargs,
    )


def build_resume_command(approved: str, reason: str | None) -> Command:
    """Build resume command"""

    if approved.lower() == "true":
        decisions = [{"type": "approve"}]
    else:
        decisions = [{"type": "reject", "message": reason or "Action rejected by user."}]

    return Command(resume={"decisions": decisions})


async def parse_messages(
    thread_id: str, messages: list[BaseMessage], store: AsyncPostgresStore
) -> list[UserResponse | AssistantResponse]:
    """
    Parse the conversation sequence.

    :param messages: the input conversation messages
    :param thread_id: the ID of the thread
    :param store: the store
    :return: the parsed conversation messages
    """

    parsed_messages = []

    for group in split_before(messages, lambda m: isinstance(m, HumanMessage)):
        user_response = await extract_user_response(thread_id, group)
        bot_response = await extract_assistant_response(thread_id, group, store)
        parsed_messages.extend((user_response, bot_response))

    return parsed_messages


async def extract_user_response(thread_id: str, messages: list[BaseMessage]) -> UserResponse:
    """
    Extract the user response.

    :param messages: the input conversation messages
    :param thread_id: the ID of the thread
    :return: the user response
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
                            elif item_type == "file":  # GCP
                                attachment_data = Attachment(
                                    id=file_metadata["id"],
                                    file_uri=item["file_id"],
                                    type=item["mime_type"],
                                    size=file_metadata.get("size"),
                                )
                            elif item_type == "document":  # AWS
                                attachment_data = Attachment(
                                    id=file_metadata["id"],
                                    file_uri=item["document"]["source"]["s3Location"]["uri"],
                                    type=item["document"]["format"],
                                    size=file_metadata.get("size"),
                                )

                    response.content = "\n".join(text_parts)
                    response.attachment = attachment_data

    return response


async def extract_assistant_response(
    thread_id: str, messages: list[BaseMessage], store: AsyncPostgresStore
) -> AssistantResponse:
    """
    Extract the assistant response.

    :param messages: the input conversation messages
    :param thread_id: the ID of the thread
    :param store: the store
    :return: the assistant response
    """

    response = AssistantResponse(id=str(uuid.uuid4()), name="AI Assistant", content="")

    is_tool_used = any(isinstance(msg, ToolMessage) for msg in messages)

    for msg in messages:
        if isinstance(msg, ToolMessage):
            elapsed = msg.response_metadata.get("elapsed_time", 0)
            response.thinking.append(ThinkingStep(step_name=f"Tool Response ({elapsed}s)", details=str(msg.content)))
            if msg.artifact and msg.artifact.get("mime_type") == "image/png":
                item = await store.aget(namespace=("artifacts", thread_id), key=msg.artifact["file_id"])
                response.image = item.value["data"]
            if msg.artifact and msg.artifact.get("mime_type") == "text/csv":
                item = await store.aget(namespace=("artifacts", thread_id), key=msg.artifact["table_name"])
                response.table = item.value["data"]
            if msg.artifact and msg.artifact.get("mime_type") == "application/pdf":
                item = await store.aget(namespace=("artifacts", thread_id), key=msg.artifact["file_id"])
                response.pdf = item.value["data"]

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
                elif isinstance(msg.content, list):
                    for item in msg.content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            response.content = item.get("text", "")
                            break
                else:
                    response.content = ""

                if is_tool_used:
                    elapsed = msg.response_metadata.get("elapsed_time", 0)
                    response.thinking.append(
                        ThinkingStep(step_name=f"AI Response ({elapsed}s)", details=f"{str(response.content)[:10]}...")
                    )

    return response

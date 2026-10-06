import asyncio
import re
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from google.genai.errors import ServerError
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langchain_postgres import PGVectorStore
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, StateSnapshot
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from chatbot.api.dependencies import (
    ALLOWED_MIME_TYPES,
    get_current_user,
    get_graph,
    get_owned_thread_state,
    get_session,
    get_vector_store,
)
from chatbot.db.models import UserModel
from chatbot.graphs.config import ALL_TOOLS
from chatbot.graphs.prompts import SYSTEM_PROMPT
from chatbot.graphs.state import Context
from chatbot.llm.config import CATEGORIZED_MODELS, get_model
from chatbot.llm.pricing import compute_cost
from chatbot.schemas import Attachment, UserResponse
from chatbot.settings import S
from chatbot.storage.config import storage_client
from chatbot.utils import extract_assistant_response, parse_messages

chat_router = APIRouter(tags=["Chat"])

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent.parent / "templates"))


@chat_router.get(path="/", dependencies=[Depends(get_current_user)])
async def new_chat():
    """Create a new chat"""

    thread_id = uuid.uuid4()

    return RedirectResponse(url=f"/chat/{thread_id}", status_code=303)


@chat_router.delete(path="/chat", dependencies=[Depends(get_owned_thread_state)])
async def delete_chats(
    current_user: Annotated[UserModel, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
):
    """Delete all the chat threads belonging to the current user"""

    await session.execute(
        text("""
            DELETE FROM store 
            WHERE prefix IN (
                SELECT 'artifacts.' || thread_id 
                FROM checkpoints 
                WHERE metadata ->> 'user_id' = :user_id
            );
        """),
        {"user_id": str(current_user.id)},
    )

    await session.execute(
        text("""
            DELETE FROM checkpoint_writes 
            WHERE thread_id IN (
                SELECT DISTINCT thread_id FROM checkpoints WHERE metadata ->> 'user_id' = :user_id
            );
        """),
        {"user_id": str(current_user.id)},
    )
    await session.execute(
        text("""
            DELETE FROM checkpoint_blobs 
            WHERE thread_id IN (
                SELECT DISTINCT thread_id FROM checkpoints WHERE metadata ->> 'user_id' = :user_id
            );
        """),
        {"user_id": str(current_user.id)},
    )
    await session.execute(
        text("""
            DELETE FROM checkpoints 
            WHERE metadata ->> 'user_id' = :user_id;
        """),
        {"user_id": str(current_user.id)},
    )

    response = HTMLResponse()
    response.headers["HX-Redirect"] = "/"

    return response


@chat_router.get(path="/chat/{thread_id}", response_class=HTMLResponse)
async def get_chat(
    request: Request,
    thread_id: uuid.UUID,
    state: Annotated[StateSnapshot, Depends(get_owned_thread_state)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
):
    """Retrieve the chat history for the given ID thread and the current user"""

    values = (state.values if state else None) or {}

    # Retrieve chat data.
    messages = values.get("messages", [])
    in_tokens = values.get("total_input_tokens", 0)
    out_tokens = values.get("total_output_tokens", 0)

    cost_amount = compute_cost(S.MODEL_NAME, in_tokens, out_tokens)  # TODO: adjust

    # Parse messages for UI rendering purposes.
    parsed_messages = await parse_messages(str(thread_id), messages)

    # Retrieve the list of all the threads of the current user in order to visualize them into the sidebar chronology.
    query = text("SELECT DISTINCT thread_id FROM checkpoints WHERE metadata ->> 'user_id' = :user_id;")
    result = await session.execute(query, {"user_id": str(current_user.id)})

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "messages": parsed_messages,
            "thread_id": str(thread_id),
            "threads": [str(row[0]) for row in result.all()],
            "model_categories": {category: list(models.keys()) for category, models in CATEGORIZED_MODELS.items()},
            "default_model": S.MODEL_NAME,
            "chat_cost": f"${cost_amount:.5f}",
        },
    )


@chat_router.post(path="/chat/{thread_id}", dependencies=[Depends(get_owned_thread_state)], response_class=HTMLResponse)
async def add_user_message(
    request: Request,
    thread_id: uuid.UUID,
    current_user: Annotated[UserModel, Depends(get_current_user)],
    message: str = Form(...),
    file: UploadFile | None = File(None),
):
    """Post a user message to the chat history for the given ID thread"""

    message = message.strip()

    if not message:
        return RedirectResponse(url=f"/chat/{thread_id}", status_code=303)

    # Handle the allowed optional attachment file.
    attachment = None
    if file and file.filename and file.content_type and file.size:
        size_kb = round(file.size / 1024, 2)
        if file.content_type not in ALLOWED_MIME_TYPES:
            return templates.TemplateResponse(
                request=request,
                name="partials/error_message.html",
                context={"error": "File type not allowed."},
            )
        if size_kb > S.MAX_ATTACHMENTS_SIZE:
            return templates.TemplateResponse(
                request=request,
                name="partials/error_message.html",
                context={"error": f"The maximum allowed file size is {S.MAX_ATTACHMENTS_SIZE}KB."},
            )

        # Download file.
        content_bytes = await file.read()

        # Upload su Google Cloud Storage in order to keep them tracked.
        safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(file.filename).name)
        filename = f"{uuid.uuid4()}_{safe_name}"
        bucket = storage_client.bucket(S.GCS_PROCESSED_BUCKET)
        blob = bucket.blob(f"attachments/{filename}")
        await asyncio.to_thread(blob.upload_from_string, content_bytes, content_type=file.content_type)

        # Create attachment object.
        attachment = Attachment(
            id=filename,
            file_uri=f"gs://{S.GCS_PROCESSED_BUCKET}/attachments/{filename}",
            type=file.content_type,
            size=f"{size_kb}KB",
        )

    # Create user response.
    user_message = UserResponse(
        id=str(uuid.uuid4()),
        name=current_user.username or "user",
        content=message,
        attachment=attachment,
    )

    return templates.TemplateResponse(
        request=request,
        name="partials/message_pair.html",
        context={
            "user_message": user_message,
            "thread_id": str(thread_id),
        },
    )


@chat_router.post(
    path="/chat/{thread_id}/assistant", dependencies=[Depends(get_owned_thread_state)], response_class=HTMLResponse
)
async def add_assistant_message(
    request: Request,
    thread_id: uuid.UUID,
    graph: Annotated[CompiledStateGraph, Depends(get_graph)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
    vector_store: PGVectorStore = Depends(get_vector_store),
    user_message_json: str | None = Form(None),
    approved: str | None = Form(None),
    reason: str | None = Form(None),
    model_name: str | None = Form(None),
):
    """Generate an assistant message for the given ID thread"""

    # Define input for the graph execution.
    if user_message_json:
        user_message = UserResponse.model_validate_json(user_message_json)
        content = [{"type": "text", "text": user_message.content}]
        additional_kwargs = {}

        if attachment := user_message.attachment:
            content.append(
                {
                    "type": "media",
                    "mime_type": attachment.type,
                    "file_uri": attachment.file_uri,
                }
            )
            additional_kwargs["file_metadata"] = {
                "id": attachment.id,
                "size": attachment.size,
            }

        message = HumanMessage(
            id=user_message.id,
            name=user_message.name,
            content=content,
            additional_kwargs=additional_kwargs,
        )
        graph_input = {
            "messages": [message],
            "user_preferences": {"style": "technical", "verbosity": "detailed"},
        }
    elif approved is not None:
        is_approved = approved.lower() == "true"
        decisions = (
            [{"type": "approve"}]
            if is_approved
            else [{"type": "reject", "message": reason or "Action rejected by user."}]
        )
        graph_input = Command(resume={"decisions": decisions})
    else:
        return templates.TemplateResponse(
            request=request,
            name="partials/error_message.html",
            context={"error": "Request not valid: no message or decision provided."},
        )

    # Define the graph configuration.
    config = RunnableConfig(
        configurable={"thread_id": str(thread_id)},
        metadata={
            "user_id": str(current_user.id),
            "username": current_user.username,
        },
        recursion_limit=S.RECURSION_LIMIT,
    )
    context = Context(
        thread_id=str(thread_id),
        chatbot_name=S.CHATBOT_NAME,
        model=get_model(model_name or S.MODEL_NAME),
        system_prompt=SYSTEM_PROMPT.format(
            user_name=current_user.username or "user",
            bot_name=S.CHATBOT_NAME,
        ),
        tools=ALL_TOOLS,
        vector_store=vector_store,
    )

    # Graph execution.
    try:
        response = await graph.ainvoke(
            graph_input,
            stream_mode="values",
            config=config,
            context=context,
        )
    except (
        GraphRecursionError,
        ChatGoogleGenerativeAIError,
        ValueError,
        ServerError,
    ):
        return templates.TemplateResponse(
            request=request,
            name="partials/error_message.html",
            context={"error": f"A system error occurred while processing your request. Please try again in a moment."},
        )

    # Check for interrupt.
    state_snapshot = await graph.aget_state(config)
    if state_snapshot.next:
        pending_interrupt = state_snapshot.tasks[0].interrupts[0].value
        return templates.TemplateResponse(
            request=request,
            name="partials/approval_request.html",
            context={
                "thread_id": str(thread_id),
                "approval_info": pending_interrupt,
            },
        )

    in_tokens = state_snapshot.values.get("total_input_tokens", 0)
    out_tokens = state_snapshot.values.get("total_output_tokens", 0)

    cost_amount = compute_cost(model_name or S.MODEL_NAME, in_tokens, out_tokens)

    # Get messages.
    messages = response["messages"]
    start_idx = next((i for i in range(len(messages) - 1, -1, -1) if isinstance(messages[i], HumanMessage)), -1)
    messages = messages[start_idx:]

    # Create assistant response.
    assistant_message = await extract_assistant_response(str(thread_id), messages)

    return templates.TemplateResponse(
        request=request,
        name="partials/assistant_message.html",
        context={
            "response": assistant_message,
            "total_cost": f"${cost_amount:.5f}",
        },
    )

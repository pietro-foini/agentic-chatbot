import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from google.genai.errors import ServerError
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from chatbot.data import Attachment, UserResponse
from chatbot.db.session import async_pool
from chatbot.graphs.agent import GRAPH
from chatbot.graphs.prompts import SYSTEM_PROMPT
from chatbot.graphs.state import Context
from chatbot.graphs.tools import (
    create_plot,
    describe_table,
    download_and_analyze_pdf_from_gcs,
    execute_sql,
    list_tables,
    retrieve_information,
)
from chatbot.llm.google import llm
from chatbot.settings import S
from chatbot.storage.config import storage_client
from chatbot.utils import extract_bot_response, parse_messages

chat_router = APIRouter()

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent.parent / "templates"))


@chat_router.get(path="/")
async def new_chat():
    """Create a new chat"""

    thread_id = str(uuid.uuid4())

    return RedirectResponse(url=f"/chat/{thread_id}", status_code=303)


@chat_router.delete(path="/chat")
async def delete_chats():
    """Delete all the chat threads"""

    async with async_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("DELETE FROM checkpoint_writes WHERE TRUE;")
            await cur.execute("DELETE FROM checkpoints WHERE TRUE;")

    response = HTMLResponse()
    response.headers["HX-Redirect"] = "/"

    return response


@chat_router.get(path="/chat/{thread_id}", response_class=HTMLResponse)
async def get_chat(request: Request, thread_id: str):
    """Retrieve the chat history for the given thread_id"""

    # Retrieve messages from graph state of the current thread.
    state = await GRAPH.aget_state({"configurable": {"thread_id": thread_id}})
    messages = state.values.get("messages", []) if state and hasattr(state, "values") else []

    # Parse messages for rendering purposes.
    parsed_messages = parse_messages(messages)

    # Retrieve the list of all the threads.
    async with async_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT DISTINCT thread_id FROM checkpoints;")
            rows = await cur.fetchall()
            threads = [str(row[0]) for row in rows]

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "messages": parsed_messages,
            "thread_id": thread_id,
            "threads": threads,
        },
    )


@chat_router.post(path="/chat/{thread_id}", response_class=HTMLResponse)
async def add_user_message(
    request: Request,
    thread_id: str,
    message: str = Form(...),
    file: UploadFile | None = File(None),
):
    """Post a user message to the chat history for the given thread_id"""

    message = message.strip()
    if not message:
        return RedirectResponse(url=f"/chat/{thread_id}", status_code=303)

    # Handle the optional attachment file.
    attachment = None
    if file and file.filename and file.content_type:
        mime_type = file.content_type
        content_bytes = await file.read()
        size_kb = round(len(content_bytes) / 1024, 2)

        if size_kb > S.MAX_ATTACHMENTS_SIZE:
            return templates.TemplateResponse(
                request=request,
                name="partials/message_pair.html",
                context={"error": "The maximum allowed file size is 1MB.", "thread_id": thread_id},
                status_code=400,
            )

        # Upload su Google Cloud Storage.
        filename = f"{uuid.uuid4()}_{file.filename}"
        bucket = storage_client.bucket(S.GCS_PROCESSED_BUCKET)
        blob = bucket.blob(f"attachments/{filename}")
        blob.upload_from_string(content_bytes, content_type=mime_type)

        attachment = Attachment(
            id=filename,
            file_uri=f"gs://{S.GCS_PROCESSED_BUCKET}/attachments/{filename}",
            type=mime_type,
            size=f"{size_kb}KB",
        )

    # Create user response for rendering purposes.
    user_message = UserResponse(
        id=str(uuid.uuid4()),
        name="Pietro",
        content=message,
        attachment=attachment,
    )

    return templates.TemplateResponse(
        request=request,
        name="partials/message_pair.html",
        context={
            "user_message": user_message,
            "thread_id": thread_id,
        },
    )


@chat_router.post(path="/chat/{thread_id}/assistant", response_class=HTMLResponse)
async def add_assistant_message(
    request: Request,
    thread_id: str,
    user_message_json: str = Form(...),
):
    """Generate an assistant message for the given thread_id"""

    user_message = UserResponse.model_validate_json(user_message_json)

    content = [{"type": "text", "text": user_message.content}]
    additional_kwargs = {}

    # Handle optional attachment
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

    # Create human message.
    message = HumanMessage(
        id=user_message.id,
        name=user_message.name,
        content=content,
        additional_kwargs=additional_kwargs,
    )

    inputs = {"messages": [message], "user_preferences": {"style": "technical", "verbosity": "detailed"}}

    thread_config = RunnableConfig(configurable={"thread_id": thread_id}, recursion_limit=S.RECURSION_LIMIT)

    context_obj = Context(
        chatbot_name=S.CHATBOT_NAME,
        model=llm,
        system_prompt=SYSTEM_PROMPT.format(
            user_name="Pietro",
            bot_name=S.CHATBOT_NAME,
        ),
        tools=[
            retrieve_information,
            download_and_analyze_pdf_from_gcs,
            execute_sql,
            list_tables,
            describe_table,
            create_plot,
        ],
        vector_store=request.app.state.vector_store,
    )

    # Initial graph execution.
    try:
        response = await GRAPH.ainvoke(
            inputs,
            stream_mode="values",
            config=thread_config,
            context=context_obj,
        )
    except (GraphRecursionError, ChatGoogleGenerativeAIError, ValueError, ServerError) as e:
        return templates.TemplateResponse(
            request=request,
            name="partials/error_message.html",
        )

    # Check graph state if stopped for interrupt.
    state_snapshot = await GRAPH.aget_state(thread_config)
    if state_snapshot.next:
        pending_interrupt = state_snapshot.tasks[0].interrupts[0].value
        return templates.TemplateResponse(
            request=request,
            name="partials/approval_request.html",
            context={
                "thread_id": thread_id,
                "approval_info": pending_interrupt,
            },
        )

    # If not interrupt, continue the execution.
    messages = response["messages"]
    start_idx = next((i for i in range(len(messages) - 1, -1, -1) if isinstance(messages[i], HumanMessage)), -1)
    messages = messages[start_idx:]

    assistant_message = extract_bot_response(messages)

    return templates.TemplateResponse(
        request=request,
        name="partials/assistant_message.html",
        context={"response": assistant_message},
    )


@chat_router.post(path="/chat/{thread_id}/resume", response_class=HTMLResponse)
async def resume_chat(
    request: Request,
    thread_id: str,
    approved: str = Form(...),
    reason: str | None = Form(None),
):
    """Resume a paused agent thread based on human feedback."""

    is_approved = approved.lower() == "true"

    thread_config = RunnableConfig(
        configurable={"thread_id": thread_id},
        recursion_limit=S.RECURSION_LIMIT,
    )

    context_obj = Context(
        chatbot_name=S.CHATBOT_NAME,
        model=llm,
        system_prompt=SYSTEM_PROMPT.format(
            user_name="Pietro",
            bot_name=S.CHATBOT_NAME,
        ),
        tools=[
            retrieve_information,
            download_and_analyze_pdf_from_gcs,
            execute_sql,
            list_tables,
            describe_table,
            create_plot,
        ],
        vector_store=request.app.state.vector_store,
    )

    resume_payload = {"approved": is_approved}
    if not is_approved and reason:
        resume_payload["reason"] = reason

    # Resume graph execution.
    try:
        response = await GRAPH.ainvoke(
            Command(resume=resume_payload),
            config=thread_config,
            context=context_obj,
        )
    except (GraphRecursionError, ChatGoogleGenerativeAIError, ValueError, ServerError) as e:
        return templates.TemplateResponse(
            request=request,
            name="partials/error_message.html",
        )

    messages = response["messages"]
    start_idx = next((i for i in range(len(messages) - 1, -1, -1) if isinstance(messages[i], HumanMessage)), -1)
    messages = messages[start_idx:]

    assistant_message = extract_bot_response(messages)

    return templates.TemplateResponse(
        request=request,
        name="partials/assistant_message.html",
        context={"response": assistant_message},
    )

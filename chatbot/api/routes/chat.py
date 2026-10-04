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
from chatbot.db.config import async_pool
from chatbot.graphs.config import ALL_TOOLS
from chatbot.graphs.prompts import SYSTEM_PROMPT
from chatbot.graphs.state import Context
from chatbot.graphs.v2.agent import GRAPH
from chatbot.llm.ollama import llm
from chatbot.settings import S
from chatbot.storage.config import storage_client
from chatbot.utils import extract_assistant_response, parse_messages

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

    # Delete all the data referring to the persistent memory stored on database.
    async with async_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "TRUNCATE TABLE checkpoints, checkpoint_writes, checkpoint_blobs, checkpoint_migrations RESTART IDENTITY CASCADE;"
            )

    response = HTMLResponse()
    response.headers["HX-Redirect"] = "/"

    return response


@chat_router.get(path="/chat/{thread_id}", response_class=HTMLResponse)
async def get_chat(request: Request, thread_id: str):
    """Retrieve the chat history for the given ID thread"""

    # Retrieve messages from graph state of the current thread.
    state = await GRAPH.aget_state({"configurable": {"thread_id": thread_id}})
    messages = state.values.get("messages", []) if state and hasattr(state, "values") else []

    # Parse messages for UI rendering purposes.
    parsed_messages = parse_messages(messages)

    # Retrieve the list of all the threads in order to visualize them into the chronology.
    async with async_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT DISTINCT thread_id FROM checkpoints;")
            rows = await cur.fetchall()

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "messages": parsed_messages,
            "thread_id": thread_id,
            "threads": [str(row[0]) for row in rows],
        },
    )


@chat_router.post(path="/chat/{thread_id}", response_class=HTMLResponse)
async def add_user_message(
    request: Request,
    thread_id: str,
    message: str = Form(...),
    file: UploadFile | None = File(None),
):
    """Post a user message to the chat history for the given ID thread"""

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
                name="partials/error_message.html",
                context={"error": "The maximum allowed file size is 1MB."},
            )

        # Upload su Google Cloud Storage in order to keep them tracked.
        filename = f"{uuid.uuid4()}_{file.filename}"
        bucket = storage_client.bucket(S.GCS_PROCESSED_BUCKET)
        blob = bucket.blob(f"attachments/{filename}")
        blob.upload_from_string(content_bytes, content_type=mime_type)

        # Create attachment object.
        attachment = Attachment(
            id=filename,
            file_uri=f"gs://{S.GCS_PROCESSED_BUCKET}/attachments/{filename}",
            type=mime_type,
            size=f"{size_kb}KB",
        )

    # Create user response.
    user_message = UserResponse(
        id=str(uuid.uuid4()),
        name="Pietro",  # TODO: cambialo in modo da personalizzarlo
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
    user_message_json: str | None = Form(None),
    approved: str | None = Form(None),
    reason: str | None = Form(None),
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
        tools=ALL_TOOLS,
        vector_store=request.app.state.vector_store,
    )

    # Graph execution.
    try:
        response = await GRAPH.ainvoke(
            graph_input,
            stream_mode="values",
            config=thread_config,
            context=context_obj,
        )
    except (GraphRecursionError, ChatGoogleGenerativeAIError, ValueError, ServerError) as e:
        return templates.TemplateResponse(
            request=request,
            name="partials/error_message.html",
            context={"error": "A system error occurred while processing your request. Please try again in a moment."},
        )

    # Check for interrupt.
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

    # Get messages.
    messages = response["messages"]
    start_idx = next((i for i in range(len(messages) - 1, -1, -1) if isinstance(messages[i], HumanMessage)), -1)
    messages = messages[start_idx:]

    # Create assistant response.
    assistant_message = extract_assistant_response(messages)

    return templates.TemplateResponse(
        request=request,
        name="partials/assistant_message.html",
        context={"response": assistant_message},
    )

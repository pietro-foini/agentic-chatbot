import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from google.genai.errors import ServerError
from langchain_core.runnables import RunnableConfig
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langchain_postgres import PGVectorStore
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.postgres import AsyncPostgresStore
from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from chatbot.api.dependencies import (
    get_checkpointer,
    get_current_user,
    get_graph,
    get_owned_thread,
    get_session,
    get_store,
    get_vector_store,
)
from chatbot.db.models import ThreadModel, UserModel
from chatbot.errors import UserFacingError
from chatbot.graphs.config import ALL_TOOLS, DEFAULT_USER_PREFERENCES
from chatbot.graphs.prompts import SYSTEM_PROMPT
from chatbot.graphs.state import Context
from chatbot.llm.config import MODELS
from chatbot.log import logger
from chatbot.services.attachment import save_attachment
from chatbot.services.chat import (
    build_human_message,
    build_resume_command,
    extract_assistant_response,
    get_last_turn_messages,
    parse_messages,
)
from chatbot.settings import S

chat_router = APIRouter(tags=["Chat"])

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent.parent / "templates"))


@chat_router.get(path="/", dependencies=[Depends(get_current_user)], include_in_schema=False)
async def new_chat():
    """Create a new chat"""

    thread_id = uuid.uuid4()

    return RedirectResponse(url=f"/chat/{thread_id}", status_code=303, headers={"Cache-Control": "no-store"})


@chat_router.delete(path="/chat", response_class=Response)
async def delete_chats(
    current_user: Annotated[UserModel, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
    checkpointer: Annotated[AsyncPostgresSaver, Depends(get_checkpointer)],
    store: Annotated[AsyncPostgresStore, Depends(get_store)],
):
    """Delete all the chat threads information belonging to the current user"""

    # Retrieve the list of all the threads of the current user.
    query = select(ThreadModel.thread_id).where(ThreadModel.user_id == current_user.id)
    threads = (await session.execute(query)).scalars().all()

    # TODO: optimize for a production scenario with many threads per user or add limit to chat threads.
    # Delete all the thread information from graph checkpointer saver.
    for thread in threads:
        await checkpointer.adelete_thread(str(thread))
        # Delete all the thread information from graph store artifacts.
        items = await store.asearch(("artifacts", str(thread)), limit=1000)
        for item in items:
            await store.adelete(item.namespace, item.key)

    if threads:
        # Delete threads from main dataset.
        await session.execute(delete(ThreadModel).where(ThreadModel.user_id == current_user.id))

    return Response(status_code=204, headers={"HX-Redirect": "/"})


@chat_router.get(path="/chat/{thread_id}", response_class=HTMLResponse)
async def get_chat(
    request: Request,
    thread_id: Annotated[uuid.UUID, Depends(get_owned_thread)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
    graph: Annotated[CompiledStateGraph, Depends(get_graph)],
):
    """Retrieve the chat history for the given ID thread and the current user"""

    # Retrieve the graph state of the current thread.
    state = await graph.aget_state({"configurable": {"thread_id": str(thread_id)}})
    values = (state.values if state else None) or {}

    # Retrieve chat data.
    messages = values.get("messages", [])
    total_cost = values.get("total_cost", 0)
    parsed_messages = await parse_messages(str(thread_id), messages)

    # Retrieve the list of all the threads of the current user.
    query = select(ThreadModel.thread_id).where(ThreadModel.user_id == current_user.id)
    threads = (await session.execute(query)).scalars().all()

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "messages": parsed_messages,
            "thread_id": str(thread_id),
            "threads": [str(thread) for thread in threads],
            "models": list(MODELS),
            "default_model": S.MODEL_NAME,
            "total_cost": total_cost,
            "user": current_user,
        },
    )


@chat_router.post(path="/chat/{thread_id}", response_class=HTMLResponse)
async def process_chat(
    request: Request,
    thread_id: Annotated[uuid.UUID, Depends(get_owned_thread)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
    graph: Annotated[CompiledStateGraph, Depends(get_graph)],
    vector_store: Annotated[PGVectorStore, Depends(get_vector_store)],
    message: str | None = Form(None, max_length=500),
    file: UploadFile | None = File(None),
    approved: str | None = Form(None),
    reason: str | None = Form(None),
    model_name: str | None = Form(None),
):
    """Post a user message to assistant conversation to the chat history for the given ID thread"""

    if approved is not None:  # Human into the loop
        # Prepare input for graph execution.
        graph_input = build_resume_command(approved, reason)
    elif (message and message.strip()) or file:  # New message
        # Prepare attachment.
        attachment = await save_attachment(file)
        # Prepare human message.
        message = message.strip() if message else "Analyze the attachment."
        human_message = build_human_message(current_user, message, attachment)
        # Prepare input for graph execution.
        graph_input = {"messages": [human_message], "user_preferences": DEFAULT_USER_PREFERENCES}
        # Create the thread record into the reference table if not already exist (idempotency).
        stmt = (
            insert(ThreadModel)
            .values(thread_id=thread_id, user_id=str(current_user.id))
            .on_conflict_do_nothing(index_elements=[ThreadModel.thread_id])
        )
        await session.execute(stmt)
    else:
        return Response(status_code=204)

    await session.commit()

    # Define the graph setting.
    config = RunnableConfig(
        configurable={"thread_id": str(thread_id)},
        metadata={"user_id": str(current_user.id), "username": current_user.username},
        recursion_limit=S.RECURSION_LIMIT,
    )
    context = Context(
        thread_id=str(thread_id),
        model=MODELS.get(model_name) or MODELS[S.MODEL_NAME],
        system_prompt=SYSTEM_PROMPT.format(user_name=current_user.username, bot_name=S.CHATBOT_NAME),
        tools=ALL_TOOLS,
        vector_store=vector_store,
    )

    # Graph execution.
    try:
        response = await graph.ainvoke(graph_input, stream_mode="values", config=config, context=context)
    except ServerError:
        logger.exception("LLM provider unavailable (thread=%s)", thread_id)
        raise UserFacingError("The AI service is temporarily unavailable. Please try again in a moment.")
    except (GraphRecursionError, ChatGoogleGenerativeAIError):
        logger.exception("Graph execution failed (thread=%s)", thread_id)
        raise UserFacingError("A system error occurred while processing your request. Please try again.")

    # Check for a new interrupt.
    state_snapshot = await graph.aget_state(config)
    if state_snapshot.next and state_snapshot.tasks and state_snapshot.tasks[0].interrupts:
        pending_interrupt = state_snapshot.tasks[0].interrupts[0].value
        return templates.TemplateResponse(
            request=request,
            name="partials/approval_request.html",
            context={"thread_id": str(thread_id), "approval_info": pending_interrupt},
        )

    # Get assistant message.
    current_messages = get_last_turn_messages(response["messages"])
    assistant_message = await extract_assistant_response(str(thread_id), current_messages)
    # Get total cost.
    total_cost = state_snapshot.values.get("total_cost", 0)

    return templates.TemplateResponse(
        request=request,
        name="partials/assistant_message.html",
        context={"response": assistant_message, "total_cost": total_cost},
    )

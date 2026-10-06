import uuid
from collections.abc import AsyncGenerator
from typing import Annotated

import jwt
from fastapi import Cookie, Depends, HTTPException, Request
from jwt.exceptions import InvalidTokenError
from langchain_postgres import PGVectorStore
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import StateSnapshot
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from chatbot.db.config import db
from chatbot.db.models import UserModel
from chatbot.graphs.v2.agent import GRAPH
from chatbot.settings import S

ALLOWED_MIME_TYPES = {"application/pdf", "image/png", "image/jpeg"}


def get_graph() -> CompiledStateGraph:
    """Return the compiled graph"""

    return GRAPH


async def get_owned_thread_state(
    thread_id: uuid.UUID,
    current_user: Annotated[UserModel, Depends(get_current_user)],
    graph: Annotated[CompiledStateGraph, Depends(get_graph)],
) -> StateSnapshot | None:
    """Get the thread state only if the thread is new or belongs to the current user"""

    state = await graph.aget_state({"configurable": {"thread_id": str(thread_id)}})

    if not state or not state.values:
        return None

    thread_owner = (state.metadata or {}).get("user_id")
    if thread_owner and thread_owner != str(current_user.id):
        raise HTTPException(
            status_code=404,
            detail="Chat not found.",
        )

    return state


def get_vector_store(request: Request) -> PGVectorStore:
    """Get PGVectorStore instance from request object initialized during startup"""

    return request.app.state.vector_store


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async database session per request lifecycle"""

    async with db.get_session() as session:
        yield session


async def get_current_user(
    session: Annotated[AsyncSession, Depends(get_session)],
    access_token_cookie: Annotated[str | None, Cookie(alias="access_token")] = None,
) -> UserModel:
    """Authenticate API requests based on token returning the active user or redirect to login page"""

    token = access_token_cookie

    try:
        payload = jwt.decode(token, S.SECRET_KEY, algorithms=[S.ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=303,
                headers={"Location": "/login"},
            )
    except InvalidTokenError:
        raise HTTPException(
            status_code=303,
            headers={"Location": "/login"},
        )

    # Retrieve the current user if it is not disabled.
    query = select(UserModel).where(UserModel.username == username, UserModel.is_active.is_(True))
    result = await session.execute(query)

    current_user = result.scalar_one_or_none()
    if not current_user:
        raise HTTPException(
            status_code=303,
            headers={"Location": "/login"},
        )

    return current_user

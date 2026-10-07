import uuid
from collections.abc import AsyncGenerator
from typing import Annotated

import jwt
from fastapi import Cookie, Depends, HTTPException, Request
from jwt.exceptions import InvalidTokenError
from langchain_postgres import PGVectorStore
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.postgres import AsyncPostgresStore
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from chatbot.db.models import ThreadModel, UserModel
from chatbot.errors import AuthenticationRequiredError
from chatbot.settings import S


def get_graph(request: Request) -> CompiledStateGraph:
    """Return the compiled graph"""

    return request.app.state.graph


async def get_owned_thread(
    thread_id: uuid.UUID,
    current_user: Annotated[UserModel, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> uuid.UUID:
    """Ensure the thread is either new or belongs to the current user"""

    owner = await session.scalar(select(ThreadModel.user_id).where(ThreadModel.thread_id == thread_id))
    if owner is not None and owner != current_user.id:
        raise HTTPException(status_code=404, detail="Chat not found.")

    return thread_id


def get_vector_store(request: Request) -> PGVectorStore:
    """Get PGVectorStore instance from request object initialized during startup"""

    return request.app.state.vector_store


def get_checkpointer(request: Request) -> AsyncPostgresSaver:
    """Get checkpointer instance from request object initialized during startup"""

    return request.app.state.checkpointer


def get_store(request: Request) -> AsyncPostgresStore:
    """Get store instance from request object initialized during startup"""

    return request.app.state.store


async def get_session(request: Request) -> AsyncGenerator[AsyncSession, None]:
    """Yield an async database session per request lifecycle"""

    async with request.app.state.db.session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_current_user(
    session: Annotated[AsyncSession, Depends(get_session)],
    token_cookie: Annotated[str | None, Cookie(alias="access_token")] = None,
) -> UserModel:
    """Authenticate API requests based on token returning the active user or redirect to login page"""

    token = token_cookie

    if not token:
        raise AuthenticationRequiredError()

    try:
        payload = jwt.decode(token, S.SECRET_KEY, algorithms=[S.ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise AuthenticationRequiredError()
    except InvalidTokenError:
        raise AuthenticationRequiredError()

    # Retrieve the current user if it is not disabled.
    query = select(UserModel).where(UserModel.id == user_id, UserModel.is_active.is_(True))
    result = await session.execute(query)

    current_user = result.scalar_one_or_none()
    if not current_user:
        raise AuthenticationRequiredError()

    return current_user

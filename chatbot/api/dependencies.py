import uuid
from collections.abc import AsyncGenerator
from typing import Annotated

import jwt
from fastapi import Cookie, Depends, HTTPException, Request
from jwt.exceptions import InvalidTokenError
from langchain_core.vectorstores import VectorStore
from langgraph.checkpoint.postgres.base import BasePostgresSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.postgres.base import BasePostgresStore
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from chatbot.db.models import ThreadModel, UserModel
from chatbot.errors import AuthenticationRequiredError, FormLoginException, FormRegisterException
from chatbot.schemas.auth import LoginForm, RegisterForm
from chatbot.settings import S


def get_graph(request: Request) -> CompiledStateGraph:
    """Return the compiled graph from request object initialized during startup"""

    return request.app.state.graph


def get_vector_store(request: Request) -> VectorStore:
    """Get vector store instance from request object initialized during startup"""

    return request.app.state.vector_store


def get_checkpointer(request: Request) -> BasePostgresSaver:
    """Get checkpointer instance from request object initialized during startup"""

    return request.app.state.checkpointer


def get_store(request: Request) -> BasePostgresStore:
    """Get store instance from request object initialized during startup"""

    return request.app.state.store


async def get_session(request: Request) -> AsyncGenerator[AsyncSession, None]:
    """Yield an async database session per request lifecycle"""

    async with request.app.state.db.session_factory() as session:
        yield session


async def get_register_form(request: Request) -> RegisterForm:
    """Validate the register form object model used during the authentication in order to raise custom error"""

    form_data = await request.form()
    try:
        return RegisterForm.model_validate(form_data)
    except ValidationError:
        raise FormRegisterException("Please check the data you entered.")


async def get_login_form(request: Request) -> LoginForm:
    """Validate the login form object model used during the authentication in order to raise custom error"""

    form_data = await request.form()
    try:
        return LoginForm.model_validate(form_data)
    except ValidationError:
        raise FormLoginException("Please check the data you entered.")


async def get_current_user(
    session: Annotated[AsyncSession, Depends(get_session)],
    token_cookie: Annotated[str | None, Cookie(alias="access_token")] = None,
) -> UserModel:
    """Authenticate API requests based on token returning the active user or redirect to login page"""

    if not token_cookie:
        raise AuthenticationRequiredError()

    try:
        payload = jwt.decode(
            token_cookie, S.SECRET_KEY.get_secret_value(), algorithms=[S.ALGORITHM], options={"require": ["exp", "sub"]}
        )
        user_id = payload.get("sub")
    except InvalidTokenError:
        raise AuthenticationRequiredError()

    # Retrieve the current user if it is not disabled.
    query = select(UserModel).where(UserModel.id == user_id, UserModel.is_active.is_(True))
    user = (await session.execute(query)).scalar_one_or_none()
    if not user:
        raise AuthenticationRequiredError()

    return user


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

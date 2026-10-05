from collections.abc import AsyncGenerator
from typing import Annotated

import jwt
from fastapi import Cookie, Depends, HTTPException, Request
from jwt.exceptions import InvalidTokenError
from langchain_postgres import PGVectorStore
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from chatbot.db.config import db
from chatbot.db.models import UserModel
from chatbot.settings import S


def get_vector_store(request: Request) -> PGVectorStore:
    """Get PGVectorStore instance from request object initialized during startup"""

    return request.app.state.vector_store


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async database session per request lifecycle.

    :return: Async generator yielding an active database session
    :rtype: AsyncGenerator[AsyncSession, None]
    """
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

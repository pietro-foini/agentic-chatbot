from typing import Annotated

from fastapi import Depends, HTTPException, Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from chatbot.api.dependencies import get_current_user, get_session
from chatbot.db.models import UserModel
from chatbot.settings import S


async def enforce_rate_limit(
    session: AsyncSession,
    bucket_key: str,
    limit: int,
    window_seconds: int,
) -> None:
    """Atomically increment a PostgreSQL request bucket and reject excess requests."""

    result = await session.execute(
        text("""
            INSERT INTO rate_limit_buckets (bucket_key, window_started_at, request_count)
            VALUES (:bucket_key, now(), 1)
            ON CONFLICT (bucket_key) DO UPDATE
            SET
                window_started_at = CASE
                    WHEN rate_limit_buckets.window_started_at
                        + (:window_seconds * INTERVAL '1 second') <= now()
                    THEN now()
                    ELSE rate_limit_buckets.window_started_at
                END,
                request_count = CASE
                    WHEN rate_limit_buckets.window_started_at
                        + (:window_seconds * INTERVAL '1 second') <= now()
                    THEN 1
                    ELSE rate_limit_buckets.request_count + 1
                END
            RETURNING
                request_count,
                GREATEST(
                    1,
                    CEIL(EXTRACT(EPOCH FROM (
                        window_started_at + (:window_seconds * INTERVAL '1 second') - now()
                    )))
                )::integer AS retry_after
            """),
        {"bucket_key": bucket_key, "window_seconds": window_seconds},
    )
    bucket = result.one()

    await session.execute(text("""
            WITH expired AS (
                SELECT bucket_key
                FROM rate_limit_buckets
                WHERE window_started_at < now() - INTERVAL '1 day'
                ORDER BY window_started_at
                LIMIT 100
            )
            DELETE FROM rate_limit_buckets
            USING expired
            WHERE rate_limit_buckets.bucket_key = expired.bucket_key
              AND rate_limit_buckets.window_started_at < now() - INTERVAL '1 day'
            """))

    await session.commit()

    if bucket.request_count > limit:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please try again later.",
            headers={"Retry-After": str(bucket.retry_after)},
        )


async def limit_login_requests(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> None:
    """Limit login attempts for each client IP."""

    client_ip = request.client.host if request.client else "unknown"
    await enforce_rate_limit(session, f"login:{client_ip}", S.LOGIN_LIMIT, S.AUTH_WINDOW_SECONDS)


async def limit_registration_requests(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> None:
    """Limit registrations for each client IP."""

    client_ip = request.client.host if request.client else "unknown"
    await enforce_rate_limit(session, f"register:{client_ip}", S.REGISTER_LIMIT, S.AUTH_WINDOW_SECONDS)


async def limit_chat_requests(
    session: Annotated[AsyncSession, Depends(get_session)],
    current_user: Annotated[UserModel, Depends(get_current_user)],
) -> None:
    """Limit chat requests for each authenticated user."""

    await enforce_rate_limit(session, f"chat:{current_user.id}", S.CHAT_LIMIT, S.CHAT_WINDOW_SECONDS)

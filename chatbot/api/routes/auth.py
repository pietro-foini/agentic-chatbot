from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Form, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from chatbot.api.dependencies import get_register_form, get_session
from chatbot.db.models import UserModel
from chatbot.errors import FormRegisterException
from chatbot.schemas.auth import RegisterForm
from chatbot.services.security import create_access_token, get_password_hash, verify_password
from chatbot.settings import S

auth_router = APIRouter(tags=["Auth"])

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent.parent / "templates"))


@auth_router.get(path="/login", response_class=HTMLResponse)
async def login_page(request: Request) -> HTMLResponse:
    """Render the HTML login page"""

    return templates.TemplateResponse(request=request, name="login.html")


@auth_router.get(path="/register", response_class=HTMLResponse)
async def register_page(request: Request) -> HTMLResponse:
    """Render the HTML registration page"""

    return templates.TemplateResponse(request=request, name="register.html")


@auth_router.post(path="/register")
async def register_form_submit(
    session: Annotated[AsyncSession, Depends(get_session)],
    form: Annotated[RegisterForm, Depends(get_register_form)],
) -> Response:
    """Register a new user account if username and email are not already registered and redirect to login page"""

    # Check if the user has been already registered.
    query = select(UserModel).where(or_(UserModel.username == form.username, UserModel.email == form.email))
    user = (await session.execute(query)).scalar_one_or_none()
    if user:
        raise FormRegisterException("Username or email already exists.")

    # Add new user to the user table for authentication.
    new_user = UserModel(
        username=form.username,
        email=form.email,
        full_name=form.full_name,
        hashed_password=get_password_hash(form.password),
    )
    session.add(new_user)

    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise FormRegisterException("Username or email already exists.")

    return RedirectResponse(url="/login", status_code=303)


@auth_router.post(path="/login")
async def login_form_submit(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
    username: Annotated[str, Form(min_length=5, max_length=50)],
    password: Annotated[str, Form(min_length=8, max_length=128)],
) -> Response:
    """Process login form submissions and redirect to main page"""

    query = select(UserModel).where(UserModel.username == username)
    user = (await session.execute(query)).scalar_one_or_none()
    if not user or not verify_password(password, user.hashed_password):
        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={"error": "Incorrect username or password."},
            status_code=400,
        )

    # Create access token.
    access_token = create_access_token(data={"sub": str(user.id)})

    redirect = RedirectResponse(url="/", status_code=303)
    redirect.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        secure=S.COOKIE_SECURE,
        samesite="lax",
        max_age=S.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )

    return redirect


@auth_router.post("/logout")
async def logout() -> RedirectResponse:
    """Clear user session cookie and redirect client to login page"""

    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(key="access_token")

    return response

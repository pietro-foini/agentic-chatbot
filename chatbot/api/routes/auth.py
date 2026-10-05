from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Form, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from chatbot.api.dependencies import get_session
from chatbot.db.models import UserModel
from chatbot.security import create_access_token, get_password_hash, verify_password
from chatbot.settings import S

auth_router = APIRouter(tags=["Auth"])

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent.parent / "templates"))


@auth_router.get(path="/login", response_class=HTMLResponse)
async def login_page(request: Request) -> HTMLResponse:
    """Render the HTML login page for browser clients"""

    return templates.TemplateResponse(request=request, name="login.html")


@auth_router.get(path="/register", response_class=HTMLResponse)
async def register_page(request: Request) -> HTMLResponse:
    """Render the HTML registration page for browser clients"""

    return templates.TemplateResponse(request=request, name="register.html")


@auth_router.post(path="/register", response_class=HTMLResponse)
async def register_form_submit(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
    username: str = Form(...),
    email: str = Form(...),
    full_name: str = Form(None),
    password: str = Form(...),
) -> Response:
    """Register a new user account if username and email are not already registered"""

    # Check if the user already exist.
    query = select(UserModel).where(or_(UserModel.username == username, UserModel.email == email))
    existing_user = (await session.execute(query)).scalar_one_or_none()

    if existing_user:
        return templates.TemplateResponse(
            request=request,
            name="register.html",
            context={"error": "Username or email already exists."},
            status_code=400,
        )

    # Add user to the user table.
    new_user = UserModel(
        username=username,
        email=email,
        full_name=full_name,
        hashed_password=get_password_hash(password),
    )
    session.add(new_user)

    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        return templates.TemplateResponse(
            request=request,
            name="register.html",
            context={"error": "Username or email already exists."},
            status_code=400,
        )

    access_token = create_access_token(data={"sub": new_user.username})
    response = RedirectResponse(url="/", status_code=303)
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        secure=False,  # True for production environment
        samesite="lax",
        max_age=S.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )
    return response


@auth_router.post(path="/login", response_class=HTMLResponse)
async def login_form_submit(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
    username: str = Form(...),
    password: str = Form(...),
) -> Response:
    """Process login form submissions from browser and set access token cookie"""

    query = select(UserModel).where(UserModel.username == username)
    user = (await session.execute(query)).scalar_one_or_none()

    if not user or not verify_password(password, user.hashed_password):
        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={"error": "Incorrect username or password."},
            status_code=400,
        )

    access_token = create_access_token(data={"sub": user.username})

    redirect = RedirectResponse(url="/", status_code=303)
    redirect.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        secure=False,  # Set True in production
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

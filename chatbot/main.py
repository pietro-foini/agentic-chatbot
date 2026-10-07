from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from langchain_postgres import PGVectorStore

from chatbot.api.routes import auth_router, chat_router
from chatbot.db.config import checkpointer, connection_pool, db, pg_engine, store
from chatbot.embeddings.config import embeddings
from chatbot.errors import AuthenticationRequiredError, UserFacingError
from chatbot.graphs.v2.agent import LangGraphAgent
from chatbot.settings import S

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Define the startup and shutdown logic"""

    # Open connections at the application startup.
    await connection_pool.open()
    await db.connect()

    try:
        app.state.vector_store = await PGVectorStore.create(
            engine=pg_engine,
            table_name=S.TABLE_NAME_VECTORS,
            embedding_service=embeddings,
            metadata_columns=["source"],
        )
        app.state.graph = LangGraphAgent
        app.state.checkpointer = checkpointer
        app.state.store = store
        app.state.db = db
        yield
    finally:  # Close connections at the application shutdown or crash.
        await connection_pool.close()
        await db.disconnect()


app = FastAPI(
    title="chatbot",
    lifespan=lifespan,
)


@app.exception_handler(AuthenticationRequiredError)
async def auth_exception_handler(request: Request, exc: AuthenticationRequiredError):
    """Raise error when authentication fails"""

    if request.headers.get("HX-Request"):
        return Response(status_code=401, headers={"HX-Redirect": "/login"})

    # Browser standard.
    if "text/html" in request.headers.get("Accept", ""):
        return RedirectResponse(url="/login", status_code=303)

    # Swagger UI / API Client.
    return JSONResponse(
        status_code=401,
        content={"detail": "Not authenticated"},
        headers={"WWW-Authenticate": "Bearer"},
    )


@app.exception_handler(UserFacingError)
async def user_exception_handler(request: Request, exc: UserFacingError):
    return templates.TemplateResponse(
        request=request,
        name="partials/error_message.html",
        context={"error": str(exc)},
        status_code=200,
    )


app.include_router(auth_router)
app.include_router(chat_router)

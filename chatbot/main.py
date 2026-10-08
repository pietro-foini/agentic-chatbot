from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from langchain_postgres import PGEngine, PGVectorStore
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from chatbot.api.routes import auth_router, chat_router
from chatbot.db.postgres import PostgresAsyncHandler
from chatbot.embeddings.config import embeddings
from chatbot.errors import AuthenticationRequiredError, FormRegisterException, UserFacingError
from chatbot.graphs.v1.agent import build_agent
from chatbot.settings import S

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Define the application startup and shutdown lifecycle."""

    # Initialize postgres database handler.
    db = PostgresAsyncHandler(
        db_uri=S.asyncpg_uri,
        readonly=False,
        statement_timeout_ms=S.SQL_STATEMENT_TIMEOUT_MS,
    )

    # Initialize and open the async connection pool.
    async with AsyncConnectionPool(
        conninfo=S.postgres_uri,
        open=False,
        kwargs={"autocommit": True, "row_factory": dict_row},
    ) as pool:
        await db.connect()

        try:
            # Initialize LangGraph checkpointer and key-value store.
            checkpointer = AsyncPostgresSaver(pool)
            store = AsyncPostgresStore(pool)

            # Initialize PGVector store.
            vector_store = await PGVectorStore.create(
                engine=PGEngine.from_engine(engine=db.engine),
                table_name=S.TABLE_NAME_VECTORS,
                embedding_service=embeddings,
                metadata_columns=["source"],
            )

            # Build the agent instance with initialized dependencies.
            graph = build_agent(checkpointer=checkpointer, store=store)

            # Attach resources to app.state for dependency injection in routes.
            app.state.pool = pool
            app.state.db = db
            app.state.checkpointer = checkpointer
            app.state.store = store
            app.state.vector_store = vector_store
            app.state.graph = graph

            # Application runs here until shutdown is triggered.
            yield

        finally:
            # Graceful shutdown logic.
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


@app.exception_handler(FormRegisterException)
async def form_exception_handler(request: Request, exc: FormRegisterException):
    """Handle the exception raised when the register form validation fails"""

    return templates.TemplateResponse(
        request=request,
        name="register.html",
        context={"error": str(exc)},
        status_code=200,
    )


app.include_router(auth_router)
app.include_router(chat_router)

from contextlib import asynccontextmanager

from fastapi import FastAPI
from langchain_postgres import PGVectorStore

from chatbot.api.routes import chat_router
from chatbot.db.config import async_pool, engine, pg_engine
from chatbot.embeddings.config import embeddings
from chatbot.settings import S


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Define the startup and shutdown logic"""

    # Open pool at the application startup.
    await async_pool.open()

    # Initialize Vector Database.
    vector_store = await PGVectorStore.create(
        engine=pg_engine,
        table_name=S.TABLE_NAME_VECTORS,
        embedding_service=embeddings,
        metadata_columns=["source"],
    )
    app.state.vector_store = vector_store

    yield

    # Close pool at the application stop.
    await async_pool.close()
    await engine.dispose()


app = FastAPI(
    title="chatbot",
    lifespan=lifespan,
)

app.include_router(chat_router, tags=["Chat"])

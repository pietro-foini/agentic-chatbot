from contextlib import asynccontextmanager

from fastapi import FastAPI
from langchain_postgres import PGVectorStore

from chatbot.api.routes import auth_router, chat_router
from chatbot.db.config import db, pg_engine, pool
from chatbot.embeddings.config import embeddings
from chatbot.settings import S


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Define the startup and shutdown logic"""

    # Open connections at the application startup.
    await pool.open()
    await db.connect()

    try:
        app.state.vector_store = await PGVectorStore.create(
            engine=pg_engine,
            table_name=S.TABLE_NAME_VECTORS,
            embedding_service=embeddings,
            metadata_columns=["source"],
        )
        yield
    finally:  # Close connections at the application shutdown or crash.
        await pool.close()
        await db.disconnect()


app = FastAPI(
    title="chatbot",
    lifespan=lifespan,
)

app.include_router(auth_router)
app.include_router(chat_router)

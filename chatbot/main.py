from contextlib import asynccontextmanager

from fastapi import FastAPI
from langchain_postgres import PGEngine, PGVectorStore

from chatbot.api.routes import auth_router, chat_router
from chatbot.db.config import checkpointer, db, pool
from chatbot.embeddings.config import embeddings
from chatbot.settings import S


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Define the startup and shutdown logic"""

    # Open connections at the application startup.
    await pool.open()
    await db.connect()

    # Create tables if they do not exist.
    await db.init_db()
    await checkpointer.setup()

    # Initialize Vector Database.
    vector_store = await PGVectorStore.create(
        engine=PGEngine.from_engine(engine=db.engine),
        table_name=S.TABLE_NAME_VECTORS,
        embedding_service=embeddings,
        metadata_columns=["source"],
    )
    app.state.vector_store = vector_store

    yield

    # Close connections at the application shutdown.
    await pool.close()
    await db.disconnect()


app = FastAPI(
    title="chatbot",
    lifespan=lifespan,
)

app.include_router(auth_router)
app.include_router(chat_router)

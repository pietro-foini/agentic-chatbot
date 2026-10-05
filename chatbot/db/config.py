from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from chatbot.db.postgres import PostgresAsyncHandler
from chatbot.settings import S

# Strictly required by LangGraph (AsyncPostgresSaver).
pool = AsyncConnectionPool(
    conninfo=f"postgresql://{S.database_url}",
    open=False,
    kwargs={
        "autocommit": True,
        "prepare_threshold": 0,
    },
)
checkpointer = AsyncPostgresSaver(pool)

# Required by langchain-postgres (PGEngine / PGVectorStore) and authentication.
db = PostgresAsyncHandler(
    db_uri=f"postgresql+asyncpg://{S.database_url}",
    readonly=False,
    statement_timeout_ms=S.SQL_STATEMENT_TIMEOUT_MS,
)
get_db_session = db.get_session

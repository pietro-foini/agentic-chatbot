from langchain_postgres import PGEngine
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from chatbot.db.postgres import PostgresAsyncHandler
from chatbot.settings import S

# Strictly required by LangGraph (AsyncPostgresSaver).
connection_pool = AsyncConnectionPool(
    conninfo=f"postgresql://{S.DB_USER}:{S.DB_PASSWORD.get_secret_value()}@{S.DB_HOST}:{S.DB_PORT}/{S.DB_NAME}",
    open=False,
    kwargs={
        "autocommit": True,
        "row_factory": dict_row,
    },
)
# Define the checkpointer for only keeping the graph messages and the light state.
checkpointer = AsyncPostgresSaver(connection_pool)
# Define the store for persistent heavy data.
store = AsyncPostgresStore(connection_pool)

# Required by langchain-postgres (PGEngine / PGVectorStore) and authentication.
db = PostgresAsyncHandler(
    db_uri=f"postgresql+asyncpg://{S.DB_USER}:{S.DB_PASSWORD.get_secret_value()}@{S.DB_HOST}:{S.DB_PORT}/{S.DB_NAME}",
    readonly=False,
    statement_timeout_ms=S.SQL_STATEMENT_TIMEOUT_MS,
)
pg_engine = PGEngine.from_engine(engine=db.engine)

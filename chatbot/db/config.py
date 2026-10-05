from langchain_postgres import PGEngine
from psycopg_pool import AsyncConnectionPool
from sqlalchemy.ext.asyncio import create_async_engine

from chatbot.settings import S

# Async connection pool.
async_pool = AsyncConnectionPool(
    conninfo=f"postgresql://{S.DB_USER}:{S.DB_PASSWORD.get_secret_value()}@{S.DB_HOST}:{S.DB_PORT}/{S.DB_NAME}",
    max_size=10,
    min_size=1,
    open=False,
)
# Async pg engine.
engine = create_async_engine(
    f"postgresql+asyncpg://{S.DB_USER}:{S.DB_PASSWORD.get_secret_value()}@{S.DB_HOST}:{S.DB_PORT}/{S.DB_NAME}",
    pool_size=5,
    max_overflow=5,
    pool_pre_ping=True,
)
pg_engine = PGEngine.from_engine(engine=engine)


def create_readonly_async_engine(db_name: str):
    """Create a short-lived async engine forced to read-only with a statement timeout"""

    db_uri = (
        f"postgresql+asyncpg://{S.DB_USER}:{S.DB_PASSWORD.get_secret_value()}" f"@{S.DB_HOST}:{S.DB_PORT}/{db_name}"
    )

    return create_async_engine(
        db_uri,
        connect_args={
            "server_settings": {
                "default_transaction_read_only": "on",
                "statement_timeout": str(S.SQL_STATEMENT_TIMEOUT_MS),
            }
        },
    )

import urllib

from langchain_postgres import PGEngine
from psycopg_pool import AsyncConnectionPool
from sqlalchemy.ext.asyncio import create_async_engine

from chatbot.settings import S

# Create connections for the database storing graph agent information.
DB_URI = f"postgresql://{S.DB_USER}:{urllib.parse.quote_plus(S.DB_PASSWORD)}@{S.DB_HOST}:{S.DB_PORT}/{S.DB_NAME}"

# Async connection pool.
async_pool = AsyncConnectionPool(conninfo=DB_URI, max_size=20, open=False)
# Async pg engine.
engine = create_async_engine(DB_URI)
pg_engine = PGEngine.from_engine(engine=engine)


def create_readonly_async_engine(db_name: str):
    """Create a short-lived async engine forced to read-only with a statement timeout"""

    db_uri = (
        f"postgresql+asyncpg://{S.DB_USER}:{urllib.parse.quote_plus(S.DB_PASSWORD)}"
        f"@{S.DB_HOST}:{S.DB_PORT}/{db_name}"
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

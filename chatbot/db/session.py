import urllib

from langchain_postgres import PGEngine
from psycopg_pool import AsyncConnectionPool
from sqlalchemy.ext.asyncio import create_async_engine

from chatbot.settings import S

DB_URI = f"postgresql://{S.DB_USER}:{urllib.parse.quote_plus(S.DB_PASSWORD)}@{S.DB_HOST}:{S.DB_PORT}/{S.DB_NAME}"

# Async connection pool.
async_pool = AsyncConnectionPool(conninfo=DB_URI, max_size=20, open=False)

# Async pg engine.
engine = create_async_engine(DB_URI)
pg_engine = PGEngine.from_engine(engine=engine)

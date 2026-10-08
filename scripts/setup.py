import asyncio

from langchain_postgres.v2.engine import Column, PGEngine
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore
from psycopg_pool import AsyncConnectionPool

from chatbot.db.postgres import PostgresAsyncHandler
from chatbot.settings import S


async def init_database() -> None:
    """Initialize relational, checkpoint, store, and vector database schemas"""

    db = PostgresAsyncHandler(
        db_uri=S.postgres_uri,
        readonly=False,
        statement_timeout_ms=S.SQL_STATEMENT_TIMEOUT_MS,
    )
    await db.connect()

    try:
        await db.init_db()

        async with AsyncConnectionPool(
            conninfo=S.asyncpg_uri,
            kwargs={"autocommit": True},
        ) as pool:
            await pool.open()

            async with pool.connection() as conn:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            checkpointer = AsyncPostgresSaver(pool)
            await checkpointer.setup()

            store = AsyncPostgresStore(pool)
            await store.setup()

        try:
            await PGEngine.from_engine(engine=db.engine).ainit_vectorstore_table(
                table_name=S.TABLE_NAME_VECTORS,
                vector_size=S.EMBEDDINGS_SIZE,
                metadata_columns=[Column(name="source", data_type="text")],
                overwrite_existing=False,
            )
        except (Exception,):
            pass

    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(init_database())

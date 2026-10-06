import asyncio

from langchain_postgres.v2.engine import Column

from chatbot.db.config import checkpointer, connection_pool, db, pg_engine, store
from chatbot.settings import S


async def main() -> None:
    """Creation of the different tables at the project initialization"""

    await connection_pool.open()
    await db.connect()

    await db.init_db()
    await checkpointer.setup()
    await store.setup()
    await pg_engine.ainit_vectorstore_table(
        table_name=S.TABLE_NAME_VECTORS,
        vector_size=S.EMBEDDINGS_SIZE,
        metadata_columns=[Column(name="source", data_type="text")],
        overwrite_existing=False,
    )

    await connection_pool.close()
    await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())

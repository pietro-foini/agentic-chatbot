from typing import Any, Mapping

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from chatbot.db.models import Base


class PostgresAsyncHandler:
    """PostgreSQL async handler implementation"""

    def __init__(self, db_uri: str, readonly: bool = True, statement_timeout_ms: int = 30000) -> None:
        """Initialize a PostgreSQL async handler"""

        server_settings: dict[str, str] = {
            "statement_timeout": str(statement_timeout_ms),
        }
        if readonly:
            server_settings["default_transaction_read_only"] = "on"

        self.engine: AsyncEngine = create_async_engine(
            db_uri,
            connect_args={"server_settings": server_settings},
            pool_pre_ping=True,
        )
        self.session_factory = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )

    async def init_db(self) -> None:
        """Create all tables defined in Base.metadata if they do not exist"""

        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def connect(self) -> None:
        """Verify the connection is alive"""

        async with self.engine.connect() as conn:
            await conn.execute(text("SELECT 1;"))

    async def disconnect(self) -> None:
        """Dispose of the connection pool"""

        await self.engine.dispose()

    async def query(self, query: str, params: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        """Execute a query and return rows as dictionaries"""

        async with self.engine.connect() as conn:
            result = await conn.execute(text(query), params or {})
            if result.returns_rows:
                return [{str(k): v for k, v in row._mapping.items()} for row in result.fetchall()]
            return []

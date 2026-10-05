import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from chatbot.settings import S


class Base(DeclarativeBase):
    """Base declarative class for all SQLAlchemy ORM (Object-Relational Mapping) models"""

    pass


class VectorStore(Base):
    """Table for storing vectors to be compatible with LangChain PGVectorStore"""

    __tablename__ = S.TABLE_NAME_VECTORS

    langchain_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(1024), nullable=False)
    langchain_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    source: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)

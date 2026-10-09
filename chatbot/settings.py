from typing import Literal, Self
from urllib.parse import quote_plus

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Store here environment variables"""

    ENVIRONMENT: Literal["local", "cloud"] = Field(description="Where the app is running")

    # Authentication.
    SECRET_KEY: SecretStr = Field(min_length=32, description="The secret key for the authentication")
    ALGORITHM: Literal["HS256"] = Field(default="HS256", description="The algorithm to use for authentication")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=60, description="The number of minutes after which the access token expires"
    )
    COOKIE_SECURE: bool = Field(description="Enable secure cookies; disable only for local development")

    # LLM.
    # Default.
    MODEL_NAME: str = Field(default="qwen2.5:7b", description="The default LLM model")
    # Ollama.
    OLLAMA_MODELS: list[str] = Field(
        default=["qwen2.5:7b", "gemma4:12b"], description="The names of the Ollama LLM models"
    )

    # Embeddings.
    # Default.
    EMBEDDINGS_NAME: str = Field(default="bge-m3:latest", description="The name of the embedding model")
    EMBEDDINGS_SIZE: int = Field(default=1024, description="The size of the embedding model")

    # Database PostgreSQL.
    DB_USER: str = Field(description="The user of the PostgreSQL database")
    DB_PASSWORD: SecretStr = Field(description="The password of the PostgreSQL database")
    DB_NAME: str = Field(description="The PostgreSQL database name")
    # Local.
    DB_HOST: str | None = Field(default=None, description="The host of the PostgreSQL database")
    DB_PORT: int | None = Field(default=None, description="The port of the PostgreSQL database")
    # Cloud.
    INSTANCE_NAME: str | None = Field(default=None, description="Cloud SQL instance name")

    TABLE_NAME_VECTORS: str = Field(
        default="vectors",
        description="Database table used to store vector embeddings for similarity search",
    )

    # Assistant.
    CHATBOT_NAME: str = Field(default="Genius", description="The chatbot name")
    TEMPERATURE: float = Field(ge=0, le=1, default=0.9, description="Controls randomness in LLM responses")
    MAX_TOKENS: int = Field(
        default=1000,
        description="Maximum number of tokens the LLM can generate per request",
    )
    THINKING_BUDGET: int = Field(
        default=1024,
        description="Internal LLM processing budget for a single request",
    )
    TIMEOUT_LLM: int = Field(
        default=120,
        description="Timeout for single LLM call",
    )
    MAX_ATTACHMENTS_SIZE: int = Field(default=1024, description="Maximum file size per attachment in kilobytes")
    RECURSION_LIMIT: int = Field(
        default=25,
        description="Max recursive steps the agent can perform per request",
    )
    MAX_ATTEMPTS: int = Field(
        default=3,
        description="Maximum retries if an agent node fails during execution",
    )
    SQL_STATEMENT_TIMEOUT_MS: int = Field(
        default=8000,
        description="PostgreSQL statement_timeout applied to agent SQL connections, in milliseconds",
    )
    SQL_RESULT_LIMIT: int = Field(
        default=100,
        description="Maximum number of rows returned by agent SQL queries",
    )
    PLOT_EXEC_TIMEOUT_SECONDS: int = Field(
        default=8,
        description="Maximum seconds allowed for generated plot code to run",
    )
    ALLOWED_MIME_TYPES: list[str] = Field(
        default=["application/pdf", "image/png", "image/jpeg"],
        description="Allowed mime types for attachment",
    )

    @model_validator(mode="after")
    def _check_db_config(self) -> Self:
        """Checks that the database config is valid"""

        if self.ENVIRONMENT == "cloud" and not self.INSTANCE_NAME:
            raise ValueError("INSTANCE_NAME is required when ENVIRONMENT=cloud")
        if self.ENVIRONMENT == "local" and not self.DB_HOST and not self.DB_PORT:
            raise ValueError("DB_HOST and DB_PORT are required when ENVIRONMENT=local")
        return self

    @property
    def postgres_uri(self) -> str:
        """URI for standard PostgreSQL drivers (psycopg / psycopg_pool)"""

        pwd = quote_plus(self.DB_PASSWORD.get_secret_value())
        if self.ENVIRONMENT == "cloud":
            return f"postgresql://{self.DB_USER}:{pwd}@/{self.DB_NAME}?host=/cloudsql/{self.PROJECT_ID}:{self.REGION}:{self.INSTANCE_NAME}"
        return f"postgresql://{self.DB_USER}:{pwd}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    @property
    def asyncpg_uri(self) -> str:
        """URI for SQLAlchemy + asyncpg driver"""

        pwd = quote_plus(self.DB_PASSWORD.get_secret_value())
        if self.ENVIRONMENT == "cloud":
            return f"postgresql+asyncpg://{self.DB_USER}:{pwd}@/{self.DB_NAME}?host=/cloudsql/{self.PROJECT_ID}:{self.REGION}:{self.INSTANCE_NAME}"
        return f"postgresql+asyncpg://{self.DB_USER}:{pwd}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"


S = Settings()  # type: ignore[call-arg]

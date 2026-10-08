from typing import Literal
from urllib.parse import quote_plus

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Store here environment variables"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Authentication.
    SECRET_KEY: SecretStr = Field(min_length=32, description="The secret key for the authentication")
    ALGORITHM: Literal["HS256", "HS384", "HS512"] = Field(
        default="HS256", description="The algorithm to use for authentication"
    )
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=60, description="The number of minutes after which the access token expires"
    )

    # Google Cloud Platform setting.
    PROJECT_ID: str = Field(description="The ID of the Google Cloud Platform project")

    # Google Cloud Storage.
    GCS_PROCESSED_BUCKET: str = Field(description="The bucket name where to store data")

    # LLM.
    # Default.
    MODEL_NAME: str = Field(default="gemini-2.5-flash-lite", description="The default LLM model")
    # Google.
    GOOGLE_MODELS: dict[str, str] = Field(
        default={
            "gemini-2.5-flash-lite": "europe-west1",
            "gemini-3.5-flash-lite": "global",
            "gemini-3.8-flash": "global",
        },
        description="The available Google LLM models with the corresponding region",
    )
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
    DB_HOST: str = Field(description="The host of the PostgreSQL database")
    DB_PORT: int = Field(description="The port of the PostgreSQL database")
    DB_NAME: str = Field(description="The PostgreSQL database name")
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

    @property
    def postgres_uri(self) -> str:
        """URI for standard PostgreSQL drivers (psycopg / psycopg_pool)"""
        return (
            f"postgresql://{self.DB_USER}:{quote_plus(self.DB_PASSWORD.get_secret_value())}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @property
    def asyncpg_uri(self) -> str:
        """URI for SQLAlchemy + asyncpg driver"""
        return (
            f"postgresql+asyncpg://{self.DB_USER}:{quote_plus(self.DB_PASSWORD.get_secret_value())}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )


S = Settings()  # type: ignore[call-arg]

from enum import StrEnum
from typing import Literal, Self
from urllib.parse import quote_plus

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings


class Environment(StrEnum):
    """Environment to use"""

    LOCAL = "local"
    CLOUD = "cloud"


class CloudProvider(StrEnum):
    """Cloud provider"""

    GCP = "gcp"
    AWS = "aws"


class Settings(BaseSettings):
    """Store here environment variables"""

    # RUNTIME & ACTIVE PROVIDER.
    ENVIRONMENT: Environment = Field(
        default=Environment.LOCAL,
        description="Physical deployment target: local workstation or cloud container",
    )
    CLOUD_PROVIDER: CloudProvider = Field(
        default=CloudProvider.GCP,
        description="Active provider for managed services (Storage, LLMs, Embeddings)",
    )

    # AUTHENTICATION.
    SECRET_KEY: SecretStr = Field(min_length=32, description="JWT/Cookie secret key")
    ALGORITHM: Literal["HS256"] = Field(default="HS256", description="JWT token algorithm")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=60, description="Token TTL in minutes")
    COOKIE_SECURE: bool = Field(default=False, description="Set True in production")

    # STORAGE.
    STORAGE_BUCKET: str = Field(default="chatbot-genius", description="Target bucket name for either S3 or GCS")

    # LLM CONFIGURATION.
    LLM_MODEL: str = Field(description="Default LLM model selected for chat requests")
    GOOGLE_LLM_MODELS: list[str] = Field(
        default=["gemini-2.5-flash-lite", "gemini-3.5-flash-lite", "gemini-3.8-flash"],
        description="Available Google models running on cloud",
    )
    AWS_LLM_MODELS: list[str] = Field(
        default=["amazon.nova-micro-v1:0", "amazon.nova-lite-v1:0", "amazon.nova-pro-v1:0"],
        description="Available AWS Bedrock models running on cloud",
    )
    OLLAMA_LLM_MODELS: list[str] = Field(
        default=["qwen2.5:7b", "gemma4:12b"],
        description="Available Ollama models running locally",
    )

    # EMBEDDINGS CONFIGURATION.
    EMBEDDINGS_MODEL: str = Field(description="Default Embeddings model selected for chat requests")
    EMBEDDINGS_SIZE: int = Field(description="Embeddings dimension size")
    GOOGLE_EMBEDDINGS_MODELS: list[str] = Field(
        default=["models/gemini-embedding-001"],
        description="Available Google models for embedding running on cloud",
    )
    AWS_EMBEDDINGS_MODELS: list[str] = Field(
        default=[],
        description="Available AWS models for embedding running on cloud",
    )
    OLLAMA_EMBEDDINGS_MODELS: list[str] = Field(
        default=["bge-m3:latest"],
        description="Available Ollama models for embedding running locally",
    )

    # DATABASE (PostgreSQL / AWS RDS / GCP Cloud SQL).
    DB_USER: str = Field(description="PostgreSQL username")
    DB_PASSWORD: SecretStr = Field(description="PostgreSQL password")
    DB_NAME: str = Field(description="PostgreSQL database name")
    # Used for Localhost and AWS RDS (both use standard host:port)
    DB_HOST: str | None = Field(default=None, description="PostgreSQL host")
    DB_PORT: int | None = Field(default=None, description="PostgreSQL port")
    # Used ONLY for GCP Cloud SQL (Unix domain socket: project:region:instance-name)
    INSTANCE_NAME: str | None = Field(default=None, description="Cloud SQL connection name")

    # Name of the table for storing vectors/embeddings.
    TABLE_NAME_VECTORS: str = Field(default="vectors", description="Database table used for vector embeddings")

    # ASSISTANT & AGENT RUNTIME.
    CHATBOT_NAME: str = Field(default="Genius", description="Assistant display name")
    TEMPERATURE: float = Field(ge=0.0, le=1.0, default=0.7)
    MAX_TOKENS: int = Field(default=2048)
    THINKING_BUDGET: int = Field(default=1024)
    TIMEOUT_LLM: int = Field(default=120)
    MAX_ATTACHMENTS_SIZE: int = Field(default=1024)
    RECURSION_LIMIT: int = Field(default=25)
    MAX_ATTEMPTS: int = Field(default=3)
    SQL_STATEMENT_TIMEOUT_MS: int = Field(default=8000)
    SQL_RESULT_LIMIT: int = Field(default=100)
    PLOT_EXEC_TIMEOUT_SECONDS: int = Field(default=8)
    ALLOWED_MIME_TYPES: list[str] = Field(default=["application/pdf"])

    @model_validator(mode="after")
    def _validate_configuration(self) -> Self:
        """Validate configuration"""

        if (
            self.ENVIRONMENT == Environment.CLOUD
            and self.CLOUD_PROVIDER == CloudProvider.GCP
            and not self.INSTANCE_NAME
        ):
            raise ValueError("INSTANCE_NAME is required when ENVIRONMENT=cloud and CLOUD_PROVIDER=GCP")

        if self.ENVIRONMENT == Environment.CLOUD and (not self.DB_HOST or not self.DB_PORT):
            raise ValueError("DB_HOST and DB_PORT are required for when ENVIRONMENT=local")

        return self

    @property
    def all_models(self) -> list[str]:
        """Aggregate list of all enabled models across environment and providers"""

        if self.ENVIRONMENT == Environment.LOCAL and self.CLOUD_PROVIDER == CloudProvider.AWS:
            return self.OLLAMA_LLM_MODELS + self.AWS_LLM_MODELS
        elif self.ENVIRONMENT == Environment.LOCAL and self.CLOUD_PROVIDER == CloudProvider.GCP:
            return self.OLLAMA_LLM_MODELS + self.GOOGLE_LLM_MODELS
        if self.ENVIRONMENT == Environment.CLOUD and self.CLOUD_PROVIDER == CloudProvider.AWS:
            return self.AWS_LLM_MODELS
        elif self.ENVIRONMENT == Environment.CLOUD and self.CLOUD_PROVIDER == CloudProvider.GCP:
            return self.GOOGLE_LLM_MODELS
        else:
            return []

    # TODO: mettere controllo su modelli default

    @property
    def postgres_uri(self) -> str:
        """Synchronous URI for psycopg"""

        pwd = quote_plus(self.DB_PASSWORD.get_secret_value())

        if self.ENVIRONMENT == Environment.CLOUD and self.CLOUD_PROVIDER == CloudProvider.GCP:
            return f"postgresql://{self.DB_USER}:{pwd}@/{self.DB_NAME}?host=/cloudsql/{self.INSTANCE_NAME}"

        return f"postgresql://{self.DB_USER}:{pwd}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    @property
    def asyncpg_uri(self) -> str:
        """Asynchronous URI for SQLAlchemy and asyncpg"""

        pwd = quote_plus(self.DB_PASSWORD.get_secret_value())

        if self.ENVIRONMENT == Environment.CLOUD and self.CLOUD_PROVIDER == CloudProvider.GCP:
            return f"postgresql+asyncpg://{self.DB_USER}:{pwd}@/{self.DB_NAME}?host=/cloudsql/{self.INSTANCE_NAME}"

        return f"postgresql+asyncpg://{self.DB_USER}:{pwd}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"


S = Settings()  # type: ignore[call-arg]

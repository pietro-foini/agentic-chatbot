from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Store here env variable."""

    # Google Cloud Platform.
    PROJECT_ID: str = Field(description="The ID of the Google Cloud Platform project")
    REGION: str = Field(description="The region of the Google Cloud Platform project")
    GCS_PROCESSED_BUCKET: str = Field(description="The bucket name where to store data")

    # Database PostgreSQL.
    DB_USER: str = Field(description="The user of the PostgreSQL database")
    DB_PASSWORD: str = Field(description="The password of the PostgreSQL database")
    DB_HOST: str = Field(description="The host of the PostgreSQL database")
    DB_PORT: str = Field(description="The port of the PostgreSQL database")
    DB_NAME: str = Field(description="The PostgreSQL database name")
    TABLE_NAME_VECTORS: str = Field(
        default="ml_vectors",
        description="Database table used to store vector embeddings for similarity search",
    )

    # Assistant.
    CHATBOT_NAME: str = Field(default="Genius", description="The chatbot name")
    TEMPERATURE: float | int = Field(default=0.9, description="Controls randomness in LLM responses")
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
    MINUTES_TEMPORARY_URL: int = Field(
        default=720,
        description="Duration (in minutes) a signed Cloud Storage URL remains valid",
    )
    RECURSION_LIMIT: int = Field(
        default=15,
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


S = Settings()  # type: ignore[call-arg]

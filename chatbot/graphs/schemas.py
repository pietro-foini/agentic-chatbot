import re

from pydantic import BaseModel, Field, field_validator

from chatbot.graphs.data import DBName, Source
from chatbot.graphs.security import wrap_sql_with_limit
from chatbot.settings import S


class ListTablesInput(BaseModel):
    """Input schema for listing available tables"""

    db_name: DBName = Field(description="Name of the database whose tables to list")


class DescribeTableInput(BaseModel):
    """Input schema for describing a table"""

    db_name: DBName = Field(description="Name of the database")
    table_name: str = Field(description="Name of the table to inspect")


class SQLQueryInput(BaseModel):
    """Input schema for executing SQL queries on PostgreSQL"""

    query: str = Field(
        description="The SQL query to execute on PostgreSQL. Do not include explanations, comments, or extra text"
    )
    db_name: DBName = Field(
        description="The name of the database to user for applying the query",
    )
    ref_table_name: str = Field(
        description="A unique logical name to store this result used as reference in subsequent tool calls. Use descriptive names like 'sales_data' or 'user_metrics'",
    )

    @field_validator("query", mode="before")
    @classmethod
    def check_safety(cls, v: str) -> str:
        """Allow only a single read-only SELECT and cap the number of rows."""

        return wrap_sql_with_limit(v, limit=S.SQL_RESULT_LIMIT)


class CodePlotInput(BaseModel):
    """Input schema for generating plots from Python code"""

    ref_table_name: str = Field(description="The name of the table stored in the store object to use for this plot")
    code: str = Field(
        description="Python plotting code only. A populated pandas DataFrame named `df` is ALREADY loaded. DO NOT create, mock, or import `df` or any data. Use `df` directly with sns/plt. Do NOT import libraries (pd, sns, plt are pre-loaded)."
    )
    description: str = Field(description="A brief argumentative and not technical description of the generated plot")

    @field_validator("code", mode="after")
    @classmethod
    def sanitize_plot_code(cls, code: str) -> str:
        code = re.sub(r"^import\s+.*$", "", code, flags=re.MULTILINE)
        code = re.sub(r"^from\s+.*\s+import\s+.*$", "", code, flags=re.MULTILINE)
        code = code.strip()
        return code


class PDFRetrieveInput(BaseModel):
    """Input schema for downloading and analyzing a PDF file from Google Cloud Storage"""

    gcs_uri: str = Field(description="Google Cloud Storage URI to a PDF file.", pattern=r"(?i)^gs://.+\.pdf$")


class RAGInput(BaseModel):
    """Input schema for the Retrieve Information tool (RAG-based search)"""

    query: str = Field(
        description="The user's query to search for relevant information from the selected source through RAG approach"
    )
    source: Source = Field(description="The source of information to query. Example: `confluence` for ML documentation")

import re

from pydantic import BaseModel, Field, field_validator

from chatbot.graphs.data import DBName, Source


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
    table_name: str = Field(
        description="A unique name to store this result. Use descriptive names like 'sales_data' or 'user_metrics'",
    )

    @field_validator("query", mode="before")
    @classmethod
    def check_safety(cls, v: str) -> str:
        """Field validator that ensures the SQL query does not contain forbidden commands"""

        forbidden_keywords = ["DELETE", "INSERT", "UPDATE", "TRUNCATE"]

        pattern = r"\b(" + "|".join(forbidden_keywords) + r")\b"

        if re.search(pattern, v, flags=re.IGNORECASE):
            raise ValueError(f"Forbidden commands: {forbidden_keywords}")

        return v

    @field_validator("query", mode="after")
    @classmethod
    def limit_query(cls, v: str) -> str:
        """Wrap the query to limit results"""

        original_query = v.strip().rstrip(";")
        wrapped_query = f"SELECT * FROM ({original_query}) LIMIT 100"

        return wrapped_query


class CodePlotInput(BaseModel):
    """Input schema for generating plots from Python code"""

    table_name: str = Field(description="The name of the table stored in the store object to use for this plot")
    code: str = Field(description="""
        Python code using seaborn to create a beautiful, publication-ready plot. 

        CRITICAL INSTRUCTIONS:
        - A pandas DataFrame named `df` is already injected at runtime. DO NOT create or modify any data.
        - Do NOT generate any synthetic data or DataFrames.
        - Only generate code that manipulates `df` to produce the plot.
        - Do NOT import libraries (pd, sns, plt are pre-loaded).

        REQUIRED STYLING GUIDELINES:
        - Use seaborn style: sns.set_style('whitegrid') for better readability
        - Set figure size appropriately: plt.figure(figsize=(12, 6)) for wide plots or (10, 8) for square plots
        - Use seaborn color palettes: sns.color_palette('husl'), 'Set2', 'deep', or 'muted'
        - Add clear, descriptive labels: plt.xlabel('Label', fontsize=12, fontweight='bold')
        - Add informative title: plt.title('Title', fontsize=14, fontweight='bold', pad=20)
        - Rotate x-axis labels if needed: plt.xticks(rotation=45, ha='right')
        - Add grid for better readability: plt.grid(True, alpha=0.3, linestyle='--')
        - Use tight layout: plt.tight_layout() to prevent label cutoff
        - Set appropriate font sizes: use fontsize=11 for axis labels, fontsize=10 for tick labels
        - For categorical plots, consider using palette parameter in seaborn functions
        - Add legends when multiple series: plt.legend(fontsize=10, frameon=True, shadow=True)
        - Use despine for cleaner look: sns.despine() to remove top and right spines
        - Set background color if needed: fig.patch.set_facecolor('white')
        """)
    description: str = Field(description="A brief argumentative and not technical description of the generated plot")

    @field_validator("code", mode="after")
    @classmethod
    def remove_imports(cls, code: str) -> str:
        code = re.sub(r"^import\s+.*$", "", code, flags=re.MULTILINE)
        code = re.sub(r"^from\s+.*\s+import\s+.*$", "", code, flags=re.MULTILINE)
        return code.strip()


class PDFRetrieveInput(BaseModel):
    """Input schema for downloading and analyzing a PDF file from Google Cloud Storage"""

    gcs_uri: str = Field(description="Google Cloud Storage URI to a PDF file.", pattern=r"(?i)^gs://.+\.pdf$")


class RAGInput(BaseModel):
    """Input schema for the Retrieve Information tool (RAG-based search)"""

    query: str = Field(
        description="The user's query to search for relevant information from the selected source through RAG approach"
    )
    source: Source = Field(description="The source of information to query. Example: `confluence` for ML documentation")

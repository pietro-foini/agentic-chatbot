from pydantic import BaseModel, Field

from chatbot.graphs.data import Source


class PDFRetrieveInput(BaseModel):
    """Input schema for downloading and analyzing a PDF file from Google Cloud Storage"""

    gcs_uri: str = Field(description="Google Cloud Storage URI to a PDF file.", pattern=r"(?i)^gs://.+\.pdf$")


class RAGInput(BaseModel):
    """Input schema for the Retrieve Information tool (RAG-based search)"""

    query: str = Field(
        description="The user's query to search for relevant information from the selected source through RAG approach"
    )
    source: Source = Field(description="The source of information to query. Example: `confluence` for ML documentation")

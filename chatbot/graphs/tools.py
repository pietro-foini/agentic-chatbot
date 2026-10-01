import base64
import re
from datetime import timedelta
from pathlib import Path

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from chatbot.graphs.data import OutputTool, Source
from chatbot.graphs.models import PDFRetrieveInput, RAGInput
from chatbot.settings import S
from chatbot.storage.config import storage_client


@tool("download_and_analyze_pdf_from_gcs", args_schema=PDFRetrieveInput, response_format="content")
def download_and_analyze_pdf_from_gcs(gcs_uri: str) -> OutputTool:
    """
    Retrieve, download and analyze a single PDF file from Google Cloud Storage.

    :param gcs_uri: Google Cloud Storage URI in format 'gs://bucket-name/path/to/file.pdf'
    :type gcs_uri: str
    :return: message status, with a Base64 encoded PDF artifact
    :rtype: OutputTool
    """

    try:
        bucket_name, blob_path = re.match("gs://(.*?)/(.*)", gcs_uri).groups()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.get_blob(blob_path)

        if not blob.exists():
            return OutputTool(status="error", content="File not found")

        pdf_bytes = blob.download_as_bytes()
        pdf_base64 = base64.b64encode(pdf_bytes)
        size = round(len(pdf_bytes) / 1024, 2)

        if size > S.MAX_ATTACHMENTS_SIZE:
            return OutputTool(
                status="success",
                content=[
                    {
                        "type": "text",
                        "text": "File successfully downloaded, but it cannot be analyzed because it exceeds size limit",
                    },
                ],
                artifact={"type": "pdf", "base64": f"data:application/pdf;base64,{pdf_base64.decode('utf-8')}"},
            )

        bucket = storage_client.bucket(S.GCS_PROCESSED_BUCKET)
        blob = bucket.blob(f"downloads/{Path(blob_path).name}")
        blob.upload_from_string(pdf_bytes, content_type="application/pdf")

        return OutputTool(
            status="success",
            content=[
                {"type": "text", "text": "File successfully downloaded"},
                {"type": "media", "mime_type": "application/pdf", "file_uri": gcs_uri},
            ],
            artifact={"type": "pdf", "base64": f"data:application/pdf;base64,{pdf_base64.decode('utf-8')}"},
        )
    except Exception as e:
        return OutputTool(status="error", content=f"Error processing file: {e}")


@tool("retrieve_information_from_source", args_schema=RAGInput, response_format="content")
async def retrieve_information_from_source(query: str, source: Source, runtime: ToolRuntime) -> OutputTool:
    """
    Retrieve relevant information from a knowledge source using a RAG (Retrieval-Augmented Generation) approach.
    This tool searches a vector store to find content related to a user query. Currently, the supported sources are:

    - `unimatica`: contains information about italian CCNL Unimatica contract.

    :param query: the query to use for retrieving information from documentation
    :type query: str
    :param source: the knowledge source to query
    :type source: Source
    :param runtime: the runtime (the parameter is not visible to the model)
    :type runtime: ToolRuntime
    :return: the information extracted from the source
    :rtype: OutputTool
    """

    try:
        docs = await runtime.context.vector_store.asimilarity_search(query, filter={"source": {"$eq": source.value}})
        return OutputTool(status="success", content="\n\n".join([doc.page_content for doc in docs]))
    except Exception as e:
        return OutputTool(status="error", content=f"Error retrieving information: {e}")

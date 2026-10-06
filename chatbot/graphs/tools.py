import asyncio
import base64
import json
import re
import uuid
from io import BytesIO

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from chatbot.db.postgres import PostgresAsyncHandler
from chatbot.graphs.data import DBName, Source
from chatbot.graphs.schemas import (
    CodePlotInput,
    DescribeTableInput,
    ListTablesInput,
    PDFRetrieveInput,
    RAGInput,
    SQLQueryInput,
)
from chatbot.graphs.security import validate_plot_code
from chatbot.graphs.state import Context
from chatbot.settings import S
from chatbot.storage.config import storage_client


@tool("list_tables", args_schema=ListTablesInput, response_format="content")
async def list_tables(db_name: DBName) -> str:
    """
    Retrieve a list of all available tables in a specified PostgreSQL database.

    :param db_name: name of the database to list tables from
    :return: list of table names
    """

    # Define the database.
    db = PostgresAsyncHandler(
        db_uri=f"postgresql+asyncpg://{S.DB_USER}:{S.DB_PASSWORD.get_secret_value()}@{S.DB_HOST}:{S.DB_PORT}/{db_name}",
        readonly=True,
        statement_timeout_ms=S.SQL_STATEMENT_TIMEOUT_MS,
    )

    try:
        query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public';
        """
        result = await db.query(query)
        return json.dumps(result)
    finally:
        await db.disconnect()


@tool("describe_table", args_schema=DescribeTableInput, response_format="content")
async def describe_table(db_name: DBName, table_name: str) -> str:
    """
    Retrieve information about a PostgreSQL table, like information, schema and column details.

    :param db_name: name of the database containing the table
    :param table_name: the name of the table to search info about
    :return: the table information
    """

    # Define the database.
    db = PostgresAsyncHandler(
        db_uri=f"postgresql+asyncpg://{S.DB_USER}:{S.DB_PASSWORD.get_secret_value()}@{S.DB_HOST}:{S.DB_PORT}/{db_name}",
        readonly=True,
        statement_timeout_ms=S.SQL_STATEMENT_TIMEOUT_MS,
    )

    try:
        query = """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = :table_name;
        """
        result = await db.query(query, {"table_name": table_name})
        return json.dumps(result)
    finally:
        await db.disconnect()


@tool("execute_postgresql_sql", args_schema=SQLQueryInput, response_format="content_and_artifact")
async def execute_sql(
    query: str, db_name: DBName, ref_table_name: str, runtime: ToolRuntime[Context]
) -> tuple[dict | str, dict | None]:
    """
    Executes a SQL query on PostgreSQL and returns the result as a parsed table.

    :param query: the SQL query to execute on PostgreSQL
    :param db_name: name of the database to use for applying the query
    :param ref_table_name: logical name assigned to the table results used as reference in subsequent tool calls
    :param runtime: the runtime (parameter is not visible to the model)
    :return: the query results and the corresponding artifact data
    """

    # Define the database.
    db = PostgresAsyncHandler(
        db_uri=f"postgresql+asyncpg://{S.DB_USER}:{S.DB_PASSWORD.get_secret_value()}@{S.DB_HOST}:{S.DB_PORT}/{db_name}",
        readonly=True,
        statement_timeout_ms=S.SQL_STATEMENT_TIMEOUT_MS,
    )

    try:
        # Run query.
        result = await db.query(query)

        # Parse results.
        df = pd.DataFrame(result)
        records = df.to_dict(orient="records")

        # Convert to base64.
        csv_base64 = base64.b64encode(df.to_csv(index=False, encoding="utf-8").encode("utf-8")).decode("utf-8")

        # Store data in order to not overhead the messages thread.
        await runtime.store.aput(
            namespace=("artifacts", runtime.context.thread_id),
            key=ref_table_name,
            value={
                "mime_type": "text/csv",
                "data": f"data:text/csv;base64,{csv_base64}",
                "records": records,
            },
        )

        if len(df) > 20 or len(df.columns) > 10:
            content = {
                "table_name": ref_table_name,
                "shape": df.shape,
                "schema": [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns],
                "data": f"Previewing {min(5, len(df))} rows out of {len(df)} rows: {json.dumps(records[:5])}. "
                f"Download to view all data.",
                "truncated": True,
            }
            artifact_data = {"table_name": ref_table_name, "mime_type": "text/csv"}

            return content, artifact_data
        else:
            content = {
                "table_name": ref_table_name,
                "shape": df.shape,
                "schema": [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns],
                "data": json.dumps(records),
                "truncated": False,
            }

            return content, None
    finally:
        await db.disconnect()


@tool("create_plot", args_schema=CodePlotInput, response_format="content_and_artifact")
async def create_plot(
    code: str, description: str, ref_table_name: str, runtime: ToolRuntime[Context]
) -> tuple[str, dict | None]:
    """
    Executes Python code to create a custom seaborn plot and provide an argumentative related description.

    :param code: Python code using seaborn to create a plot starting from provided data
    :param description: a brief description of the generated graph
    :param ref_table_name: the name of the table stored in the store previously generated object to use for this plot
    :param runtime: the runtime (parameter is not visible to the model)
    :return: graph description with a Base64 encoded PNG image artifact
    """

    matplotlib.use("Agg")

    item = await runtime.store.aget(namespace=("artifacts", runtime.context.thread_id), key=ref_table_name)

    if item is None:
        return f"Table '{ref_table_name}' not found.", None

    df = pd.DataFrame(item.value["records"])

    tree = validate_plot_code(code)
    compiled = compile(tree, "<plot>", "exec")

    safe_builtins = {
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "set": set,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "len": len,
        "sum": sum,
        "min": min,
        "max": max,
        "sorted": sorted,
        "reversed": reversed,
        "any": any,
        "all": all,
        "abs": abs,
        "round": round,
    }

    safe_namespace = {
        "plt": plt,
        "sns": sns,
        "np": np,
        "pd": pd,
        "df": df,
        "__builtins__": safe_builtins,
    }

    def _render_plot() -> bytes:
        exec(compiled, safe_namespace)
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        _image_bytes = buf.read()
        buf.close()
        return _image_bytes

    try:
        image_bytes = await asyncio.wait_for(asyncio.to_thread(_render_plot), timeout=S.PLOT_EXEC_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        plt.close("all")
        return "Plot generation timed out", None

    plt.close("all")

    # Assign ad ID to the current file.
    file_id = str(uuid.uuid4())

    # Store data in order to not overhead the messages thread.
    await runtime.store.aput(
        namespace=("artifacts", runtime.context.thread_id),
        key=file_id,
        value={
            "mime_type": "image/png",
            "data": f"data:image/png;base64,{base64.b64encode(image_bytes).decode('utf-8')}",
        },
    )

    # Format the output: content and artifact.
    content = description
    artifact_data = {"file_id": file_id, "mime_type": "image/png"}

    return content, artifact_data


@tool("download_and_analyze_pdf_from_gcs", args_schema=PDFRetrieveInput, response_format="content_and_artifact")
async def download_and_analyze_pdf_from_gcs(
    gcs_uri: str, runtime: ToolRuntime[Context]
) -> tuple[list[dict] | str, dict | None]:
    """
    Download and analyze a single PDF file from Google Cloud Storage.

    :param gcs_uri: Google Cloud Storage URI
    :param runtime: the runtime (parameter is not visible to the model)
    :return: the content retrieval information and the corresponding artifact data
    """

    # Retrieve the blob already validated by the pydantic model.
    bucket_name, blob_path = re.match(pattern="gs://(.*?)/(.*)", string=gcs_uri).groups()  # type: ignore
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    # Check if the file exists.
    exists = await asyncio.to_thread(blob.exists)
    if not exists:
        return "File not found", None

    # Check the file size.
    await asyncio.to_thread(blob.reload)
    size_kb = round(blob.size / 1024, 2)
    if size_kb > S.MAX_ATTACHMENTS_SIZE:
        return "File exceeds size limit and was not downloaded", None

    # Download the file from Cloud Storage.
    pdf_bytes = await asyncio.to_thread(blob.download_as_bytes)

    # Assign ad ID to the current file.
    file_id = str(uuid.uuid4())

    # Store data in order to not overhead the messages thread.
    await runtime.store.aput(
        namespace=("artifacts", runtime.context.thread_id),
        key=file_id,
        value={
            "mime_type": "application/pdf",
            "data": f"data:application/pdf;base64,{base64.b64encode(pdf_bytes).decode('utf-8')}",
        },
    )

    # Format the output: content and artifact.
    content = [
        {"type": "text", "text": "File successfully downloaded"},
        {"type": "file", "url": gcs_uri, "mime_type": "application/pdf"},
    ]
    artifact_data = {"file_id": file_id, "mime_type": "application/pdf"}

    return content, artifact_data


@tool("retrieve_information", args_schema=RAGInput, response_format="content")
async def retrieve_information(query: str, source: Source, runtime: ToolRuntime[Context]) -> str:
    """
    Retrieve relevant information from a knowledge source using a RAG (Retrieval-Augmented Generation) approach.

    :param query: the query to use for retrieving information from documentation
    :param source: the knowledge source to query
    :param runtime: the runtime (the parameter is not visible to the model)
    :return: the information extracted from the source
    """

    # Retrieve the most similar chunks.
    docs = await runtime.context.vector_store.asimilarity_search(query, filter={"source": {"$eq": source.value}})
    # Merge most representative chunks together into a unique string.
    content = "\n\n".join([doc.page_content for doc in docs])

    return content

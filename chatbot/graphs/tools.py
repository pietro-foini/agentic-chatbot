import asyncio
import base64
import json
import re
from io import BytesIO
from pathlib import Path
from typing import Any

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
from chatbot.settings import S
from chatbot.storage.config import storage_client


@tool("list_tables", args_schema=ListTablesInput, response_format="content")
async def list_tables(db_name: DBName) -> str:
    """
    Retrieve a list of all available tables in a specified PostgreSQL database.

    :param db_name: name of the database to list tables from
    :return: list of table names
    """

    db = None
    try:
        db = PostgresAsyncHandler(
            db_uri=f"postgresql+asyncpg://{S.DB_USER}:{S.DB_PASSWORD.get_secret_value()}@{S.DB_HOST}:{S.DB_PORT}/{db_name}",
            readonly=True,
            statement_timeout_ms=S.SQL_STATEMENT_TIMEOUT_MS,
        )
        query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public';
        """
        result = await db.query(query)
        return json.dumps(result)
    except Exception as e:
        return f"Error retrieving tables: {e}"
    finally:
        if db is not None:
            await db.disconnect()


@tool("describe_table", args_schema=DescribeTableInput, response_format="content")
async def describe_table(db_name: DBName, table_name: str) -> str:
    """
    Retrieve information about a PostgreSQL table, like information, schema and column details.

    :param db_name: name of the database containing the table
    :param table_name: the name of the table to search info about
    :return: the table information
    """

    db = None
    try:
        db = PostgresAsyncHandler(
            db_uri=f"postgresql+asyncpg://{S.DB_USER}:{S.DB_PASSWORD.get_secret_value()}@{S.DB_HOST}:{S.DB_PORT}/{db_name}",
            readonly=True,
            statement_timeout_ms=S.SQL_STATEMENT_TIMEOUT_MS,
        )
        query = """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = :table_name;
        """
        result = await db.query(query, {"table_name": table_name})
        return json.dumps(result)
    except Exception as e:
        return f"Error retrieving schema: {e}"
    finally:
        if db is not None:
            await db.disconnect()


@tool("execute_postgresql_sql", args_schema=SQLQueryInput, response_format="content_and_artifact")
async def execute_sql(query: str, db_name: str, table_name: str, runtime: ToolRuntime) -> tuple[dict | str, Any]:
    """
    Executes a SQL query on PostgreSQL and returns the result as a parsed table.

    :param query: the SQL query to execute on PostgreSQL
    :param db_name: name of the database to user for applying the query
    :param table_name: logical name assigned to the resulting table, used to reference in subsequent tool calls
    :param runtime: the runtime (parameter is not visible to the model)
    :return: query results as a JSON table, with a table artifact
    """

    db = None
    try:
        db = PostgresAsyncHandler(
            db_uri=f"postgresql+asyncpg://{S.DB_USER}:{S.DB_PASSWORD.get_secret_value()}@{S.DB_HOST}:{S.DB_PORT}/{db_name}",
            readonly=True,
            statement_timeout_ms=S.SQL_STATEMENT_TIMEOUT_MS,
        )
        result = await db.query(query)

        # Parse results.
        df = pd.DataFrame(result)
        records = df.to_dict(orient="records")

        # Store table on store.
        runtime.store.put(("tables",), table_name, records)

        if len(df) > 20 or len(df.columns) > 10:
            csv_base64 = base64.b64encode(df.to_csv(index=False, encoding="utf-8").encode("utf-8")).decode("utf-8")

            content = {
                "table_name": table_name,
                "shape": df.shape,
                "schema": [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns],
                "data": f"Previewing {min(5, len(df))} rows out of {len(df)} rows: {json.dumps(records[:5])}. "
                f"Download to view all data.",
                "truncated": True,
            }
            artifact = {"type": "table", "base64": f"data:text/csv;base64,{csv_base64}"}

            return content, artifact
        else:
            content = {
                "table_name": table_name,
                "shape": df.shape,
                "schema": [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns],
                "data": json.dumps(records),
                "truncated": False,
            }
            return content, None

    except Exception as e:
        return f"Error executing query: {e}", None
    finally:
        if db is not None:
            await db.disconnect()


@tool("create_plot", args_schema=CodePlotInput, response_format="content_and_artifact")
async def create_plot(
    code: str, description: str, table_name: str, runtime: ToolRuntime
) -> tuple[list | str, dict | None]:
    """
    Executes Python code to create a custom seaborn plot and provide an argumentative related description.

    :param code: Python code using seaborn to create a plot starting from provided data
    :param description: a brief description of the generated graph
    :param table_name: the name of the table stored in the store object to use for this plot
    :param runtime: the runtime (parameter is not visible to the model)
    :return: graph description, with a Base64 encoded PNG image artifact
    """

    matplotlib.use("Agg")

    item = runtime.store.get(("tables",), table_name)

    if item is None:
        return "No data available context to generate a plot.", None

    df = pd.DataFrame(item.value)

    try:
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
            image_bytes = buf.read()
            buf.close()
            return image_bytes

        try:
            image_bytes = await asyncio.wait_for(asyncio.to_thread(_render_plot), timeout=S.PLOT_EXEC_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            plt.close("all")
            return "Plot generation timed out", None

        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        plt.close("all")

        artifact_data = {"type": "image", "base64": f"data:image/png;base64,{image_base64}"}

        return description, artifact_data
    except Exception as e:
        plt.close("all")
        return f"Error generating plot: {e}", None


@tool("download_and_analyze_pdf_from_gcs", args_schema=PDFRetrieveInput, response_format="content_and_artifact")
async def download_and_analyze_pdf_from_gcs(gcs_uri: str) -> tuple[list | str, dict | None]:
    """
    Retrieve, download and analyze a single PDF file from Google Cloud Storage.

    :param gcs_uri: Google Cloud Storage URI in format 'gs://bucket-name/path/to/file.pdf'
    :return: message, with a Base64 encoded PDF artifact
    """

    try:
        bucket_name, blob_path = re.match("gs://(.*?)/(.*)", gcs_uri).groups()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)

        exists = await asyncio.to_thread(blob.exists)
        if not exists:
            return "File not found", None

        pdf_bytes = await asyncio.to_thread(blob.download_as_bytes)
        pdf_base64 = base64.b64encode(pdf_bytes)
        size = round(len(pdf_bytes) / 1024, 2)

        artifact_data = {"type": "pdf", "base64": f"data:application/pdf;base64,{pdf_base64.decode('utf-8')}"}

        if size > S.MAX_ATTACHMENTS_SIZE:
            content = [
                {
                    "type": "text",
                    "text": "File successfully downloaded, but it cannot be analyzed because it exceeds size limit",
                }
            ]
            return content, artifact_data

        bucket = storage_client.bucket(S.GCS_PROCESSED_BUCKET)
        blob = bucket.blob(f"downloads/{Path(blob_path).name}")
        await asyncio.to_thread(blob.upload_from_string, pdf_bytes, content_type="application/pdf")

        content = [
            {"type": "text", "text": "File successfully downloaded"},
            {"type": "media", "mime_type": "application/pdf", "file_uri": gcs_uri},
        ]
        return content, artifact_data

    except Exception as e:
        return f"Error processing file: {e}", None


@tool("retrieve_information", args_schema=RAGInput, response_format="content")
async def retrieve_information(query: str, source: Source, runtime: ToolRuntime) -> str:
    """
    Retrieve relevant information from a knowledge source using a RAG (Retrieval-Augmented Generation) approach.
    This tool searches a vector store to find content related to a user query. Currently, the supported sources are:

    - `unimatica`: contains information about italian CCNL Unimatica contract.

    :param query: the query to use for retrieving information from documentation
    :param source: the knowledge source to query
    :param runtime: the runtime (the parameter is not visible to the model)
    :return: the information extracted from the source
    """

    try:
        docs = await runtime.context.vector_store.asimilarity_search(query, filter={"source": {"$eq": source.value}})
        return "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        return f"Error retrieving information: {e}"

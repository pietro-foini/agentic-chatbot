import base64
import json
import re
import urllib
from io import BytesIO
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from langchain.tools import ToolRuntime
from langchain_core.tools import tool
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from chatbot.graphs.data import DBName, OutputTool, Source
from chatbot.graphs.models import (
    CodePlotInput,
    DescribeTableInput,
    ListTablesInput,
    PDFRetrieveInput,
    RAGInput,
    SQLQueryInput,
)
from chatbot.settings import S
from chatbot.storage.config import storage_client


@tool("list_tables", args_schema=ListTablesInput, response_format="content")
async def list_tables(db_name: DBName, runtime: ToolRuntime) -> OutputTool:
    """
    Retrieve a list of all available tables in a specified PostgreSQL database.

    :param db_name: name of the database to list tables from
    :type db_name: DBName
    :param runtime: the runtime (parameter is not visible to the model)
    :type runtime: ToolRuntime
    :return: list of table names, status and message
    :rtype: OutputTool
    """

    try:
        db_uri = f"postgresql+asyncpg://{S.DB_USER}:{urllib.parse.quote_plus(S.DB_PASSWORD)}@{S.DB_HOST}:{S.DB_PORT}/{db_name}"
        engine = create_async_engine(db_uri)

        query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public';
        """
        async with engine.connect() as conn:
            result = await conn.execute(text(query))
            tables = [row[0] for row in result.fetchall()]

        await engine.dispose()
        return OutputTool(status="success", content=json.dumps(tables))
    except Exception as e:
        return OutputTool(status="error", content=f"Error retrieving tables: {e}")


@tool("describe_table", args_schema=DescribeTableInput, response_format="content")
async def describe_table(db_name: DBName, table_name: str, runtime: ToolRuntime) -> OutputTool:
    """
    Retrieve information about a PostgreSQL table, like information, schema and column details.

    :param db_name: name of the database containing the table
    :type db_name: DBName
    :param table_name: the name of the table to search info about
    :type table_name: str
    :param runtime: the runtime (parameter is not visible to the model)
    :type runtime: ToolRuntime
    :return: the table information, status and message
    :rtype: OutputTool
    """

    try:
        db_uri = f"postgresql+asyncpg://{S.DB_USER}:{urllib.parse.quote_plus(S.DB_PASSWORD)}@{S.DB_HOST}:{S.DB_PORT}/{db_name}"
        engine = create_async_engine(db_uri)

        query = """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = :table_name;
        """
        async with engine.connect() as conn:
            result = await conn.execute(text(query), {"table_name": table_name})
            columns = [dict(row._mapping) for row in result.fetchall()]

        await engine.dispose()
        return OutputTool(status="success", content=json.dumps(columns))
    except Exception as e:
        return OutputTool(status="error", content=f"Error retrieving schema: {e}")


@tool("execute_postgresql_sql", args_schema=SQLQueryInput, response_format="content")
async def execute_sql(query: str, db_name: str, table_name: str, runtime: ToolRuntime) -> OutputTool:
    """
    Executes a SQL query on PostgreSQL and returns the result as a parsed table.

    :param query: the SQL query to execute on PostgreSQL
    :type query: str
    :param db_name: name of the database to user for applying the query
    :type db_name: str
    :param table_name: logical name assigned to the resulting table, used to reference in subsequent tool calls
    :type table_name: str
    :param runtime: the runtime (parameter is not visible to the model)
    :type runtime: ToolRuntime
    :return: query results as a JSON table, with a table artifact
    :rtype: OutputTool
    """

    try:
        db_uri = f"postgresql+asyncpg://{S.DB_USER}:{urllib.parse.quote_plus(S.DB_PASSWORD)}@{S.DB_HOST}:{S.DB_PORT}/{db_name}"

        # Execute query.
        target_engine = create_async_engine(db_uri)
        async with target_engine.connect() as conn:
            result = await conn.execute(text(query))
            rows = [dict(row._mapping) for row in result.fetchall()]

        # Parse results.
        df = pd.DataFrame(rows)
        records = df.to_dict(orient="records")

        # Store table on store.
        runtime.store.put(("tables",), table_name, records)

        if len(df) > 20 or len(df.columns) > 10:
            csv_base64 = base64.b64encode(df.to_csv(index=False, encoding="utf-8").encode("utf-8")).decode("utf-8")

            return OutputTool(
                status="success",
                content={
                    "table_name": table_name,
                    "shape": df.shape,
                    "schema": [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns],
                    "data": f"Previewing {min(5, len(df))} rows out of {len(df)} rows: {json.dumps(records[:5])}. "
                    f"Download to view all data.",
                    "truncated": True,
                },
                artifact={"type": "table", "base64": f"data:text/csv;base64,{csv_base64}"},
            )
        else:
            return OutputTool(
                status="success",
                content={
                    "table_name": table_name,
                    "shape": df.shape,
                    "schema": [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns],
                    "data": json.dumps(records),
                    "truncated": False,
                },
            )
    except Exception as e:
        return OutputTool(status="error", content=f"Error executing query: {e}")


@tool("create_plot", args_schema=CodePlotInput, response_format="content")
def create_plot(code: str, description: str, table_name: str, runtime: ToolRuntime) -> OutputTool:
    """
    Executes Python code to create a custom seaborn plot and provide an argumentative related description.

    :param code: Python code using seaborn to create a plot starting from provided data
    :type code: str
    :param description: a brief description of the generated graph
    :type description: str
    :param table_name: the name of the table stored in the store object to use for this plot
    :type table_name: str
    :param runtime: the runtime (parameter is not visible to the model)
    :type runtime: ToolRuntime
    :return: graph description, with a Base64 encoded PNG image artifact
    :rtype: OutputTool
    """

    matplotlib.use("Agg")

    item = runtime.store.get(("tables",), table_name)

    if item is None:
        return OutputTool(status="error", content="No data available context to generate a plot.")

    df = pd.DataFrame(item.value)

    try:
        # Safe built-ins - only allow specific safe functions
        safe_builtins = {
            # Type conversions
            "int": int,
            "float": float,
            "str": str,
            "bool": bool,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            # Iteration and logic
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
            # Other utilities
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

        # Execute the code
        exec(code, safe_namespace)

        # Capture the current figure to bytes
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)

        # Convert to base64
        image_base64 = base64.b64encode(buf.read()).decode("utf-8")

        # Close the buffer and clear the figure
        buf.close()
        plt.close()

        return OutputTool(
            status="success",
            content=description,
            artifact={"type": "image", "base64": f"data:image/png;base64,{image_base64}"},
        )
    except Exception as e:
        return OutputTool(status="error", content=f"Error generating plot: {e}")


@tool("download_and_analyze_pdf_from_gcs", args_schema=PDFRetrieveInput, response_format="content")
async def download_and_analyze_pdf_from_gcs(gcs_uri: str) -> OutputTool:
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


@tool("retrieve_information", args_schema=RAGInput, response_format="content")
async def retrieve_information(query: str, source: Source, runtime: ToolRuntime) -> OutputTool:
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

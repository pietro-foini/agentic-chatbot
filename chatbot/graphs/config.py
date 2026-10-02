from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from chatbot.db.config import async_pool
from chatbot.graphs.tools import (
    create_plot,
    describe_table,
    download_and_analyze_pdf_from_gcs,
    execute_sql,
    list_tables,
    retrieve_information,
)

ALL_TOOLS = [
    retrieve_information,
    download_and_analyze_pdf_from_gcs,
    execute_sql,
    list_tables,
    describe_table,
    create_plot,
]

SENSITIVE_TOOLS = {download_and_analyze_pdf_from_gcs.name}

postgres_checkpointer = AsyncPostgresSaver(async_pool)

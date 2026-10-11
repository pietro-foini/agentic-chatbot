from chatbot.graphs.tools import (
    create_plot,
    describe_table,
    download_and_analyze_pdf_from_storage,
    execute_sql,
    list_tables,
    retrieve_information,
)

ALL_TOOLS = [
    retrieve_information,
    download_and_analyze_pdf_from_storage,
    execute_sql,
    list_tables,
    describe_table,
    create_plot,
]

SENSITIVE_TOOLS = {download_and_analyze_pdf_from_storage.name}

DEFAULT_USER_PREFERENCES = {"style": "technical", "verbosity": "detailed"}

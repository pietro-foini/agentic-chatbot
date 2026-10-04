from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, InterruptOnConfig

from chatbot.graphs.config import SENSITIVE_TOOLS
from chatbot.graphs.prompts import AGENT_1_PROMPT, AGENT_2_PROMPT, AGENT_3_PROMPT, AGENT_4_PROMPT
from chatbot.graphs.state import Context, State
from chatbot.graphs.tools import (
    create_plot,
    describe_table,
    download_and_analyze_pdf_from_gcs,
    execute_sql,
    list_tables,
    retrieve_information,
)
from chatbot.llm.google import llm

# noinspection PyTypeChecker
agent_1 = create_agent(
    model=llm,
    tools=[
        execute_sql,
        list_tables,
        describe_table,
    ],
    state_schema=State,
    context_schema=Context,
    system_prompt=AGENT_1_PROMPT,
    name="database_specialist",
)


# noinspection PyTypeChecker
agent_2 = create_agent(
    model=llm,
    tools=[
        retrieve_information,
    ],
    state_schema=State,
    context_schema=Context,
    system_prompt=AGENT_2_PROMPT,
    name="research_specialist",
)


# noinspection PyTypeChecker
agent_3 = create_agent(
    model=llm,
    tools=[
        download_and_analyze_pdf_from_gcs,
    ],
    state_schema=State,
    context_schema=Context,
    middleware=[
        HumanInTheLoopMiddleware(interrupt_on={tool: InterruptOnConfig(allowed_decisions=["approve", "reject"])})
        for tool in SENSITIVE_TOOLS
    ],
    system_prompt=AGENT_3_PROMPT,
    name="action_specialist",
)


# noinspection PyTypeChecker
agent_4 = create_agent(
    model=llm,
    tools=[
        create_plot,
    ],
    state_schema=State,
    context_schema=Context,
    system_prompt=AGENT_4_PROMPT,
    name="designer_specialist",
)

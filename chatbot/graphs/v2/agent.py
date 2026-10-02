from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, InterruptOnConfig
from langgraph.store.memory import InMemoryStore

from chatbot.graphs.config import ALL_TOOLS, SENSITIVE_TOOLS, postgres_checkpointer
from chatbot.graphs.prompts import SYSTEM_PROMPT
from chatbot.graphs.state import Context, State
from chatbot.llm.google import llm
from chatbot.settings import S

store = InMemoryStore()

# Create the graph instance.
# noinspection PyTypeChecker
GRAPH = create_agent(
    model=llm,
    tools=ALL_TOOLS,
    state_schema=State,
    context_schema=Context,
    checkpointer=postgres_checkpointer,
    store=store,
    middleware=[
        HumanInTheLoopMiddleware(interrupt_on={tool: InterruptOnConfig(allowed_decisions=["approve", "reject"])})
        for tool in SENSITIVE_TOOLS
    ],
    system_prompt=SYSTEM_PROMPT.format(
        user_name="Pietro",  # TODO: cambialo in modo da personalizzarlo
        bot_name=S.CHATBOT_NAME,
    ),
    name="agent",
)

import operator
from dataclasses import dataclass
from typing import Annotated

from langchain.agents import AgentState
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_postgres import PGVectorStore

__all__ = [
    "State",
    "Context",
]


class State(AgentState):
    """Agent state"""

    total_input_tokens: Annotated[int, operator.add]
    total_output_tokens: Annotated[int, operator.add]


@dataclass
class Context:
    """Context configuration for Agent runtime call"""

    thread_id: str
    chatbot_name: str
    system_prompt: str
    model: BaseChatModel
    tools: list[BaseTool]
    vector_store: PGVectorStore

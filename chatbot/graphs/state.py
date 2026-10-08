import operator
from dataclasses import dataclass
from typing import Annotated, Required, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from langchain_postgres import PGVectorStore
from langgraph.graph import add_messages


class State(TypedDict):
    """Agent state"""

    messages: Required[Annotated[list[BaseMessage], add_messages]]

    total_cost: Annotated[float, operator.add]


@dataclass
class Context:
    """Context configuration for Agent runtime call"""

    thread_id: str
    system_prompt: str
    model: BaseChatModel
    tools: list[BaseTool]
    vector_store: PGVectorStore

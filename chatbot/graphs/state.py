import operator
from dataclasses import dataclass
from typing import Annotated, Required, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AnyMessage
from langchain_core.tools import BaseTool
from langchain_postgres import PGVectorStore
from langgraph.graph import add_messages


class State(TypedDict):
    """Agent state"""

    messages: Required[Annotated[list[AnyMessage], add_messages]]

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

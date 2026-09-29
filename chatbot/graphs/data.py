from enum import StrEnum
from typing import Literal, NotRequired, TypedDict


class OutputTool(TypedDict):
    """The base output tool data structure"""

    status: Literal["success", "error"]
    content: str | list[str | dict] | dict | None
    artifact: NotRequired[dict]


class Source(StrEnum):
    """
    Enum representing different sources of documentation or knowledge.

    - `unimatica`: contains information about italian CCNL unimatica contract.
    """

    unimatica = "unimatica"

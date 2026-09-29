from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Type(StrEnum):
    """The allowed user roles"""

    human = "human"
    ai = "ai"


class Attachment(BaseModel):
    """The base attachment data structure"""

    id: str
    file_uri: str
    type: str
    size: str


class ThinkingStep(BaseModel):
    """The thinking step data structure"""

    step_name: str
    details: dict[str, Any] | str


class BaseResponse(BaseModel):
    """The base response data structure"""

    id: str
    name: str
    content: str


class UserResponse(BaseResponse):
    """The user response data structure"""

    type: Literal[Type.human] = Type.human

    attachment: Attachment | None = None


class BotResponse(BaseResponse):
    """The bot response data structure"""

    type: Literal[Type.ai] = Type.ai

    table: list[dict] | None = None
    image: list[str] = Field(default_factory=list)
    pdf: str | None = None
    thinking: list[ThinkingStep] = Field(default_factory=list)
    response_time: float = 0.0

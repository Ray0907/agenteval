"""Base agent adapter interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from agenteval.models import AgentResponse, Turn


class SessionContext(BaseModel):
    """State passed to the agent adapter for each turn."""
    session_id: str
    turn_number: int
    initial_state: dict[str, Any] = Field(default_factory=dict)
    current_state: dict[str, Any] = Field(default_factory=dict)
    history: list[Turn] = Field(default_factory=list)


class AgentAdapter(ABC):
    """Abstract base class for agent adapters."""

    @abstractmethod
    async def send_message(self, message: str, context: SessionContext) -> AgentResponse:
        ...

    @abstractmethod
    async def reset(self) -> None:
        ...

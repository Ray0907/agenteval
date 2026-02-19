"""LangGraph integration."""
from __future__ import annotations

from typing import Any

from agenteval.adapters.base import AgentAdapter, SessionContext
from agenteval.models import AgentResponse, ToolCall


def _extract_message_text(message: Any) -> str:
    """Extract text content from a LangGraph message (object or dict)."""
    if hasattr(message, "content"):
        return message.content
    if isinstance(message, dict):
        return message.get("content", "")
    return ""


class LangGraphAdapter(AgentAdapter):
    def __init__(self, graph: Any, config: dict | None = None) -> None:
        self._graph = graph
        self._config = config or {}

    async def send_message(self, message: str, context: SessionContext) -> AgentResponse:
        result = await self._graph.ainvoke(
            {"messages": [{"role": "user", "content": message}]},
            config=self._config,
        )
        messages = result.get("messages", [])
        last_message = messages[-1] if messages else None

        text = _extract_message_text(last_message) if last_message else ""

        tool_calls = [
            ToolCall(name=tc.get("name", ""), arguments=tc.get("args", {}))
            for tc in getattr(last_message, "tool_calls", [])
        ] if last_message else []

        state_changes = {k: v for k, v in result.items() if k != "messages"}
        return AgentResponse(message=text, tool_calls=tool_calls, state_changes=state_changes)

    async def reset(self) -> None:
        pass

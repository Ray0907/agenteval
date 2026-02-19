"""OpenAI Agents SDK integration."""
from __future__ import annotations

from typing import Any

from agenteval.adapters.base import AgentAdapter, SessionContext
from agenteval.models import AgentResponse, ToolCall

try:
    from agents import Runner
except ImportError:
    Runner = None


def _extract_tool_calls(result: Any) -> list[ToolCall]:
    """Extract tool calls from an OpenAI Agents SDK result."""
    tool_calls = []
    for item in getattr(result, "new_items", []):
        if not hasattr(item, "name"):
            continue
        arguments = getattr(item, "arguments", None)
        if not isinstance(arguments, dict):
            arguments = {}
        tool_calls.append(ToolCall(name=item.name, arguments=arguments))
    return tool_calls


class OpenAIAgentAdapter(AgentAdapter):
    def __init__(self, agent: Any) -> None:
        self._agent = agent

    async def send_message(self, message: str, context: SessionContext) -> AgentResponse:
        if Runner is None:
            raise ImportError("pip install agenteval[openai]")
        result = await Runner.run(self._agent, input=message)
        return AgentResponse(
            message=result.final_output or "",
            tool_calls=_extract_tool_calls(result),
        )

    async def reset(self) -> None:
        pass

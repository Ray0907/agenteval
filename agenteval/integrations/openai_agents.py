"""OpenAI Agents SDK integration."""
from __future__ import annotations
from typing import Any
from agenteval.adapters.base import AgentAdapter, SessionContext
from agenteval.models import AgentResponse, ToolCall

try:
    from agents import Runner
except ImportError:
    Runner = None


class OpenAIAgentAdapter(AgentAdapter):
    def __init__(self, agent: Any):
        self._agent = agent

    async def send_message(self, message: str, context: SessionContext) -> AgentResponse:
        if Runner is None:
            raise ImportError("pip install agenteval[openai]")
        result = await Runner.run(self._agent, input=message)
        tcs = [ToolCall(name=it.name, arguments=it.arguments if isinstance(getattr(it, "arguments", None), dict) else {})
               for it in getattr(result, "new_items", []) if hasattr(it, "name")]
        return AgentResponse(message=result.final_output or "", tool_calls=tcs)

    async def reset(self):
        pass

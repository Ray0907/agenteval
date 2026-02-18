"""LangGraph integration."""
from __future__ import annotations
from typing import Any
from agenteval.adapters.base import AgentAdapter, SessionContext
from agenteval.models import AgentResponse, ToolCall


class LangGraphAdapter(AgentAdapter):
    def __init__(self, graph: Any, config: dict | None = None):
        self._graph = graph
        self._config = config or {}

    async def send_message(self, message: str, context: SessionContext) -> AgentResponse:
        result = await self._graph.ainvoke(
            {"messages": [{"role": "user", "content": message}]}, config=self._config)
        msgs = result.get("messages", [])
        last = msgs[-1] if msgs else None
        text = getattr(last, "content", "") if last and hasattr(last, "content") else (last or {}).get("content", "")
        tcs = [ToolCall(name=tc.get("name", ""), arguments=tc.get("args", {}))
               for tc in getattr(last, "tool_calls", [])] if last else []
        sc = {k: v for k, v in result.items() if k != "messages"}
        return AgentResponse(message=text, tool_calls=tcs, state_changes=sc)

    async def reset(self):
        pass

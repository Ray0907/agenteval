"""LLM-based agent adapter with swappable providers (Anthropic, OpenAI)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import yaml

from agenteval.adapters.base import AgentAdapter, SessionContext
from agenteval.models import AgentResponse, ToolCall

ToolHandler = Callable[[str, dict[str, Any]], Any]


def _build_tool_schema(tool_def: dict) -> dict:
    """Convert simplified tool definition to JSON Schema."""
    properties = {}
    required = []
    for pname, pdef in tool_def.get("parameters", {}).items():
        if isinstance(pdef, dict):
            properties[pname] = {k: v for k, v in pdef.items()}
        else:
            properties[pname] = {"type": pdef}
        required.append(pname)
    return {"type": "object", "properties": properties, "required": required}


class LLMAdapter(AgentAdapter):
    """Agent adapter that wraps LLM providers (Anthropic, OpenAI).

    Settings can be a dict or path to a YAML file:
        provider: anthropic | openai
        model: model name
        system_prompt: system instructions
        tools: list of tool definitions
    """

    def __init__(
        self,
        settings: dict | str | Path,
        tool_handler: ToolHandler | None = None,
    ) -> None:
        if isinstance(settings, (str, Path)):
            with open(settings) as f:
                settings = yaml.safe_load(f)
        self.provider: str = settings["provider"]
        self.model: str = settings["model"]
        self.system_prompt: str = settings.get("system_prompt", "")
        self.tools_config: list[dict] = settings.get("tools", [])
        self.tool_handler = tool_handler
        self._history: list[dict] = []
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if self.provider == "anthropic":
            from anthropic import AsyncAnthropic
            self._client = AsyncAnthropic()
        elif self.provider == "openai":
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI()
        else:
            raise ValueError(f"Unknown provider: {self.provider}. Use 'anthropic' or 'openai'.")
        return self._client

    def _anthropic_tools(self) -> list[dict]:
        return [
            {
                "name": t["name"],
                "description": t.get("description", ""),
                "input_schema": _build_tool_schema(t),
            }
            for t in self.tools_config
        ]

    def _openai_tools(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": _build_tool_schema(t),
                },
            }
            for t in self.tools_config
        ]

    async def send_message(self, message: str, context: SessionContext) -> AgentResponse:
        self._history.append({"role": "user", "content": message})
        if self.provider == "anthropic":
            return await self._send_anthropic()
        elif self.provider == "openai":
            return await self._send_openai()
        raise ValueError(f"Unknown provider: {self.provider}")

    async def _send_anthropic(self) -> AgentResponse:
        client = self._get_client()
        collected_tools: list[ToolCall] = []
        state_changes: dict[str, Any] = {}
        total_tokens = 0
        all_text: list[str] = []

        while True:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "max_tokens": 4096,
                "messages": self._history,
            }
            if self.system_prompt:
                kwargs["system"] = self.system_prompt
            if self.tools_config:
                kwargs["tools"] = self._anthropic_tools()

            response = await client.messages.create(**kwargs)
            total_tokens += response.usage.input_tokens + response.usage.output_tokens

            tool_uses = []
            for block in response.content:
                if block.type == "text":
                    all_text.append(block.text)
                elif block.type == "tool_use":
                    tool_uses.append(block)

            # Store assistant response in history
            self._history.append({"role": "assistant", "content": response.content})

            if tool_uses and self.tool_handler:
                tool_results = []
                for tu in tool_uses:
                    result = self.tool_handler(tu.name, tu.input)
                    collected_tools.append(
                        ToolCall(name=tu.name, arguments=tu.input, result=result)
                    )
                    if isinstance(result, dict) and "state_changes" in result:
                        state_changes.update(result["state_changes"])
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": json.dumps(result) if not isinstance(result, str) else result,
                    })
                self._history.append({"role": "user", "content": tool_results})
                continue

            return AgentResponse(
                message="\n".join(all_text),
                tool_calls=collected_tools,
                state_changes=state_changes,
                metadata={"tokens": total_tokens},
            )

    async def _send_openai(self) -> AgentResponse:
        client = self._get_client()
        collected_tools: list[ToolCall] = []
        state_changes: dict[str, Any] = {}
        total_tokens = 0
        all_text: list[str] = []

        # Build messages with system prompt prepended
        messages: list[dict] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(self._history)

        while True:
            kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
            if self.tools_config:
                kwargs["tools"] = self._openai_tools()

            response = await client.chat.completions.create(**kwargs)
            if response.usage:
                total_tokens += response.usage.prompt_tokens + response.usage.completion_tokens
            choice = response.choices[0]

            if choice.message.tool_calls and self.tool_handler:
                # Build assistant message dict for history
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": choice.message.content,
                }
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in choice.message.tool_calls
                ]
                messages.append(assistant_msg)
                self._history.append(assistant_msg)

                if choice.message.content:
                    all_text.append(choice.message.content)

                for tc in choice.message.tool_calls:
                    args = json.loads(tc.function.arguments)
                    result = self.tool_handler(tc.function.name, args)
                    collected_tools.append(
                        ToolCall(name=tc.function.name, arguments=args, result=result)
                    )
                    if isinstance(result, dict) and "state_changes" in result:
                        state_changes.update(result["state_changes"])
                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result) if not isinstance(result, str) else result,
                    }
                    messages.append(tool_msg)
                    self._history.append(tool_msg)
                continue

            text = choice.message.content or ""
            if text:
                all_text.append(text)
            self._history.append({"role": "assistant", "content": text})

            return AgentResponse(
                message="\n".join(all_text),
                tool_calls=collected_tools,
                state_changes=state_changes,
                metadata={"tokens": total_tokens},
            )

    async def reset(self) -> None:
        self._history = []

"""Tests for LLM adapter with mocked API calls."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from agenteval.adapters.base import SessionContext


def _ctx():
    return SessionContext(session_id="s1", turn_number=0, initial_state={}, current_state={}, history=[])


def test_import():
    from agenteval.adapters.llm import LLMAdapter
    assert LLMAdapter


def test_load_settings_dict():
    from agenteval.adapters.llm import LLMAdapter
    adapter = LLMAdapter({"provider": "anthropic", "model": "claude-sonnet-4-6"})
    assert adapter.provider == "anthropic"
    assert adapter.model == "claude-sonnet-4-6"


def test_load_settings_yaml(tmp_path):
    from agenteval.adapters.llm import LLMAdapter
    import yaml
    p = tmp_path / "settings.yaml"
    p.write_text(yaml.dump({"provider": "openai", "model": "gpt-4o", "system_prompt": "hi"}))
    adapter = LLMAdapter(str(p))
    assert adapter.provider == "openai"
    assert adapter.model == "gpt-4o"
    assert adapter.system_prompt == "hi"


@pytest.mark.asyncio
async def test_anthropic_simple():
    from agenteval.adapters.llm import LLMAdapter

    mock_response = MagicMock()
    mock_response.content = [MagicMock(type="text", text="Hello!")]
    mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)

    adapter = LLMAdapter({"provider": "anthropic", "model": "claude-sonnet-4-6"})
    adapter._client = mock_client

    resp = await adapter.send_message("hi", _ctx())
    assert resp.message == "Hello!"
    assert resp.metadata["tokens"] == 15


@pytest.mark.asyncio
async def test_anthropic_tool_call():
    from agenteval.adapters.llm import LLMAdapter

    # First response: tool_use
    tool_block = MagicMock(type="tool_use", id="tu1")
    tool_block.name = "lookup"
    tool_block.input = {"id": "1"}
    resp1 = MagicMock()
    resp1.content = [MagicMock(type="text", text="Let me check."), tool_block]
    resp1.usage = MagicMock(input_tokens=20, output_tokens=10)

    # Second response: final text
    resp2 = MagicMock()
    resp2.content = [MagicMock(type="text", text="Found it!")]
    resp2.usage = MagicMock(input_tokens=30, output_tokens=8)

    handler = MagicMock(return_value={"found": True, "name": "Order 1"})

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(side_effect=[resp1, resp2])

    adapter = LLMAdapter(
        {"provider": "anthropic", "model": "claude-sonnet-4-6",
         "tools": [{"name": "lookup", "description": "Look up", "parameters": {"id": {"type": "string"}}}]},
        tool_handler=handler,
    )
    adapter._client = mock_client

    resp = await adapter.send_message("find order 1", _ctx())

    assert "Found it!" in resp.message
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "lookup"
    assert resp.tool_calls[0].arguments == {"id": "1"}
    handler.assert_called_once_with("lookup", {"id": "1"})


@pytest.mark.asyncio
async def test_openai_simple():
    from agenteval.adapters.llm import LLMAdapter

    mock_msg = MagicMock()
    mock_msg.content = "Hello!"
    mock_msg.tool_calls = None

    mock_choice = MagicMock()
    mock_choice.message = mock_msg

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    adapter = LLMAdapter({"provider": "openai", "model": "gpt-4o"})
    adapter._client = mock_client

    resp = await adapter.send_message("hi", _ctx())
    assert resp.message == "Hello!"
    assert resp.metadata["tokens"] == 15


@pytest.mark.asyncio
async def test_openai_tool_call():
    from agenteval.adapters.llm import LLMAdapter

    # First response: tool call
    tc = MagicMock()
    tc.id = "call_1"
    tc.function.name = "lookup"
    tc.function.arguments = '{"id": "1"}'

    msg1 = MagicMock()
    msg1.content = "Let me check."
    msg1.tool_calls = [tc]

    choice1 = MagicMock()
    choice1.message = msg1

    resp1 = MagicMock()
    resp1.choices = [choice1]
    resp1.usage = MagicMock(prompt_tokens=20, completion_tokens=10)

    # Second response: final text
    msg2 = MagicMock()
    msg2.content = "Found it!"
    msg2.tool_calls = None

    choice2 = MagicMock()
    choice2.message = msg2

    resp2 = MagicMock()
    resp2.choices = [choice2]
    resp2.usage = MagicMock(prompt_tokens=30, completion_tokens=8)

    handler = MagicMock(return_value={"found": True})

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=[resp1, resp2])

    adapter = LLMAdapter(
        {"provider": "openai", "model": "gpt-4o",
         "tools": [{"name": "lookup", "description": "Look up", "parameters": {"id": {"type": "string"}}}]},
        tool_handler=handler,
    )
    adapter._client = mock_client

    resp = await adapter.send_message("find order 1", _ctx())

    assert "Found it!" in resp.message
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "lookup"
    handler.assert_called_once_with("lookup", {"id": "1"})


@pytest.mark.asyncio
async def test_reset_clears_history():
    from agenteval.adapters.llm import LLMAdapter
    adapter = LLMAdapter({"provider": "anthropic", "model": "test"})
    adapter._history = [{"role": "user", "content": "hi"}]
    await adapter.reset()
    assert adapter._history == []


def test_unknown_provider():
    from agenteval.adapters.llm import LLMAdapter
    adapter = LLMAdapter({"provider": "unknown", "model": "x"})
    with pytest.raises(ValueError, match="Unknown provider"):
        adapter._get_client()

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def test_import():
    from agenteval.integrations.openai_agents import OpenAIAgentAdapter
    assert OpenAIAgentAdapter


@pytest.mark.asyncio
async def test_send():
    from agenteval.integrations.openai_agents import OpenAIAgentAdapter
    from agenteval.adapters.base import SessionContext
    agent = MagicMock()
    result = MagicMock(final_output="I can help!", new_items=[])
    with patch("agenteval.integrations.openai_agents.Runner") as R:
        R.run = AsyncMock(return_value=result)
        adapter = OpenAIAgentAdapter(agent)
        ctx = SessionContext(session_id="s1", turn_number=0, initial_state={}, current_state={}, history=[])
        resp = await adapter.send_message("hi", ctx)
        assert resp.message == "I can help!"

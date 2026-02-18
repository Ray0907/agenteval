import pytest
from unittest.mock import AsyncMock, MagicMock


def test_import():
    from agenteval.integrations.langgraph import LangGraphAdapter
    assert LangGraphAdapter


@pytest.mark.asyncio
async def test_send():
    from agenteval.integrations.langgraph import LangGraphAdapter
    from agenteval.adapters.base import SessionContext
    g = MagicMock()
    g.ainvoke = AsyncMock(return_value={"messages": [MagicMock(content="hello")]})
    adapter = LangGraphAdapter(g)
    ctx = SessionContext(session_id="s1", turn_number=0, initial_state={}, current_state={}, history=[])
    resp = await adapter.send_message("hi", ctx)
    assert resp.message == "hello"

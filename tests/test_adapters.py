import pytest
from agenteval.models import AgentResponse


def test_session_context_creation():
    from agenteval.adapters.base import SessionContext
    ctx = SessionContext(session_id="s1", turn_number=0, initial_state={"counter": 0},
                        current_state={"counter": 0}, history=[])
    assert ctx.session_id == "s1"
    assert ctx.turn_number == 0


def test_agent_adapter_is_abstract():
    from agenteval.adapters.base import AgentAdapter
    with pytest.raises(TypeError):
        AgentAdapter()


@pytest.mark.asyncio
async def test_concrete_adapter():
    from agenteval.adapters.base import AgentAdapter, SessionContext

    class EchoAdapter(AgentAdapter):
        async def send_message(self, message: str, context: SessionContext) -> AgentResponse:
            return AgentResponse(message=f"echo: {message}")
        async def reset(self) -> None:
            pass

    adapter = EchoAdapter()
    ctx = SessionContext(session_id="s1", turn_number=0, initial_state={}, current_state={}, history=[])
    resp = await adapter.send_message("hello", ctx)
    assert resp.message == "echo: hello"

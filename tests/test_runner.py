import pytest
from agenteval.models import AgentResponse, ToolCall, Scenario, Checkpoint
from agenteval.adapters.base import AgentAdapter, SessionContext


class MockAdapter(AgentAdapter):
    def __init__(self, responses):
        self._responses = responses
        self._index = 0

    async def send_message(self, message, context):
        if self._index < len(self._responses):
            resp = self._responses[self._index]
            self._index += 1
            return resp
        return AgentResponse(message="no more")

    async def reset(self):
        self._index = 0


@pytest.fixture
def scenario_2step():
    return Scenario(
        name="test", initial_state={"counter": 0},
        conversation_script=["do thing 1", "do thing 2"],
        checkpoints=[
            Checkpoint(id="step1", require={"tool_called": "action1"}),
            Checkpoint(id="step2", depends_on=["step1"], require={"tool_called": "action2"}),
        ],
        success="step2", expected_final_state={"counter": 2},
    )


@pytest.mark.asyncio
async def test_single_run(scenario_2step):
    from agenteval.runner import execute_run
    adapter = MockAdapter([
        AgentResponse(message="did 1", tool_calls=[ToolCall(name="action1", latency_ms=10)], state_changes={"counter": 1}),
        AgentResponse(message="did 2", tool_calls=[ToolCall(name="action2", latency_ms=10)], state_changes={"counter": 2}),
    ])
    run = await execute_run(adapter, scenario_2step, run_id="r1")
    assert run.success is True
    assert "step1" in run.checkpoints_reached
    assert "step2" in run.checkpoints_reached


@pytest.mark.asyncio
async def test_dag_blocks(scenario_2step):
    from agenteval.runner import execute_run
    adapter = MockAdapter([
        AgentResponse(message="skip"),
        AgentResponse(message="did 2", tool_calls=[ToolCall(name="action2", latency_ms=10)]),
    ])
    run = await execute_run(adapter, scenario_2step, run_id="r1")
    assert run.success is False
    assert "step2" not in run.checkpoints_reached


@pytest.mark.asyncio
async def test_run_k(scenario_2step):
    from agenteval.runner import run_scenarios
    adapter = MockAdapter([
        AgentResponse(message="did 1", tool_calls=[ToolCall(name="action1", latency_ms=10)], state_changes={"counter": 1}),
        AgentResponse(message="did 2", tool_calls=[ToolCall(name="action2", latency_ms=10)], state_changes={"counter": 2}),
    ])
    result = await run_scenarios(adapter, scenario_2step, k=1)
    assert result.k == 1
    assert result.pass_k == 1.0

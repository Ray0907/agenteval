import pytest


def test_tool_call_creation():
    from agenteval.models import ToolCall
    tc = ToolCall(name="lookup_order", arguments={"order_id": "123"}, result={"status": "shipped"}, latency_ms=45.2)
    assert tc.name == "lookup_order"
    assert tc.arguments == {"order_id": "123"}
    assert tc.result == {"status": "shipped"}
    assert tc.latency_ms == 45.2


def test_agent_response_creation():
    from agenteval.models import AgentResponse, ToolCall
    tc = ToolCall(name="search", arguments={"q": "test"}, result="found", latency_ms=10.0)
    resp = AgentResponse(
        message="Here are the results",
        tool_calls=[tc],
        state_changes={"cart": ["item1"]},
        metadata={"model": "gpt-4", "tokens": 150},
    )
    assert resp.message == "Here are the results"
    assert len(resp.tool_calls) == 1
    assert resp.state_changes == {"cart": ["item1"]}


def test_agent_response_defaults():
    from agenteval.models import AgentResponse
    resp = AgentResponse(message="hello")
    assert resp.tool_calls == []
    assert resp.state_changes == {}
    assert resp.metadata == {}


def test_turn_creation():
    from agenteval.models import Turn, AgentResponse
    resp = AgentResponse(message="I can help")
    turn = Turn(turn_id=1, user_message="Help me", agent_response=resp,
                elapsed_checkpoints=["greet"], cumulative_state={"greeted": True})
    assert turn.turn_id == 1
    assert turn.user_message == "Help me"
    assert turn.elapsed_checkpoints == ["greet"]


def test_run_creation():
    from agenteval.models import Run
    run = Run(run_id="run-001", scenario="refund_request", turns=[],
              final_state={"order_status": "refunded"}, checkpoints_reached=["verify", "refund"],
              success=True, total_tokens=500, total_cost=0.01, total_latency_ms=2300.0)
    assert run.run_id == "run-001"
    assert run.success is True
    assert run.total_cost == 0.01


def test_eval_result_creation():
    from agenteval.models import EvalResult
    result = EvalResult(project="my_agent", scenario="refund_request", k=5, runs=[],
                        pass_k=0.8, state_correctness=0.9, checkpoint_completion=1.0,
                        tool_accuracy=0.95, forbidden_tool_violations=0,
                        avg_turns=4.2, avg_tokens=600, avg_cost=0.015, avg_latency_ms=3000.0)
    assert result.pass_k == 0.8
    assert result.k == 5


def test_checkpoint_creation():
    from agenteval.models import Checkpoint
    cp = Checkpoint(id="verify_customer", depends_on=["greet"],
                    require={"tool_called": "verify_customer", "tool_args": {"customer_id": "c1"}})
    assert cp.id == "verify_customer"
    assert cp.depends_on == ["greet"]
    assert cp.require["tool_called"] == "verify_customer"


def test_scenario_creation():
    from agenteval.models import Scenario, Checkpoint
    cp = Checkpoint(id="done", depends_on=[], require={"tool_called": "finish"})
    scenario = Scenario(name="test_scenario", initial_state={"counter": 0},
                        conversation_script=["hello", "do the thing"],
                        checkpoints=[cp], success="done", expected_final_state={"counter": 1})
    assert scenario.name == "test_scenario"
    assert len(scenario.conversation_script) == 2
    assert scenario.success == "done"

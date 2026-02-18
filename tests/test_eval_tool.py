import pytest
from agenteval.models import Run, Turn, AgentResponse, ToolCall, Scenario, Checkpoint


def _make_run(tool_names, run_id="r1"):
    turns = []
    for i, name in enumerate(tool_names):
        tc = ToolCall(name=name, arguments={}, result="ok", latency_ms=10.0)
        turns.append(Turn(turn_id=i, user_message="go", agent_response=AgentResponse(message="done", tool_calls=[tc])))
    return Run(run_id=run_id, scenario="test", turns=turns)


@pytest.fixture
def scenario_tools():
    return Scenario(
        name="test", initial_state={}, conversation_script=["hi"],
        checkpoints=[Checkpoint(id="done", require={"tool_called": "finish"})],
        success="done", expected_final_state={},
        expected_tools={"required": ["lookup_order", "process_refund"], "forbidden": ["delete_account"]},
    )


def test_all_required(scenario_tools):
    from agenteval.evaluators.tool_accuracy import ToolAccuracyEvaluator
    run = _make_run(["lookup_order", "process_refund", "notify"])
    result = ToolAccuracyEvaluator().evaluate([run], scenario_tools)
    assert result["required_tools_score"] == 1.0


def test_missing_required(scenario_tools):
    from agenteval.evaluators.tool_accuracy import ToolAccuracyEvaluator
    run = _make_run(["lookup_order"])
    result = ToolAccuracyEvaluator().evaluate([run], scenario_tools)
    assert result["required_tools_score"] == 0.5


def test_forbidden_called(scenario_tools):
    from agenteval.evaluators.tool_accuracy import ToolAccuracyEvaluator
    run = _make_run(["lookup_order", "process_refund", "delete_account"])
    result = ToolAccuracyEvaluator().evaluate([run], scenario_tools)
    assert result["forbidden_violations"] == 1


def test_no_violations(scenario_tools):
    from agenteval.evaluators.tool_accuracy import ToolAccuracyEvaluator
    run = _make_run(["lookup_order", "process_refund"])
    result = ToolAccuracyEvaluator().evaluate([run], scenario_tools)
    assert result["forbidden_violations"] == 0

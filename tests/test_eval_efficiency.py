import pytest
from agenteval.models import Run, Turn, AgentResponse, Scenario, Checkpoint


def _make_run(num_turns, tokens, cost, latency, run_id="r1"):
    turns = [Turn(turn_id=i, user_message="go", agent_response=AgentResponse(message="ok")) for i in range(num_turns)]
    return Run(run_id=run_id, scenario="test", turns=turns,
               total_tokens=tokens, total_cost=cost, total_latency_ms=latency)


@pytest.fixture
def scenario():
    return Scenario(
        name="test", initial_state={}, conversation_script=["a", "b", "c"],
        checkpoints=[Checkpoint(id="done", require={"tool_called": "x"})],
        success="done", expected_final_state={},
        constraints={"max_turns": 5, "max_cost": 0.05},
    )


def test_basic(scenario):
    from agenteval.evaluators.efficiency import EfficiencyEvaluator
    result = EfficiencyEvaluator().evaluate([_make_run(3, 500, 0.01, 2000.0)], scenario)
    assert result["avg_turns"] == 3.0
    assert result["avg_tokens"] == 500
    assert result["avg_cost"] == 0.01


def test_multiple_runs(scenario):
    from agenteval.evaluators.efficiency import EfficiencyEvaluator
    runs = [_make_run(2, 400, 0.01, 1000.0, "r1"), _make_run(4, 600, 0.02, 3000.0, "r2")]
    result = EfficiencyEvaluator().evaluate(runs, scenario)
    assert result["avg_turns"] == 3.0
    assert result["avg_tokens"] == 500
    assert result["avg_cost"] == 0.015


def test_constraint_violations(scenario):
    from agenteval.evaluators.efficiency import EfficiencyEvaluator
    result = EfficiencyEvaluator().evaluate([_make_run(8, 500, 0.10, 2000.0)], scenario)
    assert "max_turns" in result["constraint_violations"]
    assert "max_cost" in result["constraint_violations"]

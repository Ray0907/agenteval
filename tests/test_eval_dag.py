import pytest
from agenteval.models import Run, Scenario, Checkpoint


@pytest.fixture
def scenario_4cp():
    return Scenario(
        name="test_dag", initial_state={}, conversation_script=["hi"],
        checkpoints=[
            Checkpoint(id="a", require={"tool_called": "a"}),
            Checkpoint(id="b", depends_on=["a"], require={"tool_called": "b"}),
            Checkpoint(id="c", depends_on=["a"], require={"tool_called": "c"}),
            Checkpoint(id="d", depends_on=["b", "c"], require={"tool_called": "d"}),
        ],
        success="d", expected_final_state={},
    )


def test_full_completion(scenario_4cp):
    from agenteval.evaluators.dag import DagProgressEvaluator
    runs = [Run(run_id="r1", scenario="test_dag", checkpoints_reached=["a", "b", "c", "d"], success=True)]
    assert DagProgressEvaluator().evaluate(runs, scenario_4cp) == 1.0


def test_partial_completion(scenario_4cp):
    from agenteval.evaluators.dag import DagProgressEvaluator
    runs = [Run(run_id="r1", scenario="test_dag", checkpoints_reached=["a", "b"], success=False)]
    assert DagProgressEvaluator().evaluate(runs, scenario_4cp) == 0.5


def test_no_completion(scenario_4cp):
    from agenteval.evaluators.dag import DagProgressEvaluator
    runs = [Run(run_id="r1", scenario="test_dag", checkpoints_reached=[], success=False)]
    assert DagProgressEvaluator().evaluate(runs, scenario_4cp) == 0.0


def test_avg_across_runs(scenario_4cp):
    from agenteval.evaluators.dag import DagProgressEvaluator
    runs = [
        Run(run_id="r1", scenario="test_dag", checkpoints_reached=["a", "b", "c", "d"], success=True),
        Run(run_id="r2", scenario="test_dag", checkpoints_reached=["a", "b"], success=False),
    ]
    assert DagProgressEvaluator().evaluate(runs, scenario_4cp) == 0.75


def test_blocking_info(scenario_4cp):
    from agenteval.evaluators.dag import get_blocking_info
    info = get_blocking_info(scenario_4cp, ["a"])
    assert "b" in info["unblocked"]
    assert "c" in info["unblocked"]
    assert "d" in info["blocked"]

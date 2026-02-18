import pytest
from agenteval.models import Run, Scenario, Checkpoint


def test_exact_match():
    from agenteval.evaluators.state import compare_field
    assert compare_field({"exact": "refunded"}, "refunded") is True
    assert compare_field({"exact": "refunded"}, "delivered") is False


def test_exists_match():
    from agenteval.evaluators.state import compare_field
    assert compare_field({"exists": True}, "anything") is True
    assert compare_field({"exists": True}, None) is False


def test_contains_match():
    from agenteval.evaluators.state import compare_field
    assert compare_field({"contains": "fund"}, "refunded") is True
    assert compare_field({"contains": "ship"}, "refunded") is False


def test_regex_match():
    from agenteval.evaluators.state import compare_field
    assert compare_field({"regex": r"^ref\w+ed$"}, "refunded") is True
    assert compare_field({"regex": r"^ship"}, "refunded") is False


def test_plain_value_exact():
    from agenteval.evaluators.state import compare_field
    assert compare_field("refunded", "refunded") is True
    assert compare_field(42, 42) is True
    assert compare_field("refunded", "delivered") is False


def test_compare_state_flat():
    from agenteval.evaluators.state import compare_state
    expected = {"status": {"exact": "refunded"}, "amount": 50}
    actual = {"status": "refunded", "amount": 50, "extra": "ignored"}
    result = compare_state(expected, actual)
    assert result.match is True
    assert result.correctness == 1.0


def test_compare_state_partial():
    from agenteval.evaluators.state import compare_state
    expected = {"status": {"exact": "refunded"}, "amount": 100}
    actual = {"status": "refunded", "amount": 50}
    result = compare_state(expected, actual)
    assert result.match is False
    assert result.correctness == 0.5


def test_compare_state_nested():
    from agenteval.evaluators.state import compare_state
    expected = {"orders": [{"id": "o1", "status": {"exact": "refunded"}}]}
    actual = {"orders": [{"id": "o1", "status": "refunded"}]}
    result = compare_state(expected, actual)
    assert result.match is True


def test_state_evaluator():
    from agenteval.evaluators.state import StateEvaluator
    scenario = Scenario(
        name="test", initial_state={}, conversation_script=["hi"],
        checkpoints=[Checkpoint(id="done", require={"tool_called": "x"})],
        success="done", expected_final_state={"status": {"exact": "refunded"}},
    )
    runs = [
        Run(run_id="r1", scenario="test", final_state={"status": "refunded"}, success=True),
        Run(run_id="r2", scenario="test", final_state={"status": "delivered"}, success=False),
    ]
    assert StateEvaluator().evaluate(runs, scenario) == 0.5

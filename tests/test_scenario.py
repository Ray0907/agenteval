import pytest
import yaml


@pytest.fixture
def valid_scenario_yaml(tmp_path):
    data = {
        "name": "refund_request",
        "initial_state": {"orders": [{"id": "o1", "status": "delivered", "amount": 50}]},
        "conversation_script": ["I want a refund for order o1", "Yes, please process it"],
        "checkpoints": [
            {"id": "order_found", "require": {"tool_called": "lookup_order"}},
            {"id": "refund_processed", "depends_on": ["order_found"],
             "require": {"tool_called": "process_refund", "tool_args": {"order_id": "o1"}}},
        ],
        "success": "refund_processed",
        "expected_final_state": {"orders": [{"id": "o1", "status": "refunded", "amount": 50}]},
    }
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.dump(data))
    return str(path)


@pytest.fixture
def cyclic_scenario_yaml(tmp_path):
    data = {
        "name": "cyclic", "initial_state": {}, "conversation_script": ["hello"],
        "checkpoints": [
            {"id": "a", "depends_on": ["b"], "require": {"tool_called": "a"}},
            {"id": "b", "depends_on": ["a"], "require": {"tool_called": "b"}},
        ],
        "success": "a", "expected_final_state": {},
    }
    path = tmp_path / "cyclic.yaml"
    path.write_text(yaml.dump(data))
    return str(path)


def test_load_scenario(valid_scenario_yaml):
    from agenteval.scenario import load_scenario
    scenario = load_scenario(valid_scenario_yaml)
    assert scenario.name == "refund_request"
    assert len(scenario.checkpoints) == 2
    assert len(scenario.conversation_script) == 2


def test_validate_dag_valid(valid_scenario_yaml):
    from agenteval.scenario import load_scenario, validate_dag
    scenario = load_scenario(valid_scenario_yaml)
    validate_dag(scenario)


def test_validate_dag_cyclic(cyclic_scenario_yaml):
    from agenteval.scenario import load_scenario, validate_dag
    scenario = load_scenario(cyclic_scenario_yaml)
    with pytest.raises(ValueError, match="cycle"):
        validate_dag(scenario)


def test_validate_dag_missing_dependency(tmp_path):
    data = {"name": "x", "initial_state": {}, "conversation_script": ["hi"],
            "checkpoints": [{"id": "a", "depends_on": ["nonexistent"], "require": {"tool_called": "a"}}],
            "success": "a", "expected_final_state": {}}
    (tmp_path / "m.yaml").write_text(yaml.dump(data))
    from agenteval.scenario import load_scenario, validate_dag
    with pytest.raises(ValueError, match="nonexistent"):
        validate_dag(load_scenario(str(tmp_path / "m.yaml")))


def test_validate_success_checkpoint(tmp_path):
    data = {"name": "x", "initial_state": {}, "conversation_script": ["hi"],
            "checkpoints": [{"id": "a", "require": {"tool_called": "a"}}],
            "success": "nonexistent", "expected_final_state": {}}
    (tmp_path / "b.yaml").write_text(yaml.dump(data))
    from agenteval.scenario import load_scenario, validate_dag
    with pytest.raises(ValueError, match="success.*nonexistent"):
        validate_dag(load_scenario(str(tmp_path / "b.yaml")))


def test_load_scenario_dir(tmp_path):
    for name in ["a.yaml", "b.yaml"]:
        data = {"name": name.replace(".yaml", ""), "initial_state": {},
                "conversation_script": ["hi"],
                "checkpoints": [{"id": "done", "require": {"tool_called": "x"}}],
                "success": "done", "expected_final_state": {}}
        (tmp_path / name).write_text(yaml.dump(data))
    from agenteval.scenario import load_scenarios_from_dir
    scenarios = load_scenarios_from_dir(str(tmp_path))
    assert len(scenarios) == 2
    assert {s.name for s in scenarios} == {"a", "b"}

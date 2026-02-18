import pytest, json
from agenteval.models import EvalResult


@pytest.fixture
def sample():
    return EvalResult(project="my_agent", scenario="refund", k=3,
                      pass_k=0.67, state_correctness=0.8, checkpoint_completion=1.0,
                      tool_accuracy=0.95, avg_turns=3.5, avg_tokens=500,
                      avg_cost=0.015, avg_latency_ms=2100.0)


def test_json(sample):
    from agenteval.report import generate_json_report
    data = json.loads(generate_json_report([sample]))
    assert data[0]["scenario"] == "refund"
    assert data[0]["pass_k"] == 0.67


def test_table(sample):
    from agenteval.report import generate_table_report
    out = generate_table_report([sample])
    assert "refund" in out


def test_html(sample):
    from agenteval.report import generate_html_report
    out = generate_html_report([sample])
    assert "<html" in out.lower()
    assert "refund" in out

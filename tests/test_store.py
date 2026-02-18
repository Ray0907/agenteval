import pytest
from agenteval.models import Run, EvalResult


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test.db")


@pytest.mark.asyncio
async def test_init_db(db_path):
    from agenteval.store import Store
    store = Store(db_path)
    await store.init()
    await store.close()


@pytest.mark.asyncio
async def test_save_and_load_run(db_path):
    from agenteval.store import Store
    store = Store(db_path)
    await store.init()
    run = Run(run_id="r1", scenario="refund", success=True,
              total_tokens=500, total_cost=0.01, total_latency_ms=2000.0,
              checkpoints_reached=["a", "b"])
    await store.save_run(run, project="proj")
    runs = await store.load_runs(project="proj", scenario="refund")
    assert len(runs) == 1
    assert runs[0]["run_id"] == "r1"
    assert runs[0]["success"] is True
    await store.close()


@pytest.mark.asyncio
async def test_save_and_load_result(db_path):
    from agenteval.store import Store
    store = Store(db_path)
    await store.init()
    result = EvalResult(
        project="proj", scenario="refund", k=3,
        pass_k=0.67, state_correctness=0.8, checkpoint_completion=1.0,
        tool_accuracy=0.95, forbidden_tool_violations=0,
        avg_turns=3.5, avg_tokens=500, avg_cost=0.01, avg_latency_ms=2000.0)
    await store.save_result(result)
    results = await store.load_results(project="proj")
    assert len(results) == 1
    assert results[0]["pass_k"] == 0.67
    await store.close()

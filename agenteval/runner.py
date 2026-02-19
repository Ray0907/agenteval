"""Async execution engine."""
from __future__ import annotations

import copy
import uuid
from typing import Any

from agenteval.adapters.base import AgentAdapter, SessionContext
from agenteval.evaluators.dag import DagProgressEvaluator
from agenteval.evaluators.efficiency import EfficiencyEvaluator
from agenteval.evaluators.state import StateEvaluator, compare_state
from agenteval.evaluators.tool_accuracy import ToolAccuracyEvaluator
from agenteval.models import Checkpoint, EvalResult, Run, Scenario, Turn


def _checkpoint_satisfied(
    checkpoint: Checkpoint,
    tool_names: list[str],
    tool_args_by_name: dict[str, dict[str, Any]],
) -> bool:
    """Check whether a checkpoint's requirements are met by the current turn's tool calls."""
    require = checkpoint.require
    if "tool_called" in require and require["tool_called"] not in tool_names:
        return False
    if "tool_args" in require:
        args = tool_args_by_name.get(require.get("tool_called", ""), {})
        if any(args.get(k) != v for k, v in require["tool_args"].items()):
            return False
    return True


def _newly_reachable(
    scenario: Scenario,
    reached: set[str],
    tool_names: list[str],
    tool_args_by_name: dict[str, dict[str, Any]],
) -> list[str]:
    """Return checkpoint IDs that are newly reachable after this turn."""
    newly_reached = []
    for cp in scenario.checkpoints:
        if cp.id in reached:
            continue
        if not all(dep in reached for dep in cp.depends_on):
            continue
        if _checkpoint_satisfied(cp, tool_names, tool_args_by_name):
            newly_reached.append(cp.id)
    return newly_reached


async def execute_run(
    adapter: AgentAdapter,
    scenario: Scenario,
    run_id: str | None = None,
) -> Run:
    """Execute a single run of a scenario against an adapter."""
    run_id = run_id or str(uuid.uuid4())[:8]
    state = copy.deepcopy(scenario.initial_state)
    turns: list[Turn] = []
    reached: set[str] = set()
    total_tokens = 0
    total_cost = 0.0
    total_latency_ms = 0.0

    for i, message in enumerate(scenario.conversation_script):
        ctx = SessionContext(
            session_id=run_id,
            turn_number=i,
            initial_state=scenario.initial_state,
            current_state=state,
            history=turns,
        )
        response = await adapter.send_message(message, ctx)

        if response.state_changes:
            state.update(response.state_changes)

        tool_names = [tc.name for tc in response.tool_calls]
        tool_args_by_name = {tc.name: tc.arguments for tc in response.tool_calls}
        new_checkpoints = _newly_reachable(scenario, reached, tool_names, tool_args_by_name)
        reached.update(new_checkpoints)

        total_latency_ms += sum(tc.latency_ms for tc in response.tool_calls)
        total_tokens += response.metadata.get("tokens", 0)
        total_cost += response.metadata.get("cost", 0.0)

        turns.append(Turn(
            turn_id=i,
            user_message=message,
            agent_response=response,
            elapsed_checkpoints=list(new_checkpoints),
            cumulative_state=copy.deepcopy(state),
        ))

        if scenario.success in reached:
            break

    return Run(
        run_id=run_id,
        scenario=scenario.name,
        turns=turns,
        final_state=state,
        checkpoints_reached=list(reached),
        success=scenario.success in reached,
        total_tokens=total_tokens,
        total_cost=total_cost,
        total_latency_ms=total_latency_ms,
    )


async def run_scenarios(
    adapter: AgentAdapter,
    scenario: Scenario,
    k: int = 3,
    project: str = "default",
) -> EvalResult:
    """Run a scenario k times and aggregate evaluation results."""
    runs: list[Run] = []
    for i in range(k):
        await adapter.reset()
        runs.append(await execute_run(adapter, scenario, run_id=f"{scenario.name}-{i}"))

    state_score = StateEvaluator().evaluate(runs, scenario)
    dag_score = DagProgressEvaluator().evaluate(runs, scenario)
    tool_result = ToolAccuracyEvaluator().evaluate(runs, scenario)
    efficiency = EfficiencyEvaluator().evaluate(runs, scenario)
    pass_count = sum(
        1 for r in runs
        if r.success and compare_state(scenario.expected_final_state, r.final_state).match
    )

    return EvalResult(
        project=project,
        scenario=scenario.name,
        k=k,
        runs=runs,
        pass_k=pass_count / k if k else 0.0,
        state_correctness=state_score,
        checkpoint_completion=dag_score,
        tool_accuracy=tool_result["required_tools_score"],
        forbidden_tool_violations=tool_result["forbidden_violations"],
        avg_turns=efficiency["avg_turns"],
        avg_tokens=efficiency["avg_tokens"],
        avg_cost=efficiency["avg_cost"],
        avg_latency_ms=efficiency["avg_latency_ms"],
    )

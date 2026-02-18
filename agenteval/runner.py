"""Async execution engine."""
from __future__ import annotations
import copy, uuid
from agenteval.adapters.base import AgentAdapter, SessionContext
from agenteval.evaluators.dag import DagProgressEvaluator
from agenteval.evaluators.efficiency import EfficiencyEvaluator
from agenteval.evaluators.state import StateEvaluator, compare_state
from agenteval.evaluators.tool_accuracy import ToolAccuracyEvaluator
from agenteval.models import EvalResult, Run, Scenario, Turn


def _check_cp(cp, tool_names, tool_args_map):
    req = cp.require
    if "tool_called" in req and req["tool_called"] not in tool_names:
        return False
    if "tool_args" in req:
        args = tool_args_map.get(req.get("tool_called", ""), {})
        if any(args.get(k) != v for k, v in req["tool_args"].items()):
            return False
    return True


def _reachable(scenario, reached, tool_names, tool_args_map):
    new = []
    for cp in scenario.checkpoints:
        if cp.id in reached:
            continue
        if not all(d in reached for d in cp.depends_on):
            continue
        if _check_cp(cp, tool_names, tool_args_map):
            new.append(cp.id)
    return new


async def execute_run(adapter, scenario, run_id=None):
    run_id = run_id or str(uuid.uuid4())[:8]
    state = copy.deepcopy(scenario.initial_state)
    history, reached = [], set()
    ttok, tcost, tlat = 0, 0.0, 0.0

    for i, msg in enumerate(scenario.conversation_script):
        ctx = SessionContext(session_id=run_id, turn_number=i,
                            initial_state=scenario.initial_state, current_state=state, history=history)
        resp = await adapter.send_message(msg, ctx)
        if resp.state_changes:
            state.update(resp.state_changes)
        tnames = [tc.name for tc in resp.tool_calls]
        targs = {tc.name: tc.arguments for tc in resp.tool_calls}
        new_cps = _reachable(scenario, reached, tnames, targs)
        reached.update(new_cps)
        for tc in resp.tool_calls:
            tlat += tc.latency_ms
        ttok += resp.metadata.get("tokens", 0)
        tcost += resp.metadata.get("cost", 0.0)
        history.append(Turn(turn_id=i, user_message=msg, agent_response=resp,
                            elapsed_checkpoints=list(new_cps), cumulative_state=copy.deepcopy(state)))
        if scenario.success in reached:
            break

    return Run(run_id=run_id, scenario=scenario.name, turns=history, final_state=state,
               checkpoints_reached=list(reached), success=scenario.success in reached,
               total_tokens=ttok, total_cost=tcost, total_latency_ms=tlat)


async def run_scenarios(adapter, scenario, k=3, project="default"):
    runs = []
    for i in range(k):
        await adapter.reset()
        runs.append(await execute_run(adapter, scenario, run_id=f"{scenario.name}-{i}"))

    ss = StateEvaluator().evaluate(runs, scenario)
    ds = DagProgressEvaluator().evaluate(runs, scenario)
    tr = ToolAccuracyEvaluator().evaluate(runs, scenario)
    er = EfficiencyEvaluator().evaluate(runs, scenario)
    ok = sum(1 for r in runs if r.success and compare_state(scenario.expected_final_state, r.final_state).match)

    return EvalResult(
        project=project, scenario=scenario.name, k=k, runs=runs,
        pass_k=ok / k if k else 0.0, state_correctness=ss, checkpoint_completion=ds,
        tool_accuracy=tr["required_tools_score"], forbidden_tool_violations=tr["forbidden_violations"],
        avg_turns=er["avg_turns"], avg_tokens=er["avg_tokens"],
        avg_cost=er["avg_cost"], avg_latency_ms=er["avg_latency_ms"])

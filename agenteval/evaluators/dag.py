"""DAG progress evaluator."""
from __future__ import annotations
from agenteval.evaluators.base import BaseEvaluator
from agenteval.models import Run, Scenario


def get_blocking_info(scenario: Scenario, reached: list[str]) -> dict[str, list[str]]:
    reached_set = set(reached)
    blocked, unblocked = [], []
    for cp in scenario.checkpoints:
        if cp.id in reached_set:
            continue
        if any(d not in reached_set for d in cp.depends_on):
            blocked.append(cp.id)
        else:
            unblocked.append(cp.id)
    return {"blocked": blocked, "unblocked": unblocked}


class DagProgressEvaluator(BaseEvaluator):
    def evaluate(self, runs: list[Run], scenario: Scenario) -> float:
        if not runs or not scenario.checkpoints:
            return 0.0
        total = len(scenario.checkpoints)
        return sum(len(r.checkpoints_reached) / total for r in runs) / len(runs)

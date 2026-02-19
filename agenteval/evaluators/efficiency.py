"""Efficiency metrics evaluator."""
from __future__ import annotations

from typing import Any

from agenteval.evaluators.base import BaseEvaluator
from agenteval.models import Run, Scenario

_EMPTY_RESULT: dict[str, Any] = {
    "avg_turns": 0.0,
    "avg_tokens": 0,
    "avg_cost": 0.0,
    "avg_latency_ms": 0.0,
    "constraint_violations": [],
}

_CONSTRAINT_CHECKS: list[tuple[str, str]] = [
    ("max_turns", "avg_turns"),
    ("max_cost", "avg_cost"),
    ("max_latency", "avg_latency_ms"),
]


class EfficiencyEvaluator(BaseEvaluator):
    def evaluate(self, runs: list[Run], scenario: Scenario) -> dict[str, Any]:
        if not runs:
            return dict(_EMPTY_RESULT)

        n = len(runs)
        avg_turns = sum(len(r.turns) for r in runs) / n
        avg_tokens = sum(r.total_tokens for r in runs) / n
        avg_cost = sum(r.total_cost for r in runs) / n
        avg_latency = sum(r.total_latency_ms for r in runs) / n

        metrics = {
            "avg_turns": avg_turns,
            "avg_tokens": int(avg_tokens),
            "avg_cost": avg_cost,
            "avg_latency_ms": avg_latency,
        }

        violations = [
            constraint_key
            for constraint_key, metric_key in _CONSTRAINT_CHECKS
            if scenario.constraints.get(constraint_key) and metrics[metric_key] > scenario.constraints[constraint_key]
        ]
        metrics["constraint_violations"] = violations
        return metrics

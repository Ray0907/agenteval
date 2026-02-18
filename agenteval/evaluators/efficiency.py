"""Efficiency metrics evaluator."""
from __future__ import annotations
from typing import Any
from agenteval.evaluators.base import BaseEvaluator
from agenteval.models import Run, Scenario


class EfficiencyEvaluator(BaseEvaluator):
    def evaluate(self, runs: list[Run], scenario: Scenario) -> dict[str, Any]:
        if not runs:
            return {"avg_turns": 0.0, "avg_tokens": 0, "avg_cost": 0.0,
                    "avg_latency_ms": 0.0, "constraint_violations": []}
        n = len(runs)
        avg_turns = sum(len(r.turns) for r in runs) / n
        avg_tokens = sum(r.total_tokens for r in runs) / n
        avg_cost = sum(r.total_cost for r in runs) / n
        avg_latency = sum(r.total_latency_ms for r in runs) / n

        violations = []
        c = scenario.constraints
        if c.get("max_turns") and avg_turns > c["max_turns"]:
            violations.append("max_turns")
        if c.get("max_cost") and avg_cost > c["max_cost"]:
            violations.append("max_cost")
        if c.get("max_latency") and avg_latency > c["max_latency"]:
            violations.append("max_latency")

        return {"avg_turns": avg_turns, "avg_tokens": int(avg_tokens),
                "avg_cost": avg_cost, "avg_latency_ms": avg_latency,
                "constraint_violations": violations}

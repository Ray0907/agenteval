"""State comparison evaluator."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from agenteval.evaluators.base import BaseEvaluator
from agenteval.models import Run, Scenario


@dataclass
class StateCompareResult:
    match: bool
    correctness: float
    field_results: dict[str, bool] = field(default_factory=dict)


def compare_field(expected: Any, actual: Any) -> bool:
    if isinstance(expected, dict):
        if "exact" in expected:
            return actual == expected["exact"]
        if "exists" in expected:
            return (actual is not None) == expected["exists"]
        if "contains" in expected:
            return isinstance(actual, str) and expected["contains"] in actual
        if "regex" in expected:
            return isinstance(actual, str) and bool(re.search(expected["regex"], actual))
        if actual is None or not isinstance(actual, dict):
            return False
        return all(compare_field(v, actual.get(k)) for k, v in expected.items())
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return False
        return all(compare_field(e, a) for e, a in zip(expected, actual))
    return expected == actual


def compare_state(expected: dict[str, Any], actual: dict[str, Any]) -> StateCompareResult:
    if not expected:
        return StateCompareResult(match=True, correctness=1.0)
    field_results = {}
    for key, exp in expected.items():
        field_results[key] = compare_field(exp, actual.get(key))
    matched = sum(field_results.values())
    total = len(field_results)
    return StateCompareResult(
        match=all(field_results.values()),
        correctness=matched / total if total else 1.0,
        field_results=field_results,
    )


class StateEvaluator(BaseEvaluator):
    def evaluate(self, runs: list[Run], scenario: Scenario) -> float:
        if not runs:
            return 0.0
        scores = []
        for run in runs:
            result = compare_state(scenario.expected_final_state, run.final_state)
            scores.append(1.0 if result.match else 0.0)
        return sum(scores) / len(scores)

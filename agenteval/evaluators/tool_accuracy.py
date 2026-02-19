"""Tool usage accuracy evaluator."""
from __future__ import annotations

from typing import Any

from agenteval.evaluators.base import BaseEvaluator
from agenteval.models import Run, Scenario


def _collect_tool_names(run: Run) -> set[str]:
    """Collect all unique tool names called across all turns in a run."""
    return {tc.name for turn in run.turns for tc in turn.agent_response.tool_calls}


class ToolAccuracyEvaluator(BaseEvaluator):
    def evaluate(self, runs: list[Run], scenario: Scenario) -> dict[str, Any]:
        required = scenario.expected_tools.get("required", [])
        forbidden = set(scenario.expected_tools.get("forbidden", []))
        total_req_score = 0.0
        total_violations = 0

        for run in runs:
            called = _collect_tool_names(run)
            if required:
                total_req_score += sum(1 for t in required if t in called) / len(required)
            else:
                total_req_score += 1.0
            total_violations += len(called & forbidden)

        n = len(runs) or 1
        return {"required_tools_score": total_req_score / n, "forbidden_violations": total_violations}

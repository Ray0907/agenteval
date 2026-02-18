"""Base evaluator interface."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from agenteval.models import Run, Scenario


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, runs: list[Run], scenario: Scenario) -> Any:
        ...

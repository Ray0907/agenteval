"""Scenario loading, validation, and DAG parsing."""
from __future__ import annotations

from collections import deque
from pathlib import Path

import yaml

from agenteval.models import Checkpoint, Scenario


def load_scenario(path: str) -> Scenario:
    """Load a single scenario from a YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    if "checkpoints" in data:
        data["checkpoints"] = [
            Checkpoint(**cp) if isinstance(cp, dict) else cp
            for cp in data["checkpoints"]
        ]
    return Scenario(**data)


def load_scenarios_from_dir(directory: str) -> list[Scenario]:
    """Load all scenarios from .yaml and .yml files in a directory."""
    dir_path = Path(directory)
    paths = sorted(dir_path.glob("*.yaml")) + sorted(dir_path.glob("*.yml"))
    return [load_scenario(str(p)) for p in paths]


def validate_dag(scenario: Scenario) -> None:
    """Validate that the checkpoint DAG is acyclic and well-formed."""
    checkpoint_ids = {cp.id for cp in scenario.checkpoints}

    if scenario.success not in checkpoint_ids:
        raise ValueError(
            f"success checkpoint '{scenario.success}' not found in checkpoints: {checkpoint_ids}"
        )

    for cp in scenario.checkpoints:
        for dep in cp.depends_on:
            if dep not in checkpoint_ids:
                raise ValueError(
                    f"Checkpoint '{cp.id}' depends on '{dep}' which does not exist"
                )

    # Topological sort to detect cycles
    in_degree: dict[str, int] = {cp.id: 0 for cp in scenario.checkpoints}
    adjacency: dict[str, list[str]] = {cp.id: [] for cp in scenario.checkpoints}
    for cp in scenario.checkpoints:
        for dep in cp.depends_on:
            adjacency[dep].append(cp.id)
            in_degree[cp.id] += 1

    queue = deque(node for node, degree in in_degree.items() if degree == 0)
    visited = 0
    while queue:
        node = queue.popleft()
        visited += 1
        for neighbor in adjacency[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if visited != len(checkpoint_ids):
        raise ValueError(
            f"Checkpoint DAG contains a cycle. Visited {visited}/{len(checkpoint_ids)} nodes."
        )

"""Scenario loading, validation, and DAG parsing."""
from __future__ import annotations
from pathlib import Path
import yaml
from agenteval.models import Checkpoint, Scenario


def load_scenario(path: str) -> Scenario:
    with open(path) as f:
        data = yaml.safe_load(f)
    if "checkpoints" in data:
        data["checkpoints"] = [Checkpoint(**cp) if isinstance(cp, dict) else cp for cp in data["checkpoints"]]
    return Scenario(**data)


def load_scenarios_from_dir(directory: str) -> list[Scenario]:
    scenarios = []
    dir_path = Path(directory)
    for path in sorted(dir_path.glob("*.yaml")):
        scenarios.append(load_scenario(str(path)))
    for path in sorted(dir_path.glob("*.yml")):
        scenarios.append(load_scenario(str(path)))
    return scenarios


def validate_dag(scenario: Scenario) -> None:
    checkpoint_ids = {cp.id for cp in scenario.checkpoints}
    if scenario.success not in checkpoint_ids:
        raise ValueError(f"success checkpoint '{scenario.success}' not found in checkpoints: {checkpoint_ids}")
    for cp in scenario.checkpoints:
        for dep in cp.depends_on:
            if dep not in checkpoint_ids:
                raise ValueError(f"Checkpoint '{cp.id}' depends on '{dep}' which does not exist")
    in_degree = {cp.id: 0 for cp in scenario.checkpoints}
    adjacency = {cp.id: [] for cp in scenario.checkpoints}
    for cp in scenario.checkpoints:
        for dep in cp.depends_on:
            adjacency[dep].append(cp.id)
            in_degree[cp.id] += 1
    queue = [n for n, d in in_degree.items() if d == 0]
    visited = 0
    while queue:
        node = queue.pop(0)
        visited += 1
        for nb in adjacency[node]:
            in_degree[nb] -= 1
            if in_degree[nb] == 0:
                queue.append(nb)
    if visited != len(checkpoint_ids):
        raise ValueError(f"Checkpoint DAG contains a cycle. Visited {visited}/{len(checkpoint_ids)} nodes.")

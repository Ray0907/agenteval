"""CLI entry point."""
from __future__ import annotations

import asyncio
import importlib
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console

from agenteval import __version__

app = typer.Typer(name="agenteval", help="Agent testing framework")
console = Console()


def _version_cb(value: bool) -> None:
    if value:
        console.print(f"agenteval {__version__}")
        raise typer.Exit()


@app.callback()
def main(version: bool = typer.Option(False, "--version", callback=_version_cb, is_eager=True)) -> None:
    pass


@app.command()
def init(project_path: str) -> None:
    """Scaffold a new project."""
    project_dir = Path(project_path)
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "scenarios").mkdir(exist_ok=True)

    config = {
        "project": project_dir.name,
        "agent": "agents.my_agent:MyAdapter",
        "scenarios": "scenarios/",
        "k": 3,
        "thresholds": {"min_pass_k": 0.8, "min_tool_accuracy": 0.9},
    }
    (project_dir / "agenteval.yaml").write_text(yaml.dump(config, default_flow_style=False))

    example = {
        "name": "example",
        "initial_state": {"x": 0},
        "conversation_script": ["hi"],
        "checkpoints": [{"id": "done", "require": {"tool_called": "greet"}}],
        "success": "done",
        "expected_final_state": {"greeted": True},
    }
    (project_dir / "scenarios" / "example.yaml").write_text(yaml.dump(example, default_flow_style=False))
    console.print(f"[green]Created agenteval project at {project_dir}[/green]")


@app.command()
def run(
    scenario: Optional[str] = typer.Argument(None),
    agent: str = typer.Option("", "--agent"),
    k: int = typer.Option(3, "--k"),
    config: str = typer.Option("agenteval.yaml", "--config"),
    project: str = typer.Option("default", "--project"),
    output: str = typer.Option("table", "--output"),
    ci: bool = typer.Option(False, "--ci"),
) -> None:
    """Run scenarios against an agent."""
    from agenteval.report import generate_html_report, generate_json_report, generate_table_report
    from agenteval.runner import run_scenarios
    from agenteval.scenario import load_scenario, load_scenarios_from_dir, validate_dag

    config_path = Path(config)
    cfg = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
    agent_path = agent or cfg.get("agent", "")
    if not agent_path:
        console.print("[red]--agent required[/red]")
        raise typer.Exit(1)

    module_path, class_name = agent_path.rsplit(":", 1)
    adapter = getattr(importlib.import_module(module_path), class_name)()
    scenario_path = Path(scenario or cfg.get("scenarios", "scenarios/"))
    scenarios = (
        load_scenarios_from_dir(str(scenario_path))
        if scenario_path.is_dir()
        else [load_scenario(str(scenario_path))]
    )

    results = []
    for sc in scenarios:
        validate_dag(sc)
        console.print(f"Running [cyan]{sc.name}[/cyan] (k={k})...")
        results.append(asyncio.run(run_scenarios(adapter, sc, k=k, project=project)))

    if output == "json":
        console.print(generate_json_report(results))
    elif output == "html":
        output_path = Path(f"{project}_report.html")
        output_path.write_text(generate_html_report(results))
        console.print(f"[green]Saved: {output_path}[/green]")
    else:
        console.print(generate_table_report(results))

    if ci and (thresholds := cfg.get("thresholds")):
        for r in results:
            if thresholds.get("min_pass_k") and r.pass_k < thresholds["min_pass_k"]:
                console.print(
                    f"[red]FAIL: {r.scenario} pass^k {r.pass_k:.2f} < {thresholds['min_pass_k']}[/red]"
                )
                raise typer.Exit(1)

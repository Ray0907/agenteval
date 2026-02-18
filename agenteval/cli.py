"""CLI entry point."""
from __future__ import annotations
import asyncio, importlib
from pathlib import Path
from typing import Optional
import typer, yaml
from rich.console import Console
from agenteval import __version__

app = typer.Typer(name="agenteval", help="Agent testing framework")
console = Console()


def _version_cb(value: bool):
    if value:
        console.print(f"agenteval {__version__}")
        raise typer.Exit()


@app.callback()
def main(version: bool = typer.Option(False, "--version", callback=_version_cb, is_eager=True)):
    pass


@app.command()
def init(project_path: str):
    """Scaffold a new project."""
    p = Path(project_path)
    p.mkdir(parents=True, exist_ok=True)
    (p / "scenarios").mkdir(exist_ok=True)
    (p / "agenteval.yaml").write_text(yaml.dump({
        "project": p.name, "agent": "agents.my_agent:MyAdapter",
        "scenarios": "scenarios/", "k": 3,
        "thresholds": {"min_pass_k": 0.8, "min_tool_accuracy": 0.9}
    }, default_flow_style=False))
    (p / "scenarios" / "example.yaml").write_text(yaml.dump({
        "name": "example", "initial_state": {"x": 0}, "conversation_script": ["hi"],
        "checkpoints": [{"id": "done", "require": {"tool_called": "greet"}}],
        "success": "done", "expected_final_state": {"greeted": True}
    }, default_flow_style=False))
    console.print(f"[green]Created agenteval project at {p}[/green]")


@app.command()
def run(
    scenario: Optional[str] = typer.Argument(None),
    agent: str = typer.Option("", "--agent"),
    k: int = typer.Option(3, "--k"),
    config: str = typer.Option("agenteval.yaml", "--config"),
    project: str = typer.Option("default", "--project"),
    output: str = typer.Option("table", "--output"),
    ci: bool = typer.Option(False, "--ci"),
):
    """Run scenarios against an agent."""
    from agenteval.report import generate_html_report, generate_json_report, generate_table_report
    from agenteval.runner import run_scenarios
    from agenteval.scenario import load_scenario, load_scenarios_from_dir, validate_dag

    cfg = yaml.safe_load(Path(config).read_text()) if Path(config).exists() else {}
    agent_path = agent or cfg.get("agent", "")
    if not agent_path:
        console.print("[red]--agent required[/red]")
        raise typer.Exit(1)

    mod, cls = agent_path.rsplit(":", 1)
    adapter = getattr(importlib.import_module(mod), cls)()
    sp = scenario or cfg.get("scenarios", "scenarios/")
    p = Path(sp)
    scns = load_scenarios_from_dir(str(p)) if p.is_dir() else [load_scenario(str(p))]

    results = []
    for sc in scns:
        validate_dag(sc)
        console.print(f"Running [cyan]{sc.name}[/cyan] (k={k})...")
        results.append(asyncio.run(run_scenarios(adapter, sc, k=k, project=project)))

    if output == "json":
        console.print(generate_json_report(results))
    elif output == "html":
        op = Path(f"{project}_report.html")
        op.write_text(generate_html_report(results))
        console.print(f"[green]Saved: {op}[/green]")
    else:
        console.print(generate_table_report(results))

    if ci and (t := cfg.get("thresholds")):
        for r in results:
            if t.get("min_pass_k") and r.pass_k < t["min_pass_k"]:
                console.print(f"[red]FAIL: {r.scenario} pass^k {r.pass_k:.2f} < {t['min_pass_k']}[/red]")
                raise typer.Exit(1)

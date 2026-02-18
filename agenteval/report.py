"""Report generation."""
from __future__ import annotations
import json
from rich.console import Console
from rich.table import Table
from agenteval.models import EvalResult


def generate_json_report(results: list[EvalResult]) -> str:
    return json.dumps([{
        "project": r.project, "scenario": r.scenario, "k": r.k,
        "pass_k": r.pass_k, "state_correctness": r.state_correctness,
        "checkpoint_completion": r.checkpoint_completion,
        "tool_accuracy": r.tool_accuracy,
        "forbidden_tool_violations": r.forbidden_tool_violations,
        "avg_turns": r.avg_turns, "avg_tokens": r.avg_tokens,
        "avg_cost": r.avg_cost, "avg_latency_ms": r.avg_latency_ms,
    } for r in results], indent=2, ensure_ascii=False)


def generate_table_report(results: list[EvalResult]) -> str:
    t = Table(title="agenteval Results")
    for col in ["Scenario", "k", "pass^k", "State", "Checkpoints", "Tool Acc", "Turns", "Cost"]:
        t.add_column(col, justify="right" if col != "Scenario" else "left")
    for r in results:
        t.add_row(r.scenario, str(r.k), f"{r.pass_k*100:.1f}%",
                  f"{r.state_correctness*100:.1f}%", f"{r.checkpoint_completion*100:.1f}%",
                  f"{r.tool_accuracy*100:.1f}%", f"{r.avg_turns:.1f}", f"${r.avg_cost:.4f}")
    c = Console(record=True, width=120)
    c.print(t)
    return c.export_text()


def generate_html_report(results: list[EvalResult]) -> str:
    rows = "".join(f"<tr><td>{r.scenario}</td><td>{r.k}</td><td>{r.pass_k*100:.1f}%</td>"
                   f"<td>{r.state_correctness*100:.1f}%</td><td>{r.checkpoint_completion*100:.1f}%</td>"
                   f"<td>{r.tool_accuracy*100:.1f}%</td><td>{r.avg_turns:.1f}</td>"
                   f"<td>${r.avg_cost:.4f}</td></tr>" for r in results)
    return (f'<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>agenteval</title>'
            f'<style>body{{font-family:sans-serif;max-width:1000px;margin:40px auto}}'
            f'table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ddd;padding:8px}}'
            f'th{{background:#1a1a2e;color:#fff}}</style></head><body><h1>agenteval Report</h1>'
            f'<table><thead><tr><th>Scenario</th><th>k</th><th>pass^k</th><th>State</th>'
            f'<th>Checkpoints</th><th>Tool</th><th>Turns</th><th>Cost</th></tr></thead>'
            f'<tbody>{rows}</tbody></table></body></html>')

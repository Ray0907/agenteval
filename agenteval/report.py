"""Report generation."""
from __future__ import annotations

import json

from rich.console import Console
from rich.table import Table

from agenteval.models import EvalResult

_REPORT_EXCLUDE = {"runs"}


def generate_json_report(results: list[EvalResult]) -> str:
    """Generate a JSON report from evaluation results."""
    return json.dumps(
        [r.model_dump(exclude=_REPORT_EXCLUDE) for r in results],
        indent=2,
        ensure_ascii=False,
    )


def generate_table_report(results: list[EvalResult]) -> str:
    """Generate a Rich table report from evaluation results."""
    table = Table(title="agenteval Results")
    for col in ["Scenario", "k", "pass^k", "State", "Checkpoints", "Tool Acc", "Turns", "Cost"]:
        table.add_column(col, justify="left" if col == "Scenario" else "right")
    for r in results:
        table.add_row(
            r.scenario,
            str(r.k),
            f"{r.pass_k * 100:.1f}%",
            f"{r.state_correctness * 100:.1f}%",
            f"{r.checkpoint_completion * 100:.1f}%",
            f"{r.tool_accuracy * 100:.1f}%",
            f"{r.avg_turns:.1f}",
            f"${r.avg_cost:.4f}",
        )
    console = Console(record=True, width=120)
    console.print(table)
    return console.export_text()


def generate_html_report(results: list[EvalResult]) -> str:
    """Generate an HTML report from evaluation results."""
    rows = "".join(
        f"<tr><td>{r.scenario}</td><td>{r.k}</td><td>{r.pass_k * 100:.1f}%</td>"
        f"<td>{r.state_correctness * 100:.1f}%</td><td>{r.checkpoint_completion * 100:.1f}%</td>"
        f"<td>{r.tool_accuracy * 100:.1f}%</td><td>{r.avg_turns:.1f}</td>"
        f"<td>${r.avg_cost:.4f}</td></tr>"
        for r in results
    )
    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>agenteval</title>'
        '<style>body{font-family:sans-serif;max-width:1000px;margin:40px auto}'
        'table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:8px}'
        'th{background:#1a1a2e;color:#fff}</style></head><body><h1>agenteval Report</h1>'
        '<table><thead><tr><th>Scenario</th><th>k</th><th>pass^k</th><th>State</th>'
        '<th>Checkpoints</th><th>Tool</th><th>Turns</th><th>Cost</th></tr></thead>'
        f'<tbody>{rows}</tbody></table></body></html>'
    )

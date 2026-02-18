"""Demo: run agenteval with Claude Sonnet 4.6.

Usage:
    export ANTHROPIC_API_KEY=sk-...
    cd /path/to/llm-eval
    source .venv/bin/activate
    python examples/demo.py
"""
from __future__ import annotations

import asyncio
import copy

from agenteval.adapters.llm import LLMAdapter
from agenteval.report import generate_table_report
from agenteval.runner import run_scenarios
from agenteval.scenario import load_scenario, validate_dag

# Mock database
DB = {
    "orders": {
        "o1": {"id": "o1", "status": "delivered", "amount": 49.99, "customer_id": "c1"},
    },
    "customers": {
        "c1": {"id": "c1", "name": "Alice", "email": "alice@example.com"},
    },
}


def tool_handler(name: str, args: dict):
    """Simulate tool execution against mock DB."""
    if name == "lookup_order":
        order = DB["orders"].get(args.get("order_id"))
        return {"found": True, **order} if order else {"found": False, "error": "Order not found"}

    if name == "verify_customer":
        cid = args.get("customer_id", "c1")
        customer = DB["customers"].get(cid)
        return {"verified": True, "customer": customer} if customer else {"verified": False}

    if name == "process_refund":
        oid = args.get("order_id")
        if oid in DB["orders"]:
            DB["orders"][oid]["status"] = "refunded"
            return {
                "success": True,
                "refunded_amount": DB["orders"][oid]["amount"],
                "state_changes": {"orders": [DB["orders"][oid]]},
            }
        return {"success": False, "error": "Order not found"}

    return {"error": f"Unknown tool: {name}"}


async def main():
    scenario = load_scenario("examples/scenarios/refund_request.yaml")
    validate_dag(scenario)

    adapter = LLMAdapter(
        settings="examples/demo_settings.yaml",
        tool_handler=tool_handler,
    )

    print(f"Running scenario: {scenario.name} (k=1, model=claude-sonnet-4-6)")
    print("=" * 60)

    result = await run_scenarios(adapter, scenario, k=1, project="demo")

    # Show conversation
    for run in result.runs:
        print(f"\n--- Run: {run.run_id} (success={run.success}) ---\n")
        for turn in run.turns:
            print(f"  User: {turn.user_message}")
            print(f"  Agent: {turn.agent_response.message[:200]}")
            for tc in turn.agent_response.tool_calls:
                print(f"    -> {tc.name}({tc.arguments}) = {tc.result}")
            print()

    # Show results
    print("=" * 60)
    print(generate_table_report([result]))
    print(f"pass^k: {result.pass_k}")
    print(f"State correctness: {result.state_correctness}")
    print(f"Checkpoints: {[cp for r in result.runs for cp in r.checkpoints_reached]}")


if __name__ == "__main__":
    asyncio.run(main())

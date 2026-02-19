# agenteval

Evaluation framework for multi-turn conversational AI agents. Define test scenarios as YAML, run them k times against your agent, and get pass^k scores with detailed metrics.

## Why

LLM agents are non-deterministic. A single test run doesn't tell you if your agent reliably handles a workflow. agenteval runs each scenario multiple times and measures:

- **pass^k** — success rate across k runs
- **State correctness** — does the final state match expectations?
- **Checkpoint completion** — how far through the task DAG did the agent get?
- **Tool accuracy** — did the agent call the right tools with the right arguments?
- **Efficiency** — turns, tokens, cost, latency

## Install

```bash
uv pip install -e .

# With LLM provider support
uv pip install -e ".[anthropic]"   # Anthropic Claude
uv pip install -e ".[openai]"     # OpenAI
uv pip install -e ".[all]"        # All providers
```

Requires Python 3.11+.

## Quick Start

### 1. Define a scenario

```yaml
# scenarios/refund_request.yaml
name: refund_request
description: Customer requests a refund for a delivered order

initial_state:
  orders:
    - id: "o1"
      status: delivered
      amount: 49.99

conversation_script:
  - "Hi, I'd like a refund for order o1"
  - "Yes, the item was damaged when it arrived"
  - "Yes please, process the refund"

checkpoints:
  - id: lookup_order
    require:
      tool_called: lookup_order
      tool_args:
        order_id: "o1"
  - id: verify_customer
    depends_on: [lookup_order]
    require:
      tool_called: verify_customer
  - id: process_refund
    depends_on: [verify_customer]
    require:
      tool_called: process_refund
      tool_args:
        order_id: "o1"

success: process_refund

expected_final_state:
  orders:
    - id: "o1"
      status: refunded

expected_tools:
  required: [lookup_order, verify_customer, process_refund]
  forbidden: [delete_account]

constraints:
  max_turns: 5
  max_cost: 0.10
```

### 2. Implement an adapter

```python
from agenteval import AgentAdapter, AgentResponse, SessionContext

class MyAgent(AgentAdapter):
    async def send_message(self, message: str, context: SessionContext) -> AgentResponse:
        # Call your agent here
        return AgentResponse(
            message="response text",
            tool_calls=[...],
            state_changes={...},
        )

    async def reset(self) -> None:
        # Reset agent state between runs
        pass
```

Or use the built-in `LLMAdapter` for Anthropic/OpenAI:

```python
from agenteval.adapters.llm import LLMAdapter

adapter = LLMAdapter(
    settings="settings.yaml",  # provider, model, system_prompt, tools
    tool_handler=my_tool_handler,
)
```

### 3. Run

```python
import asyncio
from agenteval import run_scenarios, load_scenario, validate_dag
from agenteval.report import generate_table_report

scenario = load_scenario("scenarios/refund_request.yaml")
validate_dag(scenario)

result = asyncio.run(run_scenarios(adapter, scenario, k=3))
print(generate_table_report([result]))
```

## CLI

```bash
# Scaffold a new project
agenteval init my_project

# Run scenarios
agenteval run --agent my_agent:MyAdapter --k 3

# Run a single scenario
agenteval run scenarios/refund.yaml --agent my_agent:MyAdapter

# JSON output
agenteval run --output json

# CI mode (fails if below thresholds)
agenteval run --ci --config agenteval.yaml
```

### Project config (`agenteval.yaml`)

```yaml
project: my_project
agent: my_agent:MyAdapter
scenarios: scenarios/
k: 3
thresholds:
  min_pass_k: 0.8
  min_tool_accuracy: 0.9
```

## Core Concepts

### Scenarios

A scenario defines a multi-turn conversation test case with:

| Field | Description |
|---|---|
| `conversation_script` | List of user messages sent in order |
| `initial_state` | State before the conversation starts |
| `checkpoints` | DAG of expected agent behaviors |
| `success` | Checkpoint ID that means the scenario passed |
| `expected_final_state` | State to verify after all turns |
| `expected_tools` | Required and forbidden tool lists |
| `constraints` | Limits on turns, cost, etc. |

### Checkpoint DAG

Checkpoints form a directed acyclic graph. Each checkpoint can:

- **depend on** other checkpoints (must be reached first)
- **require** a specific tool call with specific arguments

```
lookup_order → verify_customer → process_refund
```

The agent doesn't need to reach all checkpoints in one turn. The evaluator tracks progress across the entire conversation.

### State Comparison

Expected state supports flexible matching:

```yaml
expected_final_state:
  status: refunded              # exact match
  note: { contains: "approved" } # substring match
  code: { regex: "^R-\\d+" }    # regex match
  email: { exists: true }       # existence check
```

### Evaluators

| Evaluator | Measures |
|---|---|
| `StateEvaluator` | Final state matches expected state |
| `DagProgressEvaluator` | Fraction of checkpoints reached |
| `ToolAccuracyEvaluator` | Required tools called, forbidden tools avoided |
| `EfficiencyEvaluator` | Average turns, tokens, cost, latency |

## Integrations

### LangGraph

```python
from agenteval.integrations.langgraph import LangGraphAdapter

adapter = LangGraphAdapter(graph=my_compiled_graph)
```

### OpenAI Agents SDK

```python
from agenteval.integrations.openai_agents import OpenAIAgentAdapter

adapter = OpenAIAgentAdapter(agent=my_agent)
```

Install extras: `uv pip install -e ".[langgraph]"` or `uv pip install -e ".[openai]"`

## Reports

Three output formats:

```python
from agenteval.report import generate_table_report, generate_json_report, generate_html_report

generate_table_report([result])   # Rich table to terminal
generate_json_report([result])    # JSON string
generate_html_report([result])    # Standalone HTML file
```

## License

MIT

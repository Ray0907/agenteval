"""Core data models for agenteval."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """A single tool/function call made by the agent."""
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    latency_ms: float = 0.0


class AgentResponse(BaseModel):
    """The agent's response to a user message."""
    message: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    state_changes: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Turn(BaseModel):
    """A single conversation turn (user message + agent response)."""
    turn_id: int
    user_message: str
    agent_response: AgentResponse
    elapsed_checkpoints: list[str] = Field(default_factory=list)
    cumulative_state: dict[str, Any] = Field(default_factory=dict)


class Run(BaseModel):
    """A single execution run of a scenario."""
    run_id: str
    scenario: str
    turns: list[Turn] = Field(default_factory=list)
    final_state: dict[str, Any] = Field(default_factory=dict)
    checkpoints_reached: list[str] = Field(default_factory=list)
    success: bool = False
    total_tokens: int = 0
    total_cost: float = 0.0
    total_latency_ms: float = 0.0


class EvalResult(BaseModel):
    """Aggregated result across k runs."""
    project: str
    scenario: str
    k: int
    runs: list[Run] = Field(default_factory=list)
    pass_k: float = 0.0
    state_correctness: float = 0.0
    checkpoint_completion: float = 0.0
    tool_accuracy: float = 0.0
    forbidden_tool_violations: int = 0
    avg_turns: float = 0.0
    avg_tokens: int = 0
    avg_cost: float = 0.0
    avg_latency_ms: float = 0.0


class Checkpoint(BaseModel):
    """A DAG checkpoint in a scenario."""
    id: str
    depends_on: list[str] = Field(default_factory=list)
    require: dict[str, Any] = Field(default_factory=dict)


class Scenario(BaseModel):
    """A test scenario definition."""
    name: str
    initial_state: dict[str, Any] = Field(default_factory=dict)
    conversation_script: list[str] = Field(default_factory=list)
    checkpoints: list[Checkpoint] = Field(default_factory=list)
    success: str = ""
    expected_final_state: dict[str, Any] = Field(default_factory=dict)
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    expected_tools: dict[str, list[str]] = Field(default_factory=dict)
    constraints: dict[str, Any] = Field(default_factory=dict)

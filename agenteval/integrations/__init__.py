"""Framework integrations."""

def get_adapter(name: str):
    if name == "langgraph":
        from agenteval.integrations.langgraph import LangGraphAdapter
        return LangGraphAdapter
    elif name == "openai":
        from agenteval.integrations.openai_agents import OpenAIAgentAdapter
        return OpenAIAgentAdapter
    raise ValueError(f"Unknown: {name}. Available: langgraph, openai")

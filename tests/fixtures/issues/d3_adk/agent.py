"""ADK: sub_agents, AgentTool, a whole-run limit, and the id of one invocation."""

from google.adk.agents import Agent
from google.adk.agents.run_config import RunConfig
from google.adk.tools.agent_tool import AgentTool


def lookup(q: str) -> str:
    """Look up."""
    return q


def log_run(ctx) -> None:
    print("run", ctx.invocation_id)


researcher = Agent(name="researcher", model="gemini-2.0-flash-001", tools=[lookup])
writer = Agent(name="writer", model="gemini-2.0-flash-001")
root = Agent(
    name="root", model="gemini-2.0-flash-001", sub_agents=[writer], tools=[AgentTool(agent=researcher)],
    before_agent_callback=log_run,
)
CONFIG = RunConfig(max_llm_calls=30)

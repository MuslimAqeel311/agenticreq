"""Two ADK agents; RunConfig caps the LLM calls of the whole run."""

from google.adk.agents import Agent
from google.adk.agents.run_config import RunConfig
from google.adk.runners import Runner


def lookup(q: str) -> str:
    """Look up."""
    return q


specialist = Agent(name="specialist", model="gemini-2.0-flash-001", tools=[lookup])
root = Agent(name="root", model="gemini-2.0-flash-001", sub_agents=[specialist])
CONFIG = RunConfig(max_llm_calls=40)


def run(q: str) -> None:
    Runner(agent=root, app_name="a", session_service=None).run(user_id="u", session_id="s", new_message=q, run_config=CONFIG)

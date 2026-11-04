"""Limits from imported constants and a settings object; one limit has two different definitions."""

import config
from config import MAX_ITER, settings
from crewai import Agent as Crew
from agents import Agent, Runner, function_tool


@function_tool
def lookup(q: str) -> str:
    """Look up."""
    return q


crew = Crew(role="r", goal="g", backstory="b", max_iter=MAX_ITER, llm="gpt-4o-2024-08-06")
crew2 = Crew(role="s", goal="g", backstory="b", max_iter=config.MAX_ITER, llm="gpt-4o-2024-08-06")
agent = Agent(name="a", instructions="i", tools=[lookup], model="gpt-4o-2024-08-06")


def run(q: str) -> None:
    Runner.run_sync(agent, q, max_turns=settings.max_turns)

from agents import Agent, Runner, function_tool


@function_tool
def lookup(q: str) -> str:
    """Look something up."""
    return q


agent = Agent(name="a", model="gpt-4o-2024-08-06", tools=[lookup])

import os

if os.getenv("ENV") == "prod":
    MAX_TURNS = 6
else:
    MAX_TURNS = 9


def run(task):
    return Runner.run_sync(agent, task, max_turns=MAX_TURNS)

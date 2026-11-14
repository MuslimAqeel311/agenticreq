"""The agent this folder ships."""

from agents import Agent, Runner, function_tool

from helpers import clean


@function_tool
def lookup(q: str) -> str:
    """Look something up."""
    return clean(q)


agent = Agent(name="support", model="gpt-4o-2024-08-06", tools=[lookup])


def main(task: str) -> str:
    return str(Runner.run_sync(agent, task, max_turns=6).final_output)

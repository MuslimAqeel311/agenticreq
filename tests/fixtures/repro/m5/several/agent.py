import a_cfg, b_cfg
import random
from agents import Agent, Runner, function_tool


@function_tool
def lookup(q: str) -> str:
    """Look something up."""
    return q


agent = Agent(name="a", model="gpt-4o-2024-08-06", tools=[lookup])


def run(task):
    turns = a_cfg.TURNS if random.random() > 0.5 else b_cfg.TURNS
    return Runner.run_sync(agent, task, max_turns=turns)

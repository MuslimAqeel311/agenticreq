from agents import Agent, Runner, function_tool


@function_tool
def lookup(q: str) -> str:
    """Look something up."""
    return q


agent = Agent(name="a", model="gpt-4o-2024-08-06", tools=[lookup])


class Limits:
    TURNS = 6


def run(task):
    return Runner.run_sync(agent, task, max_turns=Limits.TURNS)

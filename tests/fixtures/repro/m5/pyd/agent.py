from pydantic_settings import BaseSettings
from agents import Agent, Runner, function_tool


@function_tool
def lookup(q: str) -> str:
    """Look something up."""
    return q


agent = Agent(name="a", model="gpt-4o-2024-08-06", tools=[lookup])


class Settings(BaseSettings):
    turns: int = 6


settings = Settings()


def run(task):
    return Runner.run_sync(agent, task, max_turns=settings.turns)

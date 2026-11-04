"""A tool named web_search in code; the provider serves one under the same name."""

from agents import Agent, function_tool


@function_tool
def web_search(q: str) -> str:
    """Search the web."""
    return "x"


agent = Agent(name="researcher", instructions="Find.", tools=[web_search], model="gpt-4o-2024-08-06")
